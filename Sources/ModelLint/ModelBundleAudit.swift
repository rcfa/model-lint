import Foundation

/// Decide what is wrong with a model bundle's weight files — and, crucially, what is NOT wrong.
///
/// ## Why a routine audit rather than one-off fixes
///
/// Converted MLX bundles arrive with real packaging defects, and every one found so far was invisible
/// until something happened to look:
///
/// - Three Nemotron bundles shipped TWO COMPLETE SHARDINGS of the same weights — 43 GB of duplicate
///   data — because a re-conversion wrote a new shard layout without removing the old one.
/// - Seven bundles ship an index that references files which do not exist, while their actual weight
///   files are referenced by nothing. They load here only because vmlx scans shards rather than
///   trusting the index; any index-trusting loader fails on them.
/// - A JANG conversion can drop `chat_template.jinja`, which leaves MC scoring fine and breaks
///   generation with `missingChatTemplate`.
///
/// ## THE RULE THAT KEEPS THIS SAFE
///
/// A file is removable ONLY if every tensor in it is reachable through the index. "Not referenced by
/// the index" is NOT sufficient and never will be: `jangtq_runtime.safetensors` is unreferenced in
/// five bundles and holds the ternary codebooks and sign tables that exist nowhere else, so a
/// delete-what-is-unindexed rule would silently destroy five models. The audit therefore compares
/// TENSOR NAMES, not file lists, and reports a unique-data orphan as something to KEEP.
///
/// Pure: it takes an already-read description of the bundle, so the decision logic is testable
/// without a model, a GPU, or a filesystem.
public enum ModelBundleAudit {

    /// What was read off disk. `tensorsByFile` maps a weight file's basename to the tensor names in
    /// its header — the header only, never the tensor data.
    public struct Bundle: Sendable, Equatable {
        public let files: [String]
        public let indexedFiles: Set<String>
        public let indexedTensors: Set<String>
        public let tensorsByFile: [String: Set<String>]
        /// Files the index names that are absent from disk.
        public let indexReferencesMissing: Set<String>
        public let hasIndex: Bool

        public init(files: [String], indexedFiles: Set<String>, indexedTensors: Set<String>,
                    tensorsByFile: [String: Set<String>], indexReferencesMissing: Set<String>,
                    hasIndex: Bool) {
            self.files = files
            self.indexedFiles = indexedFiles
            self.indexedTensors = indexedTensors
            self.tensorsByFile = tensorsByFile
            self.indexReferencesMissing = indexReferencesMissing
            self.hasIndex = hasIndex
        }
    }

    /// WHAT KIND OF BUNDLE THIS IS — the gate that decides which checks even apply.
    ///
    /// Without it, a "fill in the missing sampling defaults" pass would have injected temperature and
    /// top_p into seven bundles that are correct without them: two block-diffusion models (which use
    /// denoising steps and block size, not sampling), three DFlash DRAFT models (which never generate
    /// independently and have no chat template by design), and two image models. Exactly one bundle
    /// of the eight was a real gap. A sanitiser that cannot tell these apart does more harm than the
    /// defects it repairs.
    public enum Kind: String, Sendable, Equatable {
        case chatLLM
        /// A speculative-decoding drafter. No chat template, no sampling defaults, by design.
        case draftModel
        /// Diffusion/block-diffusion: its knobs are denoising steps and block size.
        case diffusion
        /// Image generation, not a language model.
        case imageModel
        case unknown

        /// Classify from what the bundle declares. Deliberately conservative: anything unrecognised
        /// is `unknown` and gets only the architecture-agnostic checks, never the chat-model ones.
        public static func classify(modelID: String, configKeys: Set<String>, modelType: String?,
                                    tensorNames: Set<String>) -> Kind {
            // A drafter is identified by its TENSORS, not its name: DFlash 2 carries a
            // `candidate_selector` with predecessor/successor codebooks, DFlash 1 a `dflash_config`.
            // Naming is a convention a converter can break; the weights cannot lie.
            if tensorNames.contains(where: { $0.hasPrefix("candidate_selector.") })
                || configKeys.contains("dflash_config") { return .draftModel }
            let t = (modelType ?? "").lowercased()
            if t.contains("diffusion") { return .diffusion }
            if t.contains("image") || modelID.lowercased().contains("qwen-image") { return .imageModel }
            if t.isEmpty && configKeys.isEmpty { return .unknown }
            return .chatLLM
        }
    }

    public enum Severity: String, Sendable { case fixable, keep, info }

    public struct Finding: Sendable, Equatable {
        public let kind: Kind
        public let files: [String]
        public let detail: String

        public enum Kind: String, Sendable, Equatable {
            /// A complete duplicate sharding: unindexed, and every tensor also reachable via the
            /// index. Safe to delete once byte-identity is confirmed by the caller.
            case staleDuplicateShard
            /// Unindexed but carrying tensors nothing else has. NEVER delete.
            case unindexedUniqueData
            /// The index names files that are not present. The weights may still be fine.
            case indexReferencesMissingFiles
            /// Weight files exist but no index does.
            case noIndex
            /// Everything the index names is present and every file is accounted for.
            case healthy
            /// A chat model with no sampling defaults. REPORT ONLY: the value must be imported from
            /// an authoritative source (a sibling quant of the same base model, or the upstream
            /// repo), never invented — but falling back to generic settings is worse than a
            /// published model-specific default, so this is a real defect, not a preference.
            case missingSamplingDefaults
            /// A chat model with no chat template. MC scoring still works, generation fails with
            /// `missingChatTemplate` — so the model looks selectively broken rather than
            /// mis-packaged. Never auto-filled: a guessed template produces plausible output that is
            /// subtly off-protocol, which is worse than a loud failure.
            case missingChatTemplate
        }

        public var severity: Severity {
            switch kind {
            case .staleDuplicateShard, .indexReferencesMissingFiles, .noIndex: return .fixable
            case .unindexedUniqueData: return .keep
            // Report-only: repairable in principle, but only from a SOURCE, so never by `doctor`.
            case .missingSamplingDefaults, .missingChatTemplate: return .keep
            case .healthy: return .info
            }
        }

        public init(kind: Kind, files: [String], detail: String) {
            self.kind = kind
            self.files = files
            self.detail = detail
        }
    }

    public static func audit(_ b: Bundle) -> [Finding] {
        var out: [Finding] = []

        if !b.hasIndex {
            if b.files.count > 1 {
                out.append(Finding(kind: .noIndex, files: b.files.sorted(),
                                   detail: "\(b.files.count) weight files and no index — an "
                                   + "index-trusting loader has nothing to read"))
            }
            return out.isEmpty ? [Finding(kind: .healthy, files: [], detail: "single file, no index needed")] : out
        }

        if !b.indexReferencesMissing.isEmpty {
            out.append(Finding(kind: .indexReferencesMissingFiles,
                               files: b.indexReferencesMissing.sorted(),
                               detail: "index names \(b.indexReferencesMissing.count) file(s) that do "
                               + "not exist; rebuild it from the files actually present"))
        }

        var stale: [String] = [], keep: [String] = []
        for f in b.files.sorted() where !b.indexedFiles.contains(f) {
            let tensors = b.tensorsByFile[f] ?? []
            // THE SAFETY RULE. Unique tensors mean this file is the ONLY copy of that data, whatever
            // the index says about it. Five bundles depend on exactly this branch.
            if tensors.subtracting(b.indexedTensors).isEmpty && !tensors.isEmpty {
                stale.append(f)
            } else if !tensors.isEmpty {
                keep.append(f)
            }
        }
        if !stale.isEmpty {
            out.append(Finding(kind: .staleDuplicateShard, files: stale,
                               detail: "\(stale.count) unindexed file(s) whose every tensor is also "
                               + "reachable through the index — a superseded sharding"))
        }
        if !keep.isEmpty {
            out.append(Finding(kind: .unindexedUniqueData, files: keep,
                               detail: "unindexed but holds tensors found nowhere else — REQUIRED "
                               + "despite being absent from the index"))
        }
        return out.isEmpty ? [Finding(kind: .healthy, files: [], detail: "index and files agree")] : out
    }
}
