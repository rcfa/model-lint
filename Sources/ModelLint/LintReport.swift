import Foundation

/// Scanning a model collection and rendering what is wrong with it.
///
/// The rendering functions RETURN strings rather than printing them. That is not style: a report is
/// the product of this tool, and a product you can only observe by capturing stdout cannot be
/// asserted on. Every emitter here is directly testable against a fixture bundle.
public enum ModelLint {

    public struct Report: Sendable {
        public let id: String
        public let kind: ModelBundleAudit.Kind
        public let dir: URL
        public let fileCount: Int
        public let findings: [ModelBundleAudit.Finding]

        public var fixable: [ModelBundleAudit.Finding] { findings.filter { $0.severity == .fixable } }
    }

    /// Model directories that EXIST on this machine — used to make a bad --root helpful, not to
    /// pick one silently.
    ///
    /// There is no standard location, which is the whole problem. `~/Library/MLModels` is where they
    /// belong if you follow Apple's conventions; almost nothing follows them. The Hugging Face cache
    /// is where most tooling actually writes, and every app that manages its own downloads invents a
    /// third place. So this returns the ones that EXIST on this machine and the caller reports which
    /// it used — guessing silently is how you audit an empty directory and conclude all is well.
    ///
    /// Environment overrides come first, and `HF_HOME`/`HUGGINGFACE_HUB_CACHE` are honoured because a
    /// user who set them has already said where their models are.
    public static func knownModelDirectories() -> [String] {
        let env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        var candidates: [String] = []
        if let r = env["MODEL_LINT_ROOT"], !r.isEmpty { candidates.append(r) }
        if let r = env["HUGGINGFACE_HUB_CACHE"], !r.isEmpty { candidates.append(r) }
        if let r = env["HF_HOME"], !r.isEmpty { candidates.append(r + "/hub") }
        candidates += [
            home + "/.cache/huggingface/hub",   // the de-facto default for HF tooling
            home + "/Library/MLModels",         // where Apple's conventions put them
            home + "/.lmstudio/models",         // LM Studio
            home + "/.ollama/models",           // Ollama
        ]
        var seen = Set<String>()
        return candidates.filter { path in
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue
            else { return false }
            return seen.insert(path).inserted
        }
    }

    /// Read every bundle under `root` and keep the ones with something wrong.
    ///
    /// Header-only throughout: an 8-byte length prefix then that many bytes of JSON per weight file.
    /// Auditing a 100 GB collection costs kilobytes and cannot disturb a running job.
    public static func scan(root: String, filter: String? = nil) -> [Report] {
        var out: [Report] = []
        for dir in BundleReader.discover(root: root, filter: filter) {
            guard let r = BundleReader.read(dir) else { continue }
            let f = BundleReader.findings(for: r)
            guard !f.isEmpty else { continue }
            out.append(Report(id: r.id, kind: r.kind, dir: r.dir,
                              fileCount: r.bundle.files.count, findings: f))
        }
        return out
    }

    // MARK: - text

    public static func text(_ reports: [Report]) -> String {
        var lines: [String] = []
        for r in reports {
            lines.append("")
            lines.append("\(r.id)  [\(r.kind.rawValue)]")
            for f in r.findings {
                lines.append("  \(f.severity == .fixable ? "FIXABLE" : "REPORT ") \(f.kind.rawValue): \(f.detail)")
                for n in f.files.prefix(3) { lines.append("            \(n)") }
                if f.files.count > 3 { lines.append("            … \(f.files.count - 3) more") }
            }
        }
        let fixable = reports.flatMap(\.findings).filter { $0.severity == .fixable }.count
        lines.append("")
        lines.append("""
            \(reports.count) bundle(s) with findings — \(fixable) fixable by `--doctor`, \
            the rest need a source (upstream repo or sibling quant) and are reported only.
            """)
        return lines.joined(separator: "\n")
    }

    // MARK: - Hugging Face issue report

    /// A per-model markdown report to file against the model repo.
    ///
    /// These defects are almost always the CONVERTER's, not yours: an index naming files that were
    /// never shipped, or a quantisation that dropped the chat template, affects every downloader.
    /// Repairing it locally fixes one copy and leaves the artifact broken for everyone else — so the
    /// useful output of a lint is a bug report, not just a to-do list.
    public static func huggingFace(_ reports: [Report]) -> String {
        var out: [String] = []
        for r in reports {
            out.append("""

            <!-- ─────────── \(r.id) ─────────── -->
            ### Packaging issue in `\(r.id)`

            Found by an automated bundle audit (`model-lint`) that compares the safetensors index
            against the tensor headers actually present. Reporting here because the files as
            published are affected, not just my local copy.

            **Environment:** MLX bundle, \(r.fileCount) weight file(s), classified as \
            `\(r.kind.rawValue)`.
            """)
            for f in r.findings {
                out.append("\n**\(title(f.kind))**\n")
                out.append(body(f, r))
            }
            // The repro must match the FINDING. Telling a maintainer to diff safetensors headers
            // when the complaint is a missing chat template reads as boilerplate and invites the
            // report to be dismissed.
            let weightish: Set<ModelBundleAudit.Finding.Kind> =
                [.staleDuplicateShard, .indexReferencesMissingFiles, .noIndex, .unindexedUniqueData]
            let repro = r.findings.contains { weightish.contains($0.kind) }
                ? "read each `*.safetensors` header (the 8-byte length prefix plus that many bytes "
                  + "of JSON; no tensor data needed) and compare the tensor names against "
                  + "`model.safetensors.index.json`'s `weight_map`"
                : "check the repo for the file named above; it is absent from the listing"
            out.append("\n**How to reproduce** — \(repro).")
        }
        return out.joined(separator: "\n")
    }

    static func title(_ k: ModelBundleAudit.Finding.Kind) -> String {
        switch k {
        case .staleDuplicateShard: return "The repo contains two complete shardings of the same weights"
        case .indexReferencesMissingFiles: return "`model.safetensors.index.json` references files that are not in the repo"
        case .noIndex: return "No `model.safetensors.index.json` for a multi-file bundle"
        case .unindexedUniqueData: return "A required weight file is absent from the index"
        case .missingSamplingDefaults: return "No sampling defaults in `generation_config.json`"
        case .missingChatTemplate: return "No chat template shipped"
        case .healthy: return "No issue"
        }
    }

    static func body(_ f: ModelBundleAudit.Finding, _ r: Report) -> String {
        let list = f.files.prefix(6).map { "- `\($0)`" }.joined(separator: "\n")
        let more = f.files.count > 6 ? "\n- … \(f.files.count - 6) more" : ""
        switch f.kind {
        case .indexReferencesMissingFiles:
            let bytes = ByteCountFormatter.string(
                fromByteCount: bundleBytes(r), countStyle: .file)
            return """
            The index names \(f.files.count) file(s) that the repo does not contain, while the \
            \(r.fileCount) weight file(s) it does contain (\(bytes)) are referenced by \
            nothing. Any loader that trusts the index fails; loaders that scan the directory instead \
            happen to work, which is why this can go unnoticed.

            Missing according to the index:
            \(list)\(more)
            """
        case .staleDuplicateShard:
            let bytes = ByteCountFormatter.string(
                fromByteCount: f.files.reduce(0) { $0 + BundleReader.size(r.dir.appendingPathComponent($1)) },
                countStyle: .file)
            return """
            These files are not referenced by the index, and every tensor in them also appears in an \
            indexed file with the same dtype, shape and bytes. They look like a superseded shard \
            layout left behind by a re-conversion — \(bytes) of duplicated download for everyone who \
            pulls the repo.

            \(list)\(more)
            """
        case .missingChatTemplate:
            return """
            No `chat_template.jinja`, and none in `tokenizer_config.json`. Multiple-choice scoring \
            still works because it needs no template, so the model appears selectively broken rather \
            than mis-packaged. If the base model ships a customised template, the conversion appears \
            to have dropped it.
            """
        case .missingSamplingDefaults:
            return """
            `generation_config.json` declares none of \
            \(BundleReader.samplingKeys.map { "`\($0)`" }.joined(separator: ", ")). Consumers fall \
            back to generic defaults rather than the values recommended for this model, which \
            usually differ.
            """
        case .noIndex:
            return "The repo has \(r.fileCount) weight files and no index, so a loader has "
                + "nothing to enumerate them with.\n\n\(list)\(more)"
        case .unindexedUniqueData, .healthy:
            return f.detail
        }
    }

    private static func bundleBytes(_ r: Report) -> Int64 {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: r.dir.path)) ?? []
        return files.filter { $0.hasSuffix(".safetensors") }
            .reduce(0) { $0 + BundleReader.size(r.dir.appendingPathComponent($1)) }
    }
}
