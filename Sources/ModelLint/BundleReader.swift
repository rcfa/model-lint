import Foundation

/// Reading a model bundle off disk — shared by `model-lint` and `model-doctor`.
///
/// Header-only: an 8-byte length prefix then that many bytes of JSON. Auditing a 100 GB collection
/// costs kilobytes and cannot disturb a running job.
public enum BundleReader {

    public struct Read {
        public let id: String
        public let dir: URL
        public let bundle: ModelBundleAudit.Bundle
        public let kind: ModelBundleAudit.Kind
        public let hasChatTemplate: Bool
        public let samplingKeys: Set<String>
    }

    public static let samplingKeys = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "do_sample"]

    /// Find bundles under `root`, in either layout.
    ///
    /// The CHECKS are format-generic — safetensors headers, `model.safetensors.index.json`,
    /// `config.json`, `generation_config.json`, chat templates — all standard Hugging Face
    /// conventions with nothing MLX-specific about them. Only DISCOVERY was layout-bound, which
    /// meant a plain HF cache could not be linted at all. Supporting it lets a download be checked
    /// BEFORE a conversion is spent on it, which is when a broken index is cheapest to find.
    ///
    /// - `<root>/<org>/<model>` — this project's own tree.
    /// - `<root>/models--<org>--<name>/snapshots/<sha>/` — the Hugging Face hub cache.
    public static func discover(root: String, filter: String?) -> [URL] {
        let fm = FileManager.default
        var out: [(id: String, url: URL)] = []
        let rootURL = URL(fileURLWithPath: root, isDirectory: true)

        for entry in (try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil)) ?? []
        where entry.hasDirectoryPath {
            let name = entry.lastPathComponent
            if name.hasPrefix("models--") {
                // HF cache: the weights live in a revision snapshot, and there may be several. Take
                // each, labelled by its short sha, so two revisions of one repo stay distinguishable.
                let id = name.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
                let snaps = entry.appendingPathComponent("snapshots")
                for rev in (try? fm.contentsOfDirectory(at: snaps, includingPropertiesForKeys: nil)) ?? []
                where rev.hasDirectoryPath {
                    out.append(("\(id)@\(rev.lastPathComponent.prefix(8))", rev))
                }
            } else if name.hasPrefix("datasets--") || name.hasPrefix("spaces--") {
                continue                                  // not models; nothing here to lint
            } else {
                for m in (try? fm.contentsOfDirectory(at: entry, includingPropertiesForKeys: nil)) ?? []
                where m.hasDirectoryPath {
                    out.append((m.pathComponents.suffix(2).joined(separator: "/"), m))
                }
            }
        }
        return out.filter { filter.map($0.id.contains) ?? true }
            .sorted { $0.id < $1.id }.map(\.url)
    }

    public static func read(_ dir: URL) -> Read? {
        let fm = FileManager.default
        let id = dir.pathComponents.suffix(2).joined(separator: "/")
        let all = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        // safetensors only. A PyTorch-origin repo (`pytorch_model.bin` + `.bin.index.json`) has no
        // readable header format here — its tensor names live in a pickle, which cannot be inspected
        // without executing it. Reporting such a bundle as "no weight files" would be a lie, so it is
        // skipped and named below instead.
        let files = all.filter { $0.hasSuffix(".safetensors") }.sorted()
        guard !files.isEmpty else {
            if all.contains(where: { $0.hasSuffix(".bin") }) {
                let msg = "  skipped \(id): PyTorch .bin weights — tensor names are inside a "
                    + "pickle and cannot be read without executing it\n"
                FileHandle.standardError.write(Data(msg.utf8))
            }
            return nil
        }

        var tensorsByFile: [String: Set<String>] = [:]
        for f in files { tensorsByFile[f] = (try? tensorNames(dir.appendingPathComponent(f))) ?? [] }

        var indexedFiles: Set<String> = [], indexedTensors: Set<String> = [], missing: Set<String> = []
        var hasIndex = false
        if let data = fm.contents(atPath: dir.appendingPathComponent("model.safetensors.index.json").path),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let wm = obj["weight_map"] as? [String: String] {
            hasIndex = true
            let named = Set(wm.values), present = Set(files)
            indexedFiles = named.intersection(present)
            indexedTensors = Set(wm.keys)
            missing = named.subtracting(present)
        }

        var cfgKeys: Set<String> = [], modelType: String?
        for name in ["config.json", "config_omni.json"] {
            if let d = fm.contents(atPath: dir.appendingPathComponent(name).path),
               let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                cfgKeys.formUnion(o.keys)
                modelType = modelType ?? (o["model_type"] as? String)
            }
        }
        var sampling: Set<String> = []
        if let d = fm.contents(atPath: dir.appendingPathComponent("generation_config.json").path),
           let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            sampling = Set(o.keys).intersection(samplingKeys)
        }
        let allTensors = tensorsByFile.values.reduce(into: Set<String>()) { $0.formUnion($1) }
        let kind = ModelBundleAudit.Kind.classify(modelID: id, configKeys: cfgKeys,
                                                  modelType: modelType, tensorNames: allTensors)
        let template = fm.fileExists(atPath: dir.appendingPathComponent("chat_template.jinja").path)
            || cfgKeys.contains("chat_template")
            || (fm.contents(atPath: dir.appendingPathComponent("tokenizer_config.json").path)
                    .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["chat_template"] != nil)

        return Read(id: id, dir: dir,
                    bundle: .init(files: files, indexedFiles: indexedFiles,
                                  indexedTensors: indexedTensors, tensorsByFile: tensorsByFile,
                                  indexReferencesMissing: missing, hasIndex: hasIndex),
                    kind: kind, hasChatTemplate: template, samplingKeys: sampling)
    }

    /// Findings from the weight audit PLUS the configuration checks, which are gated on bundle kind.
    public static func findings(for r: Read) -> [ModelBundleAudit.Finding] {
        var out = ModelBundleAudit.audit(r.bundle).filter { $0.kind != .healthy }
        // ONLY a chat model is expected to carry these. A drafter never generates independently, a
        // diffusion model's knobs are denoising steps, and an image model is not a language model —
        // flagging them would be seven false positives out of eight.
        if r.kind == .chatLLM {
            if r.samplingKeys.isEmpty {
                out.append(.init(kind: .missingSamplingDefaults, files: ["generation_config.json"],
                                 detail: "chat model declares no sampling defaults, so it falls back "
                                 + "to generic settings rather than the author's published ones"))
            }
            if !r.hasChatTemplate {
                out.append(.init(kind: .missingChatTemplate, files: ["chat_template.jinja"],
                                 detail: "chat model ships no chat template — MC scoring still works "
                                 + "while generation fails, so it looks selectively broken"))
            }
        }
        return out
    }

    public static func tensorNames(_ url: URL) throws -> Set<String> {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        guard let lenData = try h.read(upToCount: 8), lenData.count == 8 else { return [] }
        let n = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard n > 0, n < 200_000_000, let body = try h.read(upToCount: Int(n)),
              let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else { return [] }
        return Set(obj.keys.filter { $0 != "__metadata__" })
    }

    public static func tensorBytes(_ url: URL, name: String) throws -> Data {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        guard let lenData = try h.read(upToCount: 8), lenData.count == 8 else { return Data() }
        let n = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard let body = try h.read(upToCount: Int(n)),
              let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any],
              let e = obj[name] as? [String: Any],
              let off = e["data_offsets"] as? [Int], off.count == 2 else { return Data() }
        try h.seek(toOffset: 8 + n + UInt64(off[0]))
        return try h.read(upToCount: off[1] - off[0]) ?? Data()
    }

    public static func size(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64) ?? 0
    }
}
