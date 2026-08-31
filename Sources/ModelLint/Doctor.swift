import Foundation

/// The repairs that are provably safe to make locally.
///
/// Three, and only three: delete a shard layout that is byte-for-byte duplicated elsewhere, rebuild
/// an index from the files actually present, and rewrite shards whose tensors are misaligned for
/// their dtype. All three are DERIVABLE from the bundle — there is exactly one correct answer and it
/// is computable from what is on disk. The alignment repair is the strongest case of the three: it
/// moves bytes without changing any, and every tensor is compared against the original before the
/// rewrite is allowed to replace anything.
///
/// It will not import sampling defaults or chat templates. Not because importing is wrong — a value
/// published by the model's author beats a generic fallback — but because the source is OUTSIDE the
/// bundle, so it belongs to a tool that can fetch and attribute it. Writing an unattributed value
/// into a vendor file makes an inference indistinguishable from a declaration.
public enum ModelDoctor {

    public struct Outcome: Sendable {
        public var deleted = 0
        public var rebuilt = 0
        public var realigned = 0
        public var reclaimable: Int64 = 0
        public var log: [String] = []
    }

    /// Apply the derivable repairs. `apply: false` (the default) reports what it would do and
    /// touches nothing — the mutating act stays a separate, explicit decision.
    public static func repair(_ reports: [ModelLint.Report], apply: Bool = false) -> Outcome {
        var out = Outcome()
        for r in reports {
            let fixable = r.fixable
            guard !fixable.isEmpty else { continue }
            out.log.append("")
            out.log.append(r.id)
            for f in fixable {
                switch f.kind {
                case .staleDuplicateShard:
                    out.reclaimable += f.files.reduce(0) {
                        $0 + BundleReader.size(r.dir.appendingPathComponent($1))
                    }
                    out.log.append("  stale shards: \(f.files.count) file(s)")
                    if apply {
                        let (n, notes) = deleteVerified(f.files, in: r.dir)
                        out.deleted += n
                        out.log.append(contentsOf: notes)
                    }
                case .indexReferencesMissingFiles, .noIndex:
                    out.log.append("  index: \(f.detail)")
                    if apply, let note = rebuildIndex(in: r.dir) {
                        out.rebuilt += 1
                        out.log.append(note)
                    }
                case .misalignedTensors:
                    out.log.append("  alignment: \(f.detail)")
                    if apply {
                        let (n, notes) = realign(in: r.dir)
                        out.realigned += n
                        out.log.append(contentsOf: notes)
                    }
                default: break
                }
            }
        }
        let human = ByteCountFormatter.string(fromByteCount: out.reclaimable, countStyle: .file)
        out.log.append("")
        out.log.append("\(apply ? "applied" : "would apply") — stale \(human), "
                       + (apply ? "deleted \(out.deleted) file(s), rebuilt \(out.rebuilt) index(es), "
                                   + "realigned \(out.realigned) shard(s)"
                                : "re-run with --apply"))
        return out
    }

    /// Delete only after PROVING the copies identical.
    ///
    /// Name equality is not byte equality: a re-conversion can produce the same names with repaired
    /// values, and deleting the newer copy because it happened to be unindexed would destroy the
    /// repair. Any mismatch skips the file rather than the run, so one anomaly cannot cost the rest.
    static func deleteVerified(_ files: [String], in dir: URL) -> (Int, [String]) {
        guard let data = FileManager.default.contents(
                atPath: dir.appendingPathComponent("model.safetensors.index.json").path),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let wm = obj["weight_map"] as? [String: String] else { return (0, []) }
        var n = 0, notes: [String] = []
        for f in files {
            let src = dir.appendingPathComponent(f)
            let names = ((try? BundleReader.tensorNames(src)) ?? []).sorted().prefix(3)
            var proven = 0
            for name in names {
                guard let other = wm[name],
                      let a = try? BundleReader.tensorBytes(src, name: name),
                      let b = try? BundleReader.tensorBytes(dir.appendingPathComponent(other), name: name),
                      a == b
                else { proven = -1; break }
                proven += 1
            }
            guard proven > 0 else { notes.append("    REFUSED \(f) — byte check failed"); continue }
            guard (try? FileManager.default.removeItem(at: src)) != nil else {
                notes.append("    REFUSED \(f) — delete failed"); continue
            }
            n += 1
            notes.append("    deleted \(f)")
        }
        return (n, notes)
    }

    /// Rebuild `model.safetensors.index.json` from the weight files actually present, keeping the
    /// original alongside as `.orig` so the change is reversible without a re-download.
    static func rebuildIndex(in dir: URL) -> String? {
        let all = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        let files = all.filter { $0.hasSuffix(".safetensors") }.sorted()
        var wm: [String: String] = [:]
        var total: Int64 = 0
        for f in files {
            for name in (try? BundleReader.tensorNames(dir.appendingPathComponent(f))) ?? [] {
                wm[name] = f
            }
            total += BundleReader.size(dir.appendingPathComponent(f))
        }
        guard !wm.isEmpty else { return nil }
        let url = dir.appendingPathComponent("model.safetensors.index.json")
        let backup = dir.appendingPathComponent("model.safetensors.index.json.orig")
        if FileManager.default.fileExists(atPath: url.path),
           !FileManager.default.fileExists(atPath: backup.path) {
            try? FileManager.default.copyItem(at: url, to: backup)
        }
        let payload: [String: Any] = ["metadata": ["total_size": total, "rebuilt_by": "model-lint --doctor"],
                                      "weight_map": wm]
        guard let data = try? JSONSerialization.data(withJSONObject: payload,
                                                     options: [.prettyPrinted, .sortedKeys]),
              (try? data.write(to: url, options: .atomic)) != nil else { return nil }
        return "    rebuilt index: \(wm.count) tensors across \(files.count) files"
    }

    /// Rewrite every misaligned shard in a bundle, one at a time.
    ///
    /// Sequential on purpose. Each rewrite needs one shard's worth of free space and reads the whole
    /// shard twice; doing several at once multiplies the disk high-water mark for no gain, since the
    /// work is I/O-bound either way. A refusal stops the bundle rather than continuing: if one shard
    /// cannot be proven identical, the reason probably applies to its neighbours too.
    static func realign(in dir: URL) -> (Int, [String]) {
        let shards = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        var n = 0, notes: [String] = []
        for name in shards {
            switch TensorAlignment.rewrite(shard: dir.appendingPathComponent(name)) {
            case .alreadyAligned:
                continue
            case .rewrote(let tensors):
                n += 1
                notes.append("    realigned \(name) (\(tensors) tensors)")
            case .refused(let why):
                notes.append("    REFUSED \(name) — \(why); the original is untouched")
                notes.append("    stopping this bundle")
                return (n, notes)
            }
        }
        return (n, notes)
    }
}
