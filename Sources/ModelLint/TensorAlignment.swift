// Copyright © 2026 model-lint contributors
// SPDX-License-Identifier: MIT

import Foundation

/// Whether a bundle's tensors are naturally aligned for their dtype.
///
/// MLX's mmap loader takes the zero-copy path only when
/// `byte_offset % size_of(dtype) == 0` (`mlx/io/safetensors.cpp`). A tensor that fails it is
/// COPIED into a freshly allocated aligned buffer at load. That is invisible in every other
/// respect — the bytes are the same, so the model's OUTPUT is bit-identical — but it costs real
/// memory, and past the point where a model no longer fits, the machine swaps and throughput
/// collapses.
///
/// Observed on a 128 GB machine: a 95 GB bundle with 1,775 of 2,999 tensors misaligned copied
/// 59.3 GB into anonymous memory, which filled the compressor and dropped generation to roughly
/// three tokens per minute. Aligning the shards took the compressor from 60.1 GB to 0.8 GB.
///
/// Two things cause it, both in how the file was written: an unpadded header (so the data block
/// begins at an odd absolute byte and NOTHING after it can be 2- or 4-byte aligned), and tensors
/// packed back-to-back at arbitrary relative offsets.
public enum TensorAlignment {

    /// Byte width of each safetensors dtype. Unknown names are treated as single bytes, which can
    /// only UNDER-report — a conservative direction for a diagnostic.
    static let dtypeSize: [String: Int] = [
        "F64": 8, "I64": 8, "U64": 8,
        "F32": 4, "I32": 4, "U32": 4,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
        "I8": 1, "U8": 1, "BOOL": 1,
    ]

    /// The alignment a rewrite targets. 8 covers every dtype above, so one value serves them all,
    /// and the cost is at most 7 bytes per tensor — about 21 KB across a 95 GB model.
    public static let target = 8

    public struct Survey: Sendable, Equatable {
        public let tensors: Int
        public let misaligned: Int
        /// Bytes the loader would have to copy — the number that matters, not the tensor count:
        /// one misaligned 5 GB tensor costs more than a thousand misaligned scalars.
        public let copiedBytes: Int
        public var isClean: Bool { misaligned == 0 }
    }

    /// Read one shard's header. Returns the parsed map and the absolute offset of the data block.
    static func header(of url: URL) -> (map: [String: Any], dataStart: Int)? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else { return nil }
        let length = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        // A sane bound: a corrupt or non-safetensors file must not become a multi-gigabyte read.
        guard length > 0, length < 200_000_000,
            let raw = try? handle.read(upToCount: Int(length)), raw.count == Int(length),
            let map = try? JSONSerialization.jsonObject(with: raw) as? [String: Any]
        else { return nil }
        return (map, 8 + Int(length))
    }

    /// Survey every `.safetensors` file in a directory.
    public static func survey(directory: URL) -> Survey {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?
            .filter { $0.hasSuffix(".safetensors") }.sorted() ?? []
        var tensors = 0, misaligned = 0, copied = 0
        for name in files {
            guard let (map, dataStart) = header(of: directory.appending(path: name)) else { continue }
            for (key, value) in map where key != "__metadata__" {
                guard let meta = value as? [String: Any],
                    let dtype = meta["dtype"] as? String,
                    let offsets = meta["data_offsets"] as? [Int], offsets.count == 2
                else { continue }
                tensors += 1
                let size = dtypeSize[dtype] ?? 1
                if (dataStart + offsets[0]) % size != 0 {
                    misaligned += 1
                    copied += offsets[1] - offsets[0]
                }
            }
        }
        return Survey(tensors: tensors, misaligned: misaligned, copiedBytes: copied)
    }
}

// MARK: - Repair

extension TensorAlignment {

    /// The outcome of rewriting one shard. A refusal names its reason and leaves the original
    /// untouched — every failure path here is "do nothing", never "do half of it".
    public enum Rewrite: Sendable, Equatable {
        case alreadyAligned
        case rewrote(tensors: Int)
        case refused(String)

        public var changed: Bool { if case .rewrote = self { return true } else { return false } }
    }

    static let chunk = 8 << 20

    /// Rewrite one shard so every tensor sits at an 8-byte boundary.
    ///
    /// The tensor BYTES are never touched — only their positions — which is why this cannot change
    /// what the model computes. The proof is not a claim: before replacing anything, every tensor in
    /// the candidate is compared byte for byte against the original.
    public static func rewrite(shard url: URL) -> Rewrite {
        guard let (original, start) = Self.header(of: url) else { return .refused("unreadable header") }

        var names: [String] = [], meta: [String: (dtype: String, shape: [Int], begin: Int, end: Int)] = [:]
        for (key, value) in original where key != "__metadata__" {
            guard let m = value as? [String: Any],
                let dtype = m["dtype"] as? String,
                let shape = m["shape"] as? [Int],
                let offsets = m["data_offsets"] as? [Int], offsets.count == 2
            else { return .refused("malformed entry for \(key)") }
            names.append(key)
            meta[key] = (dtype, shape, offsets[0], offsets[1])
        }
        guard !names.isEmpty else { return .refused("no tensors") }
        guard names.contains(where: { (start + meta[$0]!.begin) % (dtypeSize[meta[$0]!.dtype] ?? 1) != 0 })
        else { return .alreadyAligned }

        // Lay out in the ORIGINAL data order so the copy is a forward scan of the source rather than
        // a seek storm — on a 20 GB shard that is the difference between minutes and much longer.
        names.sort { meta[$0]!.begin < meta[$1]!.begin }

        var newHeader: [String: Any] = [:]
        if let m = original["__metadata__"] { newHeader["__metadata__"] = m }
        var plan: [(name: String, from: Int, to: Int, at: Int)] = []
        var cursor = 0
        for name in names {
            let m = meta[name]!
            if cursor % target != 0 { cursor += target - (cursor % target) }
            newHeader[name] = ["dtype": m.dtype, "shape": m.shape,
                               "data_offsets": [cursor, cursor + (m.end - m.begin)]]
            plan.append((name, m.begin, m.end, cursor))
            cursor += m.end - m.begin
        }

        // The header's LENGTH shifts the absolute base, and the base decides whether an aligned
        // relative offset is aligned absolutely — so pad the header to a multiple of `target` and
        // the two coincide. Without this step the whole rewrite is a no-op on an odd-length header.
        guard var blob = try? JSONSerialization.data(withJSONObject: newHeader, options: [.sortedKeys])
        else { return .refused("could not serialize header") }
        // Swift's % truncates toward zero, so the Python round-up idiom `(-n) % align` yields a
        // NEGATIVE count here. Spell the round-up out instead of negating.
        blob.append(contentsOf: [UInt8](repeating: 0x20,
                                        count: (target - (8 + blob.count) % target) % target))
        guard (8 + blob.count) % target == 0 else { return .refused("header padding failed") }

        let tmp = url.appendingPathExtension("aligned-tmp")
        try? FileManager.default.removeItem(at: tmp)

        // One shard's worth of free space, not a whole second copy of the bundle. Checked up front so
        // a full disk is a refusal rather than a truncated file.
        let need = Int64(8 + blob.count + cursor)
        if let free = try? url.deletingLastPathComponent().resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage, free < need + (1 << 30) {
            return .refused("not enough free space (needs \(need >> 20) MB)")
        }

        guard FileManager.default.createFile(atPath: tmp.path, contents: nil),
            let src = try? FileHandle(forReadingFrom: url),
            let dst = try? FileHandle(forWritingTo: tmp)
        else { return .refused("could not open a temporary file") }

        func fail(_ reason: String) -> Rewrite {
            try? src.close(); try? dst.close()
            try? FileManager.default.removeItem(at: tmp)
            return .refused(reason)
        }

        var length = UInt64(blob.count).littleEndian
        dst.write(Data(bytes: &length, count: 8))
        dst.write(blob)
        let newStart = 8 + blob.count
        for step in plan {
            let gap = (newStart + step.at) - Int((try? dst.offset()) ?? 0)
            if gap > 0 { dst.write(Data(count: gap)) }
            try? src.seek(toOffset: UInt64(start + step.from))
            var remaining = step.to - step.from
            while remaining > 0 {
                // An explicit pool per chunk. `FileHandle.read` hands back autoreleased backing
                // store, and at eight megabytes a chunk a multi-shard run accumulates it faster than
                // the enclosing pool drains — which is a SIGKILL, not a slowdown. Found only at real
                // scale: a 253 MB shard is 32 chunks and never shows it.
                let ok = autoreleasepool { () -> Bool in
                    guard let block = try? src.read(upToCount: min(chunk, remaining)), !block.isEmpty
                    else { return false }
                    dst.write(block)
                    remaining -= block.count
                    return true
                }
                guard ok else { return fail("short read in \(step.name)") }
            }
        }
        try? dst.close()

        // VERIFY, then replace. A rewrite that is not proven identical is a rewrite that gets thrown
        // away: the cost of being wrong here is a silently corrupted model that still loads.
        if let why = proveIdentical(original: url, at: start, candidate: tmp,
                                    expected: plan.map { ($0.name, $0.from, $0.to, meta[$0.name]!.dtype,
                                                          meta[$0.name]!.shape) }) {
            try? src.close()
            return fail(why)
        }
        try? src.close()

        guard (try? FileManager.default.replaceItemAt(url, withItemAt: tmp)) != nil else {
            try? FileManager.default.removeItem(at: tmp)
            return .refused("replace failed")
        }
        return .rewrote(tensors: plan.count)
    }

    /// Prove a rewritten shard carries exactly the original's tensors, dtypes, shapes and BYTES.
    ///
    /// Separate from `rewrite` so it can be pointed at a candidate this code did not produce — which
    /// is the only way to test it. Verifying the output of a correct rewrite proves nothing about the
    /// verifier: deleting it entirely leaves such a test green. Returns nil when the candidate is
    /// sound, or the reason it is not.
    static func proveIdentical(
        original: URL, at start: Int, candidate: URL,
        expected: [(name: String, from: Int, to: Int, dtype: String, shape: [Int])]
    ) -> String? {
        guard let (check, checkStart) = header(of: candidate) else { return "candidate header unreadable" }
        let claimed = Set(check.keys.filter { $0 != "__metadata__" })
        guard claimed == Set(expected.map(\.name)) else { return "tensor set changed" }
        guard let a = try? FileHandle(forReadingFrom: original),
            let b = try? FileHandle(forReadingFrom: candidate)
        else { return "could not reopen for verification" }
        defer { try? a.close(); try? b.close() }

        for want in expected {
            guard let m = check[want.name] as? [String: Any] else { return "\(want.name): missing" }
            let dtype: String? = m["dtype"] as? String
            let shape: [Int]? = m["shape"] as? [Int]
            guard dtype == want.dtype, shape == want.shape else { return "\(want.name): dtype/shape changed" }
            guard let offsets = m["data_offsets"] as? [Int], offsets.count == 2,
                offsets[1] - offsets[0] == want.to - want.from,
                (checkStart + offsets[0]) % (dtypeSize[want.dtype] ?? 1) == 0
            else { return "\(want.name): still misaligned or resized" }

            try? a.seek(toOffset: UInt64(start + want.from))
            try? b.seek(toOffset: UInt64(checkStart + offsets[0]))
            var remaining = want.to - want.from
            while remaining > 0 {
                let size = min(chunk, remaining)
                let same = autoreleasepool { () -> Bool in
                    guard let x = try? a.read(upToCount: size), let y = try? b.read(upToCount: size),
                        x.count == size, x == y
                    else { return false }
                    return true
                }
                guard same else { return "\(want.name): bytes differ" }
                remaining -= size
            }
        }
        return nil
    }
}
