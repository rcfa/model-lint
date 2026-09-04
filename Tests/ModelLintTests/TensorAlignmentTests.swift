import Foundation
import Testing

@testable import ModelLint

/// Moving bytes without changing any.
///
/// The whole value of the alignment repair rests on one claim: the model computes exactly what it
/// computed before. That claim is worth only as much as the test that tries to break it, so the
/// tests here assert the PROPERTY (every absolute offset divides evenly by its dtype width) and the
/// VALUES (bytes written as literals, read back through the ordinary reader) — never the offsets the
/// implementation happened to choose. A test that recomputed those offsets would agree with the code
/// by construction and could not fail.
@Suite("safetensors alignment: detect, repair, prove nothing moved")
struct TensorAlignmentTests {

    /// Write a safetensors file, optionally with the real-world defect: an UNPADDED header.
    ///
    /// This is what the misaligned bundles on disk look like. The header length is whatever JSON
    /// serialization produced, so the data block starts at an arbitrary absolute byte and nothing
    /// after it can be 2- or 4-byte aligned no matter how the tensors are spaced.
    private func write(
        _ tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])],
        padHeader: Bool, to url: URL
    ) throws {
        var header: [String: Any] = [:]
        var cursor = 0
        for t in tensors {
            header[t.name] = ["dtype": t.dtype, "shape": t.shape,
                              "data_offsets": [cursor, cursor + t.bytes.count]]
            cursor += t.bytes.count
        }
        var blob = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        if padHeader {
            blob.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - (8 + blob.count) % 8) % 8))
        } else if (8 + blob.count) % 8 == 0 {
            blob.append(0x20)  // force the defect even if the JSON happened to land evenly
        }
        var out = Data()
        var length = UInt64(blob.count).littleEndian
        out.append(Data(bytes: &length, count: 8))
        out.append(blob)
        for t in tensors { out.append(contentsOf: t.bytes) }
        try out.write(to: url)
    }

    private func scratch() throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "model-lint-align-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Four float32 values, byte patterns chosen to be recognizable if anything shifts by one.
    private let payloadA: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04,
                                     0xFF, 0x00, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44]
    private let payloadB: [UInt8] = [0xCA, 0xFE, 0xBA, 0xBE, 0x7F, 0x80, 0x81, 0x82]

    @Test("an unpadded header is detected, and every tensor after it counts as misaligned")
    func detectsUnpaddedHeader() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write([("a", "F32", [4], payloadA), ("b", "F32", [2], payloadB)],
                  padHeader: false, to: dir.appending(path: "model.safetensors"))

        let survey = TensorAlignment.survey(directory: dir)
        #expect(survey.tensors == 2)
        #expect(survey.misaligned == 2)
        #expect(survey.copiedBytes == payloadA.count + payloadB.count)
        #expect(!survey.isClean)
    }

    /// The mutation check for the test above: the SAME tensors, the SAME layout, differing only in
    /// the header padding. If this reported misalignment too, the detector would be measuring
    /// something other than what it claims to.
    @Test("a padded header with evenly spaced tensors is clean")
    func paddedHeaderIsClean() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write([("a", "F32", [4], payloadA), ("b", "F32", [2], payloadB)],
                  padHeader: true, to: dir.appending(path: "model.safetensors"))

        let survey = TensorAlignment.survey(directory: dir)
        #expect(survey.tensors == 2)
        #expect(survey.isClean)
        #expect(survey.copiedBytes == 0)
    }

    /// THE TEST THAT MATTERS. Not "did the offsets change" — did any BYTE change.
    ///
    /// The bytes come back through `BundleReader.tensorBytes`, which resolves them from the rewritten
    /// header rather than from anything this test computed, and they are compared against literals
    /// declared before the rewrite ran.
    @Test("the repair changes offsets and nothing else")
    func repairPreservesEveryByte() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(path: "model.safetensors")
        // A 3-byte tensor between them, so the following offsets cannot land evenly by luck.
        try write([("a", "F32", [4], payloadA),
                   ("odd", "U8", [3], [0xAA, 0xBB, 0xCC]),
                   ("b", "F32", [2], payloadB)],
                  padHeader: false, to: url)

        #expect(TensorAlignment.survey(directory: dir).misaligned > 0)
        #expect(TensorAlignment.rewrite(shard: url) == .rewrote(tensors: 3))
        #expect(TensorAlignment.survey(directory: dir).isClean)

        #expect(try Array(BundleReader.tensorBytes(url, name: "a")) == payloadA)
        #expect(try Array(BundleReader.tensorBytes(url, name: "odd")) == [0xAA, 0xBB, 0xCC])
        #expect(try Array(BundleReader.tensorBytes(url, name: "b")) == payloadB)
        #expect(try BundleReader.tensorNames(url) == ["a", "odd", "b"])
    }

    /// safetensors requires the data region to be TILED: the Rust loader walks tensors in offset
    /// order and rejects the file at the first hole with "invalid offset for tensor X". The repair
    /// used to pad each tensor up to an 8-byte boundary, which aligned it and simultaneously made the
    /// file invalid for every loader except MLX — a bundle repaired here was refused outright by
    /// vMLX on 2026-09-04. These two assert the invariants together, because satisfying one while
    /// breaking the other is exactly the failure that shipped.
    @Test("the repair leaves NO gaps — tensors tile the data region")
    func repairIsContiguous() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "gapfree-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(path: "model-00001-of-00001.safetensors")
        // An F32 [3] is 12 bytes: 12 % 8 = 4, so a pad-to-8 layout puts a 4-byte hole after it.
        // Mixed dtypes are what make the ordering matter at all.
        try write([
            (name: "a.f32_odd", dtype: "F32", shape: [3], bytes: Array(repeating: 1, count: 12)),
            (name: "b.bf16", dtype: "BF16", shape: [5], bytes: Array(repeating: 2, count: 10)),
            (name: "c.i64", dtype: "I64", shape: [2], bytes: Array(repeating: 3, count: 16)),
            (name: "d.u8", dtype: "U8", shape: [7], bytes: Array(repeating: 4, count: 7)),
        ], padHeader: false, to: url)

        _ = TensorAlignment.rewrite(shard: url)

        let survey = TensorAlignment.survey(directory: dir)
        #expect(survey.gaps == 0, "the repair left \(survey.gaps) hole(s); safetensors rejects any")
        #expect(survey.misaligned == 0, "the repair left \(survey.misaligned) misaligned tensor(s)")
    }

    /// The survey must SEE a gap, or a bundle this tool damaged reports as clean — which is what
    /// happened to eleven local bundles: aligned, holed, and passed as fine for four days.
    @Test("a gapped shard is reported, not passed as clean")
    func surveyDetectsGaps() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "gapped-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(path: "model-00001-of-00001.safetensors")
        // Hand-built with a hole: two tensors, the second starting 4 bytes late.
        var header: [String: Any] = [:]
        header["a"] = ["dtype": "F32", "shape": [3], "data_offsets": [0, 12]]
        header["b"] = ["dtype": "F32", "shape": [4], "data_offsets": [16, 32]]
        var blob = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while (8 + blob.count) % 8 != 0 { blob.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(blob.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(blob)
        out.append(Data(repeating: 0, count: 32))
        try out.write(to: url)

        let survey = TensorAlignment.survey(directory: dir)
        #expect(survey.gaps == 1, "expected the 4-byte hole to be counted, got \(survey.gaps)")
        #expect(!survey.isClean, "a gapped bundle must not report clean")
    }

    @Test("a shard that is already aligned is left alone, so the repair is idempotent")
    func alreadyAlignedIsUntouched() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(path: "model.safetensors")
        try write([("a", "F32", [4], payloadA)], padHeader: false, to: url)

        #expect(TensorAlignment.rewrite(shard: url).changed)
        let after = try Data(contentsOf: url)
        #expect(TensorAlignment.rewrite(shard: url) == .alreadyAligned)
        #expect(try Data(contentsOf: url) == after, "a second pass must not rewrite the file")
    }

    /// A refusal is only useful if the original survives it. Truncating the data block leaves the
    /// header describing bytes that are not there, so the copy cannot complete.
    @Test("a shard whose data is short is refused, and the original is untouched")
    func refusalLeavesTheOriginal() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(path: "model.safetensors")
        try write([("a", "F32", [4], payloadA), ("b", "F32", [2], payloadB)],
                  padHeader: false, to: url)
        let full = try Data(contentsOf: url)
        try full.dropLast(6).write(to: url)
        let truncated = try Data(contentsOf: url)

        guard case .refused = TensorAlignment.rewrite(shard: url) else {
            Issue.record("a truncated shard must be refused, not rewritten")
            return
        }
        #expect(try Data(contentsOf: url) == truncated)
        #expect(!FileManager.default.fileExists(atPath: url.path + ".aligned-tmp"),
                "the temporary file must not be left behind")
    }

    /// Alignment is per-dtype, so a bundle of single-byte tensors is aligned at every offset. It
    /// would be easy to write a detector that flags anything not on an 8-byte boundary; that would
    /// report work MLX does not actually do.
    @Test("single-byte dtypes are aligned everywhere, whatever the offset")
    func byteDtypesAreNeverMisaligned() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write([("q", "U8", [5], [1, 2, 3, 4, 5]), ("r", "I8", [3], [6, 7, 8])],
                  padHeader: false, to: dir.appending(path: "model.safetensors"))

        let survey = TensorAlignment.survey(directory: dir)
        #expect(survey.tensors == 2)
        #expect(survey.isClean, "U8/I8 need no alignment; flagging them would invent work")
    }

    // MARK: - The verifier itself

    /// Verifying a CORRECT rewrite proves nothing about the verifier — delete the byte comparison
    /// entirely and every test above stays green. So point it at a candidate that is deliberately
    /// wrong, in each of the ways a rewrite can go wrong.

    /// Builds an original and a candidate that differ only in the ways the argument describes.
    private func verifyOutcome(
        corrupt: (inout Data) -> Void = { _ in },
        rename: Bool = false, retype: Bool = false
    ) throws -> String? {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let original = dir.appending(path: "original.safetensors")
        let candidate = dir.appending(path: "candidate.safetensors")
        let tensors: [(String, String, [Int], [UInt8])] =
            [("a", "F32", [4], payloadA), ("b", "F32", [2], payloadB)]
        try write(tensors, padHeader: false, to: original)
        try write(tensors.map { (rename && $0.0 == "b" ? "renamed" : $0.0,
                                 retype && $0.0 == "b" ? "F16" : $0.1, $0.2, $0.3) },
                  padHeader: true, to: candidate)
        var bytes = try Data(contentsOf: candidate)
        corrupt(&bytes)
        try bytes.write(to: candidate)

        guard let (_, start) = TensorAlignment.header(of: original) else { return "unreadable original" }
        return TensorAlignment.proveIdentical(
            original: original, at: start, candidate: candidate,
            expected: [("a", 0, payloadA.count, "F32", [4]),
                       ("b", payloadA.count, payloadA.count + payloadB.count, "F32", [2])])
    }

    @Test("an honest rewrite verifies")
    func verifierAcceptsAGoodCandidate() throws {
        #expect(try verifyOutcome() == nil)
    }

    @Test("a candidate with one byte changed is rejected")
    func verifierCatchesAFlippedByte() throws {
        // The last byte of the file is the last byte of tensor "b" — a change no header can reveal.
        let why = try verifyOutcome { $0[$0.count - 1] ^= 0xFF }
        #expect(why?.contains("bytes differ") == true, "got: \(why ?? "accepted!")")
    }

    @Test("a candidate missing a tensor is rejected")
    func verifierCatchesARenamedTensor() throws {
        #expect(try verifyOutcome(rename: true)?.contains("tensor set changed") == true)
    }

    @Test("a candidate that changed a dtype is rejected")
    func verifierCatchesADtypeChange() throws {
        #expect(try verifyOutcome(retype: true) != nil)
    }
}

/// The filed report has to describe the defect it actually found.
///
/// The reproduction steps are the part a maintainer checks first; steps that do not match the
/// complaint read as boilerplate and get the report dismissed. The alignment finding is the third
/// kind to need its own, and nothing but a test stops the next one from silently inheriting a
/// neighbour's.
@Suite("the filed report matches the finding")
struct LintReportReproTests {

    private func report(_ kind: ModelBundleAudit.Finding.Kind, detail: String) -> ModelLint.Report {
        ModelLint.Report(
            id: "org/model", kind: .chatLLM, dir: URL(fileURLWithPath: "/nonexistent"), fileCount: 1,
            findings: [ModelBundleAudit.Finding(kind: kind, files: [], detail: detail)])
    }

    @Test("an alignment report reproduces by arithmetic, not by diffing the index")
    func alignmentGetsItsOwnRepro() {
        let out = ModelLint.huggingFace([report(.misalignedTensors, detail: "3 of 4 tensors")])
        #expect(out.contains("dtype_size"), "the repro must state the actual check")
        #expect(!out.contains("weight_map"), "that is the INDEX repro, not this one")
        #expect(!out.contains("absent from the listing"), "that is the MISSING-FILE repro")
    }

    @Test("an index report still reproduces by diffing the index")
    func indexReproIsUnchanged() {
        let out = ModelLint.huggingFace([report(.noIndex, detail: "2 files, no index")])
        #expect(out.contains("weight_map"))
        #expect(!out.contains("dtype_size"))
    }
}

/// Candidates left behind by an interrupted rewrite.
///
/// A killed or crashed run leaves its temporary file holding a whole shard's worth of disk. Two
/// tools write these — this one uses `.aligned-tmp`, the Python reference it was ported from uses
/// `.aligned.tmp` — and sweeping only our own spelling left nine of the other kind on disk, one of
/// them 4.4 GB, which a later sync dutifully copied to a second machine.
@Suite("interrupted rewrites leave no litter")
struct OrphanCandidateTests {

    private func scratch() throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "model-lint-orphan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test("both spellings are swept, and real shards are not")
    func sweepsBothSpellings() throws {
        let dir = try scratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ours = dir.appending(path: "model-00001-of-00002.safetensors.aligned-tmp")
        let theirs = dir.appending(path: "model-00002-of-00002.safetensors.aligned.tmp")
        let keep = dir.appending(path: "model-00001-of-00002.safetensors")
        for u in [ours, theirs, keep] { try Data("x".utf8).write(to: u) }

        let (_, notes) = ModelDoctor.realign(in: dir)

        let fm = FileManager.default
        #expect(!fm.fileExists(atPath: ours.path), "our own candidate must be swept")
        #expect(!fm.fileExists(atPath: theirs.path),
                "the Python aligner's candidate must be swept too — it is the one that actually accumulated")
        #expect(fm.fileExists(atPath: keep.path), "a real shard must never be swept")
        #expect(notes.filter { $0.contains("orphaned candidate") }.count == 2)
    }
}
