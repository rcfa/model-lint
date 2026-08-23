import Testing
import Foundation
@testable import ModelLint

/// Deciding what is wrong with a model bundle — and what only LOOKS wrong.
///
/// Every case here is a real bundle on disk. The audit exists because each defect was invisible until
/// something happened to look: 43 GB of duplicate shards in three Nemotron bundles, seven bundles
/// whose index names files that do not exist, and five bundles carrying a required file the index has
/// never mentioned.
@Suite("model bundle audit: stale vs required")
struct ModelBundleAuditTests {

    private func bundle(files: [String: Set<String>], indexed: [String: Set<String>],
                        missing: Set<String> = [], hasIndex: Bool = true) -> ModelBundleAudit.Bundle {
        var byFile = files
        indexed.forEach { byFile[$0.key] = $0.value }
        return ModelBundleAudit.Bundle(
            files: Array(byFile.keys), indexedFiles: Set(indexed.keys),
            indexedTensors: Set(indexed.values.flatMap { $0 }),
            tensorsByFile: byFile, indexReferencesMissing: missing, hasIndex: hasIndex)
    }

    /// THE SAFETY TEST — the one that protects five real models.
    ///
    /// `jangtq_runtime.safetensors` is unreferenced by the index in every JANGTQ bundle and holds the
    /// ternary codebooks and sign tables that exist nowhere else. A rule of "delete what the index
    /// does not name" would destroy it and break all five. The audit must classify it as KEEP.
    @Test("an unindexed file holding unique tensors is KEEP, never stale")
    func unindexedUniqueDataIsKept() {
        let b = bundle(files: ["jangtq_runtime.safetensors": ["codebook.2048.4", "signs.2048.42"]],
                       indexed: ["model-00001-of-00002.safetensors": ["backbone.embeddings.weight"]])
        let f = ModelBundleAudit.audit(b)
        let keep = f.first { $0.kind == .unindexedUniqueData }
        #expect(keep?.files == ["jangtq_runtime.safetensors"])
        #expect(keep?.severity == .keep)
        #expect(!f.contains { $0.kind == .staleDuplicateShard },
                "a unique-data file must NEVER be reported as a stale shard")
    }

    /// The Nemotron case: a superseded sharding, every tensor also reachable through the index.
    @Test("an unindexed file wholly duplicated by the index is stale")
    func fullyDuplicatedIsStale() {
        let shared: Set<String> = ["backbone.embeddings.weight", "lm_head.scales"]
        let b = bundle(files: ["model-00001-of-00010.safetensors": shared],
                       indexed: ["model-00001-of-00013.safetensors": shared])
        let f = ModelBundleAudit.audit(b)
        #expect(f.first { $0.kind == .staleDuplicateShard }?.files == ["model-00001-of-00010.safetensors"])
        #expect(f.first { $0.kind == .staleDuplicateShard }?.severity == .fixable)
    }

    /// PARTIAL overlap is NOT stale. A file sharing some tensors with the index but holding others of
    /// its own is the dangerous middle case — deleting it loses exactly the tensors that made it
    /// unique, and the loss would be silent because most of its contents were duplicated.
    @Test("a partially duplicated file is kept, not deleted")
    func partialOverlapIsKept() {
        let b = bundle(files: ["model-00001-of-00003.safetensors": ["shared.weight", "only.here.scales"]],
                       indexed: ["model-00001-of-00006.safetensors": ["shared.weight"]])
        let f = ModelBundleAudit.audit(b)
        #expect(f.contains { $0.kind == .unindexedUniqueData })
        #expect(!f.contains { $0.kind == .staleDuplicateShard })
    }

    /// The Devstral/GLM case: the index names files that were never shipped, while the real weights
    /// are referenced by nothing. The weights are fine; the INDEX is the wrong artifact.
    @Test("an index naming absent files is reported as fixable, not as missing weights")
    func indexNamingAbsentFiles() {
        let b = bundle(files: ["model-00001-of-00003.safetensors": ["a.scales", "a.biases"]],
                       indexed: [:], missing: ["model-00001-of-00006.safetensors"])
        let f = ModelBundleAudit.audit(b)
        #expect(f.contains { $0.kind == .indexReferencesMissingFiles })
        // Its real weight file has unique data, so it must also be reported as KEEP — never as stale.
        #expect(f.contains { $0.kind == .unindexedUniqueData })
        #expect(!f.contains { $0.kind == .staleDuplicateShard })
    }

    @Test("multiple weight files with no index at all is fixable")
    func noIndexAtAll() {
        let b = bundle(files: ["model-00001-of-00002.safetensors": ["a"],
                               "model-00002-of-00002.safetensors": ["b"]],
                       indexed: [:], hasIndex: false)
        #expect(ModelBundleAudit.audit(b).first?.kind == .noIndex)
    }

    /// A single-file bundle needs no index, and flagging it would bury the real findings in noise.
    @Test("a single weight file with no index is not a defect")
    func singleFileNeedsNoIndex() {
        let b = bundle(files: ["model.safetensors": ["a"]], indexed: [:], hasIndex: false)
        #expect(ModelBundleAudit.audit(b).first?.kind == .healthy)
    }

    @Test("a bundle whose index and files agree is healthy")
    func healthyBundle() {
        let b = bundle(files: [:], indexed: ["model-00001-of-00001.safetensors": ["a", "b"]])
        #expect(ModelBundleAudit.audit(b).map(\.kind) == [.healthy])
    }

    // MARK: - Bundle kind, the gate that prevents harm

    /// THE SEVEN FALSE POSITIVES THIS PREVENTS. Of eight bundles with no sampling defaults, exactly
    /// ONE was a real gap. The other seven are correct without them — and five would be actively
    /// damaged by injecting chat-model sampling into a drafter or a diffusion model.
    @Test("a drafter is recognised by its TENSORS, not its name")
    func drafterByTensors() {
        // DFlash 2: the converted bundle ships weights only — no config, no template — so a
        // name-based rule would see an unremarkable Qwen and demand both.
        let k = ModelBundleAudit.Kind.classify(
            modelID: "ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4", configKeys: [], modelType: nil,
            tensorNames: ["candidate_selector.hidden_projection.weight", "fc.biases"])
        #expect(k == .draftModel)
        // DFlash 1 declares itself in config instead.
        #expect(ModelBundleAudit.Kind.classify(modelID: "anthonyya/Qwen3.6-27B-DFlash-4bit",
                                               configKeys: ["dflash_config"], modelType: "qwen3",
                                               tensorNames: ["layers.0.self_attn.q_proj.weight"]) == .draftModel)
    }

    @Test("diffusion and image bundles are not chat models")
    func nonChatKinds() {
        #expect(ModelBundleAudit.Kind.classify(modelID: "OsaurusAI/diffusiongemma-26B-A4B-it-MXFP4",
                                               configKeys: ["model_type"], modelType: "block_diffusion_gemma",
                                               tensorNames: []) == .diffusion)
        #expect(ModelBundleAudit.Kind.classify(modelID: "mlx-community/Qwen-Image-2512-8bit",
                                               configKeys: ["model_type"], modelType: nil,
                                               tensorNames: []) == .imageModel)
    }

    /// An ordinary chat model must still classify as one, or the checks that matter never fire — the
    /// gate has to be conservative in BOTH directions.
    @Test("an ordinary chat model is still a chat model")
    func chatModelStillClassifies() {
        #expect(ModelBundleAudit.Kind.classify(modelID: "cais/HarmBench-Llama-2-13b-cls-Jang_6M",
                                               configKeys: ["model_type", "hidden_size"],
                                               modelType: "llama",
                                               tensorNames: ["model.layers.0.self_attn.q_proj.weight"]) == .chatLLM)
    }

    /// An empty file must not be classified either way — it is neither a duplicate nor unique data,
    /// and treating "no tensors" as "no unique tensors" would mark it deletable on a technicality.
    @Test("an empty weight file is not classified as stale")
    func emptyFileIsNotStale() {
        let b = bundle(files: ["empty.safetensors": []],
                       indexed: ["model-00001-of-00001.safetensors": ["a"]])
        #expect(!ModelBundleAudit.audit(b).contains { $0.kind == .staleDuplicateShard })
    }
}
