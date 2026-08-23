import ArgumentParser
import Foundation
import ModelLint

/// `model-lint` — audit a local model collection for packaging defects.
///
/// Standalone on purpose. The checks are header-only reads of standard Hugging Face conventions —
/// safetensors headers, `model.safetensors.index.json`, `config.json`, chat templates — with nothing
/// MLX-specific about them, so this has no business requiring a model runtime to be installed. It
/// links `ModelLint`, which depends on Foundation and nothing else: it audits bundles it could never
/// load, on machines without the RAM to load them, before a conversion is spent on them.
@main
struct ModelLintCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model-lint",
        abstract: "Report packaging defects in a local model collection; --doctor repairs the derivable ones.",
        discussion: """
            Reports by default and writes nothing.

            Most findings are the CONVERTER's to fix rather than yours — an index naming files that \
            were never shipped, or a quantisation that dropped the chat template, is broken for \
            every downloader, not just you. `--format hf` emits a report ready to file against the \
            model repo for exactly that reason.

            `--doctor` switches to repair mode, and repairs only what is DERIVABLE from the bundle: \
            deleting a shard layout proven byte-identical to an indexed one, and rebuilding an index \
            from the files actually present. It will not invent a chat template or a sampling \
            default, because those come from outside the bundle — writing an unattributed value into \
            a vendor file would make a guess indistinguishable from the author's declaration.

            Repair mode previews by default; add --apply to write.
            """)

    @Option(name: .long, help: "Model root to scan (default: ~/Library/MLModels).")
    var root: String = NSString(string: "~/Library/MLModels").expandingTildeInPath

    @Option(name: .long, help: "Only bundles whose id contains this substring.")
    var filter: String?

    @Option(name: .long, help: "Output: text | hf (a report ready to file against the model repo).")
    var format: String = "text"

    @Flag(name: .long, help: "Repair the locally-derivable defects instead of only reporting.")
    var doctor = false

    @Flag(name: .long, help: "With --doctor: actually write the repairs (default: preview).")
    var apply = false

    @Flag(name: .long, help: "Exit non-zero when any defect is found (for scripting).")
    var strict = false

    func validate() throws {
        guard ["text", "hf"].contains(format) else {
            throw ValidationError("--format must be 'text' or 'hf', not '\(format)'")
        }
        // Fail here rather than silently reporting: someone who typed --apply meant to change
        // something, and a run that quietly does nothing looks like a successful repair.
        guard !apply || doctor else {
            throw ValidationError("--apply only means something with --doctor (it gates the writes).")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root, isDirectory: &isDir), isDir.boolValue else {
            throw ValidationError("no such model root: \(root)")
        }
    }

    func run() async throws {
        let reports = ModelLint.scan(root: root, filter: filter)

        if doctor {
            let outcome = ModelDoctor.repair(reports, apply: apply)
            print(outcome.log.joined(separator: "\n"))
        } else {
            print(format == "hf" ? ModelLint.huggingFace(reports) : ModelLint.text(reports))
        }

        if strict && !reports.isEmpty { throw ExitCode(1) }
    }
}
