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

    /// Repeatable, because models genuinely live in several places at once.
    @Option(name: .long, help: ArgumentHelp(
        "Directory holding models. Repeatable. Default: ~/Library/MLModels.",
        discussion: "Apple specifies no location for model assets, but ~/Library is where macOS "
            + "keeps application support data (Preferences, Application Support, Caches), and large "
            + "weights are that kind of thing — hence the default. Plenty of tools instead hide "
            + "models in a dot-directory (~/.cache/huggingface/hub, ~/.lmstudio/models, "
            + "~/.<project>/models), so point this wherever yours actually are, or set "
            + "MODEL_LINT_ROOT. Repeat it to audit several locations in one run.",
        valueName: "dir"))
    var root: [String] = []

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

    /// Apple defines no location for model assets. It does define what ~/Library is for — support
    /// data an app owns but a user may still need to find — and multi-gigabyte weights fit that far
    /// better than a hidden dot-directory does. So this default is an extrapolation, not a citation,
    /// and --root plus MODEL_LINT_ROOT exist for the world as it is.
    static let defaultRoot = ProcessInfo.processInfo.environment["MODEL_LINT_ROOT"].flatMap {
        $0.isEmpty ? nil : $0
    } ?? NSString(string: "~/Library/MLModels").expandingTildeInPath

    func validate() throws {
        guard ["text", "hf"].contains(format) else {
            throw ValidationError("--format must be 'text' or 'hf', not '\(format)'")
        }
        // Fail here rather than silently reporting: someone who typed --apply meant to change
        // something, and a run that quietly does nothing looks like a successful repair.
        guard !apply || doctor else {
            throw ValidationError("--apply only means something with --doctor (it gates the writes).")
        }
        for r in root.isEmpty ? [Self.defaultRoot] : root {
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: r, isDirectory: &isDir), isDir.boolValue else {
                // Name the alternatives rather than silently scanning one: an audit that reports
                // nothing must never be confusable with an audit that scanned nothing.
                let known = ModelLint.knownModelDirectories()
                let hint = known.isEmpty ? ""
                    : "\n\nModels appear to be in:\n" + known.map { "  \($0)" }.joined(separator: "\n")
                throw ValidationError("no such model root: \(r)\nPass --root <dir> or set MODEL_LINT_ROOT.\(hint)")
            }
        }
    }

    func run() async throws {
        let roots = root.isEmpty ? [Self.defaultRoot] : root
        // ALWAYS say what was scanned. An audit that reports nothing is indistinguishable from an
        // audit that scanned nothing, and the second is far more likely when the default is a guess.
        if format != "hf" {
            for r in roots { print("scanning \(r)") }
        }

        var reports: [ModelLint.Report] = []
        for r in roots { reports += ModelLint.scan(root: r, filter: filter) }

        if doctor {
            let outcome = ModelDoctor.repair(reports, apply: apply)
            print(outcome.log.joined(separator: "\n"))
        } else {
            print(format == "hf" ? ModelLint.huggingFace(reports) : ModelLint.text(reports))
        }

        if strict && !reports.isEmpty { throw ExitCode(1) }
    }
}
