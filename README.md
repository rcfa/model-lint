# model-lint

Audits local model bundles for packaging defects, and repairs the ones that are provably safe to
repair. Reads safetensors **headers only** — auditing a 100 GB collection costs kilobytes and cannot
disturb a running job.

```
model-lint                        # report everything wrong, write nothing
model-lint --root ~/models        # audit somewhere else (repeatable)
model-lint --filter Qwen3.8       # only bundles whose id contains this
model-lint --format hf            # a report ready to file against the model repo
model-lint --doctor               # preview the derivable repairs
model-lint --doctor --apply       # make them
```

## Where your models are

`--root` defaults to **`~/Library/MLModels`**. Apple does not specify a location for model assets —
no convention covers them directly. What `~/Library` *is* for is application support data:
`~/Library/Preferences`, `~/Library/Application Support`, `~/Library/Caches`. Multi-gigabyte weights
a user might want to find, move, or exclude from a backup are squarely that kind of thing, so
`~/Library/MLModels` is the natural extrapolation — user-visible, and findable by someone who did not
install them. The default is a gentle argument for putting them there. Plenty of tools instead hide
models in a dot-directory —
`~/.cache/huggingface/hub`, `~/.lmstudio/models`, `~/.<project>/models` — so point it at wherever
yours actually live:

```
model-lint --root ~/.cache/huggingface/hub
model-lint --root ~/Library/MLModels --root ~/.lmstudio/models   # both, one run
export MODEL_LINT_ROOT=~/models                                  # or set it once
```

The root being scanned is always printed, because an audit that reports nothing must never be
confusable with an audit that scanned nothing. If the root doesn't exist the tool says so and names
the model directories it can see, rather than quietly auditing an empty tree and pronouncing you
healthy.

## What it finds

| finding | why it matters |
|---|---|
| `staleDuplicateShard` | Two complete shardings of the same weights. A re-conversion left the old layout behind; everyone who pulls the repo downloads it twice. |
| `indexReferencesMissingFiles` | The index names files the repo doesn't contain. Loaders that trust the index fail; loaders that scan the directory happen to work — which is why it goes unnoticed. |
| `noIndex` | Multi-file bundle with nothing to enumerate it. |
| `unindexedUniqueData` | A required weight file the index never mentions. |
| `missingChatTemplate` | MC scoring still works while generation fails, so the model looks selectively broken rather than mis-packaged. |
| `missingSamplingDefaults` | Consumers fall back to generic settings instead of the author's published ones. |

The last two are reported only for models classified as chat LLMs. A drafter never generates
independently, a diffusion model's knobs are denoising steps, and an image model is not a language
model — flagging those would be seven false positives out of eight.

## What `--doctor` will and won't do

It repairs exactly two things, because exactly two are **derivable from the bundle** — there is one
correct answer and it is computable from what is on disk:

- **Delete a byte-identical duplicate sharding.** Only after proving it: name equality is not byte
  equality, and a re-conversion can produce the same tensor names with *repaired* values. Deleting
  the newer copy because it happened to be unindexed would destroy the repair. Any mismatch skips
  that file rather than aborting the run.
- **Rebuild the index** from the weight files actually present, keeping the original as
  `model.safetensors.index.json.orig`.

It will **not** write a chat template or a sampling default. Not because importing them is wrong — a
value published by the model's author beats a generic fallback — but because the source is *outside*
the bundle. Writing an unattributed value into a vendor file makes a guess indistinguishable from a
declaration.

Repair mode previews by default; `--apply` writes.

## File the bug, don't just fix your copy

Most of these defects belong to the **converter**, not to you. An index naming files that were never
shipped, or a quantisation that dropped the chat template, is broken for every downloader. Repairing
it locally fixes one copy and leaves the artifact broken for everyone else — so `--format hf` emits a
per-model markdown report, with a reproduction that matches the finding, ready to paste into a
discussion on the model repo.

## Layouts it understands

- `<root>/<org>/<model>` — a plain organised tree.
- `<root>/models--<org>--<name>/snapshots/<sha>/` — the Hugging Face hub cache, one entry per
  revision, so two revisions of one repo stay distinguishable.

Supporting the hub cache means a download can be checked **before** a conversion is spent on it,
which is when a broken index is cheapest to find.

PyTorch-origin repos (`pytorch_model.bin`) are skipped and named rather than audited. Their tensor
names are only recoverable by deserialising a Python object graph, which means running arbitrary
code from the bundle — not something an auditing tool should ever do. Reporting such a bundle as
"no weight files" would be a lie, so it says what it skipped and why.

## Dependencies

The `ModelLint` library depends on **Foundation and nothing else** — no MLX, no model runtime. The
checks are standard Hugging Face conventions with nothing framework-specific about them, so the tool
audits bundles it could never load, on machines without the RAM to load them. The CLI adds only
`swift-argument-parser`.

## Origin

Extracted from [osaurus-eval](https://github.com/rcfa), a benchmarking harness for local MLX models,
where it exists because a broken bundle is indistinguishable from a bad model until you look: an
index naming files that were never shipped fails only on loaders that trust it, and a dropped chat
template scores fine on multiple choice while generation collapses. Both were found this way, in
bundles that had been in use for weeks.

It is separated because none of that is specific to benchmarking, or to MLX.
