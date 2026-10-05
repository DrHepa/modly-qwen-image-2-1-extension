# Qwen Image 2.1 for Modly

**Generate or edit images locally, with up to 10 ordered references.** Four nodes
share one Qwen Image 2.1 checkpoint; two can optionally add separate community
prompt enhancers. This is a Modly model extension, version **0.3.3**.

> **License first:** The adapter is [MIT](LICENSE), but the model and enhancer
> checkpoints retain the **Qwen Research License for non-commercial research and
> evaluation**. Read [Third-party notices](THIRD_PARTY_NOTICES.md) before
> downloading or using them. No weights are included in this repository.

## Quick start

1. Use **Modly 0.4.3 or newer**. Its [v0.4.3 release](https://github.com/lightningpixel/modly/releases/tag/v0.4.3)
   includes multiple Hugging Face sources ([#275](https://github.com/lightningpixel/modly/pull/275)),
   extension-scoped shared weight groups ([#348](https://github.com/lightningpixel/modly/pull/348)),
   and text-only detection for multi-input models ([#205](https://github.com/lightningpixel/modly/pull/205))
   needed here. An older host may reject the manifest or fail to inject shared
   model paths.
2. In **Models/Extensions → Install from GitHub**, enter
   `https://github.com/DrHepa/modly-qwen-image-2-1-extension`. Run **Repair** if
   Modly requests environment setup or an existing environment is incompatible.
3. In **Models**, download the node you want to use and wait for all its managed
   sources to finish. Base nodes need the shared image checkpoint; enhanced
   nodes also need their own prompt enhancer.
4. Add the node to a workflow, connect its input, enter a prompt, and run it.
   Start with one reference and modest dimensions before scaling up.

Installing from a local folder only links/reloads the extension; it does **not**
run `setup.py`. Use **Repair** for that route. Install the repository root, not a
parent directory containing it.

## Choose a node

| Node | Input | Additional checkpoint | Output |
| --- | --- | --- | --- |
| **Generate Image** (`generate`) | Text prompt | None | PNG |
| **Edit Image — 10 References** (`edit`) | Prompt + 1–10 images | None | PNG |
| **Generate Image — Enhanced Prompt** (`generate-enhanced`) | Text prompt | Community T2I enhancer | PNG |
| **Edit Image — Enhanced Prompt** (`edit-enhanced`) | Prompt + 1–10 images | Community I2I enhancer | PNG |

The base nodes send your prompt directly to the image pipeline. The enhanced
nodes first rewrite it with their respective **community reduced-refusal**
enhancer. These are not official Qwen releases: reduced refusal, a particular
content policy, and equivalent image quality are **not guaranteed**. A text node
does not implicitly receive images; use an Edit node for references.

### Ordered image references

Edit nodes expose **Reference 1** through **Reference 10**. Connect Reference 1
explicitly; References 2–10 are optional. The pipeline compacts connected slots
in port order, so if Reference 2 is empty but Reference 3 is connected, the latter
becomes `<image2>` in the prompt. The log reports that mapping. There is no
automatic duplication of Reference 1. The required-port indicator is UI
metadata, not a substitute for an actual graph connection; otherwise the host
may fall back to its globally selected image.

Connected alpha is preserved for the image pipeline. The enhancer receives a
white-composited RGB view. Original/effective prompts, parameters, and reference
count are stored in PNG metadata, **not printed in logs**; treat shared PNGs as
potentially containing private prompt text.

### Useful defaults

Output is **1024 × 1024**, with **40 steps**, **seed 42**, true CFG **1**, KV
cache enabled, BF16 SDPA, and CPU model offload. Width, height, and reference
processing resolution must be multiples of 32. Negative prompt affects the
result only when true CFG is above 1. A seed does not promise bitwise-identical
results across hardware.

Enhanced nodes default to **thinking Off** and a **4096-token** enhancer budget.
Thinking On may take longer; neither mode guarantees better quality or a
completion time. Incomplete or invalid structured enhancer output fails
explicitly instead of silently falling back to the raw prompt. The enhancer's
aspect suggestion never overrides your width or height. Use a base node if you
want to bypass enhancement entirely.

## Managed weights and storage

The host downloads the main checkpoint **once** as a shared weight group used
by all four nodes. Each enhanced node has its own distinct private checkpoint:

```text
<MODELS_DIR>/qwen-image-2-1/
  _shared/qwen-image-2-1/                    # Qwen image model; all four nodes
  generate-enhanced/prompt_enhancer/         # community T2I enhancer
  edit-enhanced/prompt_enhancer/             # community I2I enhancer
```

| Download | Approximate decimal size |
| --- | ---: |
| Shared image checkpoint | 33.13 GB |
| T2I enhancer, if selected | 18.84 GB |
| I2I enhancer, if selected | 18.84 GB |
| **All three** | **70.81 GB** |

Allow additional space for the extension venv, pip cache, partial downloads,
and PNG outputs. Setup does **not** download model weights; load/generate use
only the host-injected local model directories, not a global model cache. Do not
assume removing one node deletes the shared checkpoint while other nodes still
depend on it.

Startup performs **fast path, file-size, shard-index, and file-set checks**—not
a full SHA-256 scan of these large payloads. Unexpected files or symlinks are
rejected. `assets.lock.json` retains hashes and pinned revisions as reference
metadata, but **equal-size corruption is not detected** by the fast check. If
integrity is in doubt, audit hashes offline or redownload from Models. A
sentinel config file alone does not prove a complete download.

## Requirements and limits

- **Host:** Modly 0.4.3+ for native shared weight groups, multiple Hugging Face
  sources, and the edit nodes' multi-image routing. Host capability in a release
  is not proof that this exact extension has passed a fresh upstream UI run.
- **Validated platform evidence:** the 0.3.3 environment and isolated backend
  path ran with Modly-managed **CPython 3.12 on Linux ARM64/NVIDIA GB10**. The
  installer also has candidate CPython 3.11/3.12 Linux x64 and Windows x64 CUDA
  lanes; these are **not independently qualified platform claims**. NVIDIA SM80+
  and a CUDA-13-capable driver are required by setup. CPU, macOS/MPS, ROCm, and
  Windows ARM64 are not supported routes.
- **Memory:** the main checkpoint is about 33 GB before activations; an enhancer
  adds about 19 GB while in use. Offload lowers GPU residency but still requires
  system RAM. Ten references, large images, and CFG above 1 increase memory and
  latency. **32 GB VRAM is planning guidance, not a measured minimum or peak.**
- **Dependencies:** setup pins Torch 2.11.0, torchvision 0.26.0, and a Diffusers
  source revision; other direct dependencies are pinned in
  [`requirements.txt`](requirements.txt). `causal_conv1d` and FLA are optional
  acceleration paths; their absence uses a **correct but slower** fallback.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Missing `shared_model_dirs` or shared checkpoint | Confirm Modly 0.4.3+, then finish the managed download in Models. A local copy in another node directory is not a substitute. |
| Missing Torch, native symbols, or an invalid venv | Run **Repair** with the current Modly-managed Python. Setup preserves incompatible venvs rather than silently reusing or deleting them. |
| Missing enhancer files or malformed enhancer response | Download the selected enhanced node's private checkpoint. Increase its token budget or switch to a base node; there is no silent raw-prompt fallback. |
| Out of memory or very slow generation | Try one reference, smaller dimensions, default offload, and fewer steps. Optional kernel warnings may imply slower execution, not incorrect output. |
| Suspected weight damage despite a readiness pass | Fast checks do not detect equal-size corruption; perform an offline hash audit or redownload. |

Progress reports stages, not time estimates. Cancellation is checked between
stages, during enhancer tokens, and between denoising steps; an active native
kernel or weight load may not stop immediately. Generated images are uniquely
named, validated PNGs written atomically to the host output directory. A killed
process can leave a hidden incomplete `.png.tmp` file, never a returned success.

## Validation status

**What is known:** Modly v0.4.3 publishes the required host capabilities. An
independent **0.3.3 isolated backend** `edit-enhanced` run used
**one synthetic reference** and returned a technically valid **256 × 256 PNG**
with enhancement metadata on the tested Linux ARM64/GB10 environment. CPU tests
cover manifest/lock contracts, shared versus private roots, input ordering, fast
readiness, parameter checks, enhancer parsing, atomic output, cancellation, and
cleanup.

**Not independently established here:** a clean v0.4.3 Install from GitHub,
UI-click Run of all four nodes, complete lifecycle/restart behavior, Windows/x64
execution, or perceptual image quality. A technical PNG pass is not a quality
assessment.

For local development:

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q generator.py assets.py setup.py package_extension.py
```

`package_extension.py --target modern --output <new-directory>` creates a
root-installable copy without changing this repository. Legacy targets that
would duplicate the shared checkpoint are intentionally unsupported.

## Credits and licenses

- **Modly integration:** DrHepa — [MIT adapter license](LICENSE).
- **Host:** Modly by Lightning Pixel.
- **Image model and original prompt enhancer:** Qwen team, Alibaba.
- **Optional community enhancer derivatives:** pottokao (T2I) and darrellbest
  (I2I), each pinned separately in the manifest.
- **Image pipeline:** Hugging Face Diffusers, pinned by setup.

The MIT adapter license does **not** relicense weights, tokenizers, system
prompts, or separately installed libraries. See
[Third-party notices](THIRD_PARTY_NOTICES.md) and the retained license files
for the applicable terms. Attribution does not imply endorsement.
