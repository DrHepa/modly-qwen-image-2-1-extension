# Qwen Image 2.1 for Modly — 0.3.3

A **type: model** extension for local image generation and editing powered by
**Qwen Image 2.1**, with up to ten ordered references and optional **community
reduced-refusal prompt enhancement**. The adapter is MIT; model use is subject to
the separate **Qwen Research License (non-commercial research/evaluation)**.
Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before using the checkpoints.
Reduced-refusal does not mean guaranteed uncensored, official, or quality-equivalent.

## Status and required host capability

This extension uses **one shared main checkpoint** and **two distinct
node-private prompt-enhancer checkpoints**. Version 0.3.3 requires native
`weight_groups` and host-injected `shared_model_dirs`; it does not duplicate
main weights or reconstruct sibling paths as a fallback.

The [shared-weights host PR #348](https://github.com/lightningpixel/modly/pull/348)
remains unmerged. Stock upstream Modly therefore does not provide the required
shared-weight contract. [PR #275](https://github.com/lightningpixel/modly/pull/275)
adds multiple Hugging Face sources but does not provide shared weights on its
own. Before installing, verify that your Modly build implements the required
shared-weight and multi-image input contracts; a compiled app or branch name
alone is not proof of its runtime behavior.

CPU/mock and parser tests establish source-level behavior, not real inference.
Version 0.3.3 has not undergone real inference. An exact-version 0.3.2
**backend** enhanced-edit run on a compatible non-upstream host encoded ten
references and produced a verified PNG; that earlier result does not qualify
0.3.3. The other three modes have only older-version run evidence.
No UI-initiated Run, stock-upstream inference, GitHub installation, full
lifecycle/cancellation test, or perceptual-quality assessment has been completed.

## Usage

### Nodes

| Node ID | Input | Checkpoints | Output |
|---|---|---|---|
| `generate` | text | image model | PNG |
| `edit` | 1–10 image references + prompt | image model | PNG |
| `generate-enhanced` | text | image model + community T2I PE | PNG |
| `edit-enhanced` | 1–10 image references + prompt | image model + community I2I PE | PNG |

Use a base node for direct prompting and complete enhancer bypass. Enhanced nodes
require both checkpoints to be downloaded. Original and effective prompts, exact
parameters and reference count are stored in PNG text metadata; prompts are not
printed in logs. Share PNGs carefully if prompt text is private.

The edit nodes expose Reference 1 through Reference 10. The first is required;
connect up to nine additional inputs in order. Connected images are compacted in
port order: if slot 2 is unused, connected slot 3 becomes `<image2>`. Logs show the
slot-to-effective-image mapping. Refer to effective images as `<image1>`, etc.
No primary-image duplication is added. Alpha is preserved for the image pipeline;
the enhancer sees a white-composited RGB copy. Relative extra paths resolve from
the runner's `WORKSPACE_DIR`, never from a selected output collection.

On a compatible host, `input_contract` supplies the reference labels and nine
optional indicators without changing the `image`, `image_2`, ... `image_10`
handles. Required flags are UI metadata: model preflight checks input types,
not each required port. Always connect the first slot explicitly; a host may
otherwise use its globally selected image. The manifest also retains
`input_labels` for hosts that ignore `input_contract`. This fallback does not
guarantee a first-slot graph edge.

Text nodes ignore the host's dummy image and send no image references. Use an
edit node when references are needed; a text node cannot receive them implicitly.

## Installation and managed checkpoints

Once the repository is available, use **Models/Extensions → Install from GitHub**
with `https://github.com/DrHepa/modly-qwen-image-2-1-extension`. This GitHub installation route has not yet been qualified;
first confirm that your Modly build supports the required shared-weight contract.
A source URL is not a claim of upstream compatibility or a release.

For local development, place the repository root under the Modly extension
runtime directory, reload extensions, then use **Repair**. A local-folder link
alone does not run setup. The required files are at the repository root; do not
install a parent directory containing a nested extension folder.

1. Install/Repair the extension environment.
2. In **Models**, download only the node(s) you need.
3. Wait for all model sources to finish. A sentinel config file alone is not a
   complete checkpoint.
4. Connect text or images, enter a prompt, and generate. Runtime first checks
   Torch/CUDA and required packages, then checks checkpoint paths, exact file
   sizes, shard indexes, and the complete loader-visible file set. It does **not**
   stream or hash weight payloads on load or generation. Unlisted files (including
   alternate weights, indexes, configs and adapters) or any symlinks are rejected
   to prevent loader shadowing. Only regular `<locked filename>.part` resume
   sidecars are allowed; each enhanced node checks its own `prompt_enhancer/`
   subtree separately. No unexpected files are deleted automatically.

The host still injects the exact node-private `MODEL_DIR`, normally
`<MODELS_DIR>/qwen-image-2-1/<node-id>/`, and separately injects the
shared main directory through `shared_model_dirs["qwen-image-2-1"]` (serialized
as `SHARED_MODEL_DIRS` for subprocess runners). The layout is:

```text
<MODELS_DIR>/qwen-image-2-1/
  _shared/qwen-image-2-1/                    # one main checkpoint, 25 files
  generate-enhanced/prompt_enhancer/         # distinct T2I PE, 13 files
  edit-enhanced/prompt_enhancer/             # distinct I2I PE, 13 files
```

All four nodes reference the same native shared group. Base nodes have no private
source plan. Each enhanced node adds only its matching PE source. Downloading
another node skips the already-complete shared main checkpoint; the two PE
checkpoints are distinct and must not replace each other. Setup never downloads
weights; load/generate use only explicit injected local roots, not a global cache.
A missing/malformed shared mapping fails with an actionable host-capability error.

| Payload | Exact bytes |
|---|---:|
| Main, shared by all four nodes | 33,131,609,424 |
| T2I PE, generate-enhanced only | 18,839,810,883 |
| I2I PE, edit-enhanced only | 18,839,819,300 |
| **All checkpoints together** | **70,811,239,607** |

This is about 70.81 decimal GB for all weights. Allow extra space for the venv,
pip cache, host download sidecars and outputs. File hashes and immutable
revisions remain in `assets.lock.json` as reference metadata, but runtime checks
only paths, exact sizes, shard indexes and file-set closure. Equal-size corruption
or snapshot drift is **not** detected by this fast check; perform a separate
offline hash audit or redownload from Models when integrity is in doubt. No
payload is included in this repository. The main root and each PE root have
separate strict closure checks: an unexpected PE subtree inside the shared main
is rejected.

### Shared-weight lifecycle caution

Do not assume that removing a node or uninstalling the extension also removes
its shared main checkpoint. Confirm your host's shared-weight deletion behavior
before freeing storage, and never manually delete shared assets while a
dependent node is active.

### Packaging

From the extension directory, choose a **new** output directory:

```sh
python3 package_extension.py --target modern --output /tmp/qwen-image-2-1-shared
```

The builder preserves the shared manifest and never rewrites source/runtime.
`upstream-main` and any other unsupported target are rejected **before output
creation**. It does not silently remove enhanced nodes or generate duplicate
single-repository plans.

## Requirements

### Setup and platform compatibility

`setup.py` accepts Modly's single JSON argument and the legacy positional form:

```text
setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]
```

It creates/reuses exactly `venv` in the extension directory. Symlinked venv roots,
missing/symlinked `pyvenv.cfg`, and interpreters whose `sys.prefix` is not this
isolated venv are rejected before pip. Ownership is rechecked after creation;
normal venv interpreter symlinks remain allowed. Damaged or unowned environments
fail closed rather than being silently modified. It checks real
interpreter/platform information, probes NVIDIA driver/compute capability, and
uses pinned Torch 2.11.0 + torchvision 0.26.0 CUDA 13.0 wheels. Current Modly
reports CUDA 12.8 even for newer drivers; verified driver 580+ evidence is used
to correct that stale hint without changing the host. Incompatible venvs are
renamed for preservation, not deleted. An owned, isolated legacy venv with a
dangling interpreter symlink (for example after removing Python 3.11) is also
preserved as `venv.incompatible-*` before creating the selected interpreter's
venv; suspicious roots/configs still fail closed. Preserved backups are excluded
from Git and the packaged extension, but retain their local disk usage. Required
package failures stop setup.

Candidate targets are CPython **3.11 or 3.12**, Linux ARM64/x64 or Windows x64, NVIDIA
SM80+ with a CUDA-13-capable driver. Linux ARM64 cu130 CP311/CP312 wheel hashes are pinned. Runtime qualification remains platform-specific.
These are installation candidates, **not hardware-support claims**. Other
platforms (macOS/MPS, CPU, ROCm, Windows ARM64) fail explicitly. Git is required
for the immutable Diffusers installation. PyTorch CUDA BF16 matmul and torchvision
CUDA NMS are verified during setup; successful native checks are not inference.

Runtime pins: Diffusers commit `e0abab83b5df05de9e7abd788643c1a7c1e42e28`,
Transformers 5.17.0, tokenizers 0.23.1, huggingface-hub 1.32.0,
safetensors 0.8.0, accelerate 1.11.0, Pillow 11.1.0. Required transitive packages
are resolved by pip and checked with `pip check`; this is not a fully hash-locked
transitive environment. No vLLM, FlashAttention, xformers, or compilation is used.
The setup marker attests only to verified dependencies, not model readiness.

Some dependencies may warn that `causal_conv1d` or FLA (Flash Linear Attention)
is unavailable. These are optional acceleration paths: the supported fallback
is correct but slower. This package neither installs them without a separately
audited platform-specific plan nor suppresses their warnings.

## Parameters

Defaults match the pipeline audit: 1024×1024, 40 steps, seed 42, true CFG 1,
KV cache enabled, BF16 SDPA and CPU model offload. Width, height and reference
processing resolution must be multiples of 32; invalid dimensions fail instead
of being silently rounded. Negative prompt has an effect only above true CFG 1.
The seed is explicit; bitwise reproducibility across hardware is not promised.

The enhancer emits validated structured JSON. Its aspect suggestion is recorded
but **never overrides user width/height**. Truncated/invalid JSON, nonexistent
image references or, with thinking enabled, incomplete thinking delimiters cause
an actionable error. Both enhanced nodes default to **Enhancer thinking: Off**
and a **4096-token** maximum for a faster direct JSON answer. Set thinking to
**On** only when willing to allow slower internal reasoning; the token budget
remains adjustable up to 32768. Off may reduce prompt quality for some requests,
and neither mode guarantees a completion time or a particular content policy.
In fast mode, the **first complete schema-valid JSON object wins**: generation
stops at a periodic validation point, and any later repeated prose or JSON is
ignored. A malformed first value or object fails explicitly; the parser never
searches forward for a later usable object. Thinking-on still requires a closed
`</think>` block followed by one complete JSON answer, with no trailing text.
Increase the budget or use a base node if enhancement fails; there is no silent
raw-text fallback. Internal thinking is never logged or saved. The image model and enhancer
are staged sequentially to avoid keeping both in memory. Enhancers reload for
each request after their previous memory is released.

## Limitations

The main payload is about 33 GB and the enhancer about 19 GB before activations.
Ten references, large dimensions and CFG above 1 increase memory and latency.
32 GB VRAM is planning guidance with offload, **not a measured peak requirement**;
adequate system RAM is also required. GB10 unified memory does not guarantee no
OOM. Start with one small base-generation smoke, then qualify larger tasks.

## Troubleshooting

Host stderr logs identify fast checkpoint readiness checks, actual model construction,
enhancer tokens, text/reference encoding, denoising step N/N, decoding and atomic
PNG saving. Errors retain full tracebacks. Progress percentages never decrease,
but are stage indicators, not time estimates. Check cancellation between stages,
at token callbacks and at each denoising step. A running native kernel, weight
load, or VAE operation may not stop immediately; the host may terminate the
runner. In-process cancellation never returns a PNG as success. Process death
can leave a hidden incomplete `.png.tmp` file, never a returned output; unrelated
files are not automatically deleted on restart.

## Outputs

Output is a uniquely named validated PNG in the host-injected output directory.
Unload clears the image pipeline, enhancer, processor and allocator cache.
Generation exceptions also unload state. Cleanup hooks are best-effort and cannot
replace the original error or cancellation; warnings identify cleanup failures.
Post-inference hook cleanup occurs before an output is committed; a failed hook
drops the resident pipeline while a valid generated PNG can still be delivered. No editor/viewer mesh is produced here;
use the resulting reference image in a separate 3D model node.

## Development validation

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q generator.py assets.py setup.py package_extension.py
```

CPU tests cover no-payload-read readiness, broken-venv preservation, late
shared-map injection, no-fallback failures, separate main/PE closure,
legacy-package rejection, manifest/lock contracts, input ordering/holes/alpha,
parameter validation, enhancement parsing, atomic saving, monotonic progress,
mocked denoising cancellation and cleanup. These mocks do not prove inference,
UI rendering, downloaded bytes, or Windows support. An exact 0.3.2 backend
run qualified the enhanced edit path with ten references on a compatible
non-upstream host only; no 0.3.3 inference has run. Real-host acceptance needs setup + Repair,
host-managed downloads, all four modes, visible logs/errors, cancellation,
unload/restart, UI-initiated Run and Install from GitHub.

## License

### Credits

Integration: **DrHepa**. Host: **Modly by Lightning Pixel**. Models: **Qwen team,
Alibaba**. Community enhancer publishers: **pottokao** and **darrellbest**.
Image pipeline: **Hugging Face Diffusers**. See [LICENSE](LICENSE) for the MIT
adapter and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained model and
library terms. No sponsorship or endorsement is implied.
