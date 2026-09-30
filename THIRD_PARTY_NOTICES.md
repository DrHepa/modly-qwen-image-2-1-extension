# Third-party notices

The original Modly adapter, installer, asset verifier, packaging utility and tests
in this repository are copyright 2026 DrHepa and licensed under MIT. MIT does not
relicense any model weights, tokenizer/configuration material, system prompts or
third-party dependencies. No model weights are distributed in this repository.

## Qwen Image 2.1 and derivative prompt enhancers

This integration is powered by Qwen Image 2.1 by the Qwen team, Alibaba.
The applicable Qwen Research License dated September 20, 2026 is preserved in
`licenses/image-QWEN-RESEARCH-LICENSE.txt`. It restricts model use to
non-commercial research and evaluation; commercial use requires separate
permission. Consult the complete license, including its redistribution,
attribution, naming, and other obligations, rather than treating this paragraph
as a substitute. The MIT wrapper does not remove any model-use restriction.

The exact consumed checkpoints are:

| Publisher / model | Immutable revision | Local license copy |
|---|---|---|
| [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/790c92633540aa0cb11d9abf19eb46d861714758) | `790c92633540aa0cb11d9abf19eb46d861714758` | `licenses/image-QWEN-RESEARCH-LICENSE.txt` |
| [pottokao/Qwen-Image-2.1-PE-T2I-Heretic](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-T2I-Heretic/tree/96e92081f46cd0ca2ac9679432c5b8b02c12cfdb) | `96e92081f46cd0ca2ac9679432c5b8b02c12cfdb` | `licenses/t2i-QWEN-RESEARCH-LICENSE.txt` |
| [darrellbest/Qwen-Image-2.1-PE-I2I-Heretic](https://huggingface.co/darrellbest/Qwen-Image-2.1-PE-I2I-Heretic/tree/0402e068524500b29633ccc4f85b90779fe73092) | `0402e068524500b29633ccc4f85b90779fe73092` | `licenses/i2i-QWEN-RESEARCH-LICENSE.txt` |

The latter two are community derivatives of Qwen's PE checkpoints, published by
pottokao and darrellbest respectively. They are not official Qwen releases or
endorsed by Qwen. Their publishers describe directional-ablation changes intended
to reduce refusal. Neither absence of refusals nor equivalence to original model
quality is guaranteed. This integration does not endorse or reproduce their
reported benchmark results. The system prompts come from the chosen checkpoints
and remain third-party material; they are not MIT adapter code.

The independently written adapter follows the documented interface of
[QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1/tree/fb7ae1d1f9611cd91524d03c53c5246b36ac8577),
revision `fb7ae1d1f9611cd91524d03c53c5246b36ac8577`. No Qwen prompt-rewriting
source files are vendored or relicensed here.

## Diffusers

The image pipeline is provided by Hugging Face Diffusers, revision
`e0abab83b5df05de9e7abd788643c1a7c1e42e28`, installed separately by setup.
Diffusers is Apache-2.0 licensed; a license copy is provided as
`licenses/diffusers-APACHE-2.0.txt`. Its Qwen Image 2.1 implementation is under
Apache-2.0, separately from the model weights' Qwen Research License.
No Diffusers source is vendored or modified in this repository.

## Other separately installed dependencies

PyTorch and torchvision, Hugging Face Transformers, Tokenizers, Hugging Face Hub,
Safetensors, Accelerate, and Pillow remain under their upstream licenses.
They are installed into the extension venv and are not bundled in the source
package. Their wheels carry their own metadata and license files. Transitive
packages retain their original licenses as well.

## Host and integration credit

- Modly host application: Lightning Pixel and its contributors.
- Modly integration in this repository: DrHepa.
- Image model / original prompt-enhancer model: Qwen team, Alibaba.
- Community prompt-enhancer modifications: the respective publishers above.

Names in this notice are attribution, not claims of sponsorship or endorsement.
