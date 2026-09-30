"""Local-only Modly adapter for image generation and optional prompt enhancement.

This adapter is MIT; checkpoint licenses are separate. No upstream PE source is
vendored. The image pipeline comes from the pinned Apache-2.0 Diffusers package.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

from services.generators.base import BaseGenerator, GenerationCancelled
# Pre-setup discovery does not put the extension on sys.path.
# Load only our own sibling file, with a module name isolated from other plugins.
_assets_spec = importlib.util.spec_from_file_location(
    f'{__name__}_assets', Path(__file__).resolve().parent / 'assets.py')
_assets_module = importlib.util.module_from_spec(_assets_spec)
sys.modules[_assets_spec.name] = _assets_module
_assets_spec.loader.exec_module(_assets_module)
AssetVerifier, load_lock = _assets_module.AssetVerifier, _assets_module.load_lock

DEFAULTS = {
    'prompt': '', 'negative_prompt': '', 'width': 1024, 'height': 1024,
    'output_resolution': 1024, 'steps': 40, 'seed': 42, 'true_cfg_scale': 1.0,
    'use_kv_cache': 'true', 'memory_mode': 'offload',
    'enhancer_thinking': 'off', 'enhancer_max_tokens': 4096,
}
NODE_IDS = ('generate', 'edit', 'generate-enhanced', 'edit-enhanced')


def parse_enhancement(text: str, reference_count: int, *, thinking: bool = True) -> dict:
    """Validate either direct JSON or a closed thinking block's final JSON.

    Only the final answer is retained; internal thinking is never logged or
    embedded in the output. Thinking mode requires a closing boundary; a
    missing boundary is ambiguous/truncated, never a direct-JSON fallback.
    """
    if thinking:
        if '</think>' not in text:
            raise ValueError('Prompt enhancer did not close its thinking block; increase Enhancer token budget or use a base node.')
        answer = text.rsplit('</think>', 1)[1].strip()
        if answer.startswith('```'):
            match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', answer, re.S)
            if not match: raise ValueError('Prompt enhancer returned an incomplete JSON code block.')
            answer = match.group(1)
        try: result = json.loads(answer)
        except (ValueError, TypeError) as exc:
            raise ValueError('Prompt enhancer returned invalid or truncated JSON; increase token budget or use a base node.') from exc
    else:
        answer = text.lstrip()
        if answer.startswith('```'):
            match = re.match(r'```(?:json)?(?=\s|\{)\s*', answer, re.I)
            if not match: raise ValueError('Prompt enhancer returned an unsupported JSON code fence.')
            answer = answer[match.end():]
        try: result, _ = json.JSONDecoder().raw_decode(answer)
        except (ValueError, TypeError) as exc:
            raise ValueError('Prompt enhancer returned invalid or truncated JSON; increase token budget or use a base node.') from exc
    if not isinstance(result, dict): raise ValueError('Prompt enhancer answer must be a JSON object.')
    prompt = result.get('rewritten_prompt', result.get('rewrited_prompt'))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Prompt enhancer JSON has no non-empty rewritten_prompt.')
    ratio = result.get('wh_ratio')
    follow = result.get('ratio_follow', '')
    if not isinstance(ratio, str) or not isinstance(follow, str):
        raise ValueError('Prompt enhancer ratio fields must be strings.')
    if ratio:
        match = re.fullmatch(r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)', ratio)
        if not match or any(float(v) <= 0 for v in match.groups()):
            raise ValueError('Prompt enhancer returned an invalid wh_ratio.')
    if follow:
        match = re.fullmatch(r'<image([1-9]\d*)>', follow)
        if not match or int(match.group(1)) > reference_count:
            raise ValueError('Prompt enhancer ratio_follow refers to a nonexistent image.')
    if reference_count == 0 and (not ratio or follow):
        raise ValueError('Text prompt enhancement requires wh_ratio and no ratio_follow.')
    if reference_count > 0 and not (ratio or follow):
        raise ValueError('Image prompt enhancement requires wh_ratio or ratio_follow.')
    for index in re.findall(r'<image(\d+)>', prompt):
        if not 1 <= int(index) <= reference_count:
            raise ValueError('Rewritten prompt refers to a nonexistent reference image.')
    return {'rewritten_prompt': prompt.strip(), 'wh_ratio': ratio, 'ratio_follow': follow}


class PresencePenalty:
    """Subtract once from each token generated so far, not from input tokens."""
    def __init__(self, penalty, prompt_length):
        self.penalty, self.prompt_length = penalty, prompt_length

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            seen = input_ids[row, self.prompt_length:].unique()
            if seen.numel(): scores[row, seen] -= self.penalty
        return scores


class QwenImage21Generator(BaseGenerator):
    MODEL_ID = 'qwen-image-2-1'
    DISPLAY_NAME = 'Qwen Image 2.1'
    VRAM_GB = 32  # Planning guidance, not a measured ten-reference peak.

    def __init__(self, model_dir: Path, outputs_dir: Path):
        super().__init__(Path(model_dir), Path(outputs_dir))
        self._enhancer = None
        self._processor = None
        self._verifier = AssetVerifier()
        self._lock = load_lock()
        self._memory_mode = None
        self._loaded_node = None

    def _node(self):
        node = getattr(self, 'node_id', '') or self.model_dir.name
        if node not in NODE_IDS:
            raise ValueError(f'Unknown node {node!r}. Select a declared Qwen Image 2.1 node in Modly.')
        return node

    def _image_root(self):
        # The host assigns this map AFTER construction, including in the runner.
        # Never infer a sibling path or fall back to a node-owned checkpoint.
        group = 'qwen-image-2-1'
        mapping = getattr(self, 'shared_model_dirs', None)
        guidance = (f'Modly must inject shared_model_dirs[{group!r}] as an absolute local path. '
                    'Use a shared-weights-capable host (PR348); stock upstream main/dev '
                    'without that feature are unsupported. No fallback copy or download is performed.')
        if not isinstance(mapping, dict) or group not in mapping:
            raise RuntimeError(guidance)
        value = mapping[group]
        if not isinstance(value, (str, Path)) or not str(value) or '\0' in str(value):
            raise RuntimeError(guidance)
        root = Path(value)
        if not root.is_absolute() or '..' in root.parts:
            raise RuntimeError(guidance)
        for ancestor in (root, *root.parents):
            if ancestor.is_symlink():
                raise RuntimeError(f'Modly shared weights path contains a symlink: {ancestor}. {guidance}')
        if root.exists() and not root.is_dir():
            raise RuntimeError(f'Modly shared weights path is not a directory: {root}. {guidance}')
        return root

    @staticmethod
    def _log(message):
        # Stderr belongs to the host log stream; stdout is reserved for runner IPC.
        print(f'[Qwen Image 2.1] {message}', file=sys.stderr, flush=True)

    @staticmethod
    def _offline():
        # Each extension has its own runner process. Disable fallback network use
        # before importing Hub/Transformers/Diffusers; all loader paths are local.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def _validate_assets(self, cancel_event=None):
        self._check_cancelled(cancel_event)
        node = self._node()
        self._verifier.verify(self._image_root(), self._lock['components']['image'], self._log,
                              lambda: self._check_cancelled(cancel_event))
        if node.endswith('-enhanced'):
            key = 'i2i' if node.startswith('edit') else 't2i'
            self._verifier.verify(self.model_dir / 'prompt_enhancer', self._lock['components'][key],
                                  self._log, lambda: self._check_cancelled(cancel_event))
        self._log('Shared image checkpoint and required node-private assets match their pinned SHA256 digests.')

    def is_downloaded(self):
        # Cheap readiness; construction always performs cryptographic validation.
        node = self._node()
        try:
            image_root = self._image_root()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        components = [(image_root, self._lock['components']['image'])]
        if node.endswith('-enhanced'):
            components.append((self.model_dir / 'prompt_enhancer', self._lock['components']['i2i' if node.startswith('edit') else 't2i']))
        for root, component in components:
            for name, expected in component['files'].items():
                try:
                    path = self._verifier._path(root, name)
                    if not path.is_file() or path.stat().st_size != expected['size']: return False
                except (OSError, RuntimeError): return False
        return True

    def _auto_download(self):
        raise RuntimeError('Download this node in Modly Models. The generator never downloads weights.')

    def is_loaded(self):
        return self._model is not None or self._enhancer is not None

    @staticmethod
    def _release_memory():
        gc.collect()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()

    def _safe_release_memory(self):
        try:
            self._release_memory()
        except Exception as exc:
            self._log(f'WARNING: allocator cleanup failed: {type(exc).__name__}: {exc}')

    def _release_image(self):
        # Drop ownership before invoking fallible native hooks. A hook exception
        # must not prevent independent enhancer/processor cleanup or mask errors.
        pipe, self._model = self._model, None
        self._memory_mode = None
        try:
            if pipe is not None and hasattr(pipe, 'maybe_free_model_hooks'):
                pipe.maybe_free_model_hooks()
        except Exception as exc:
            self._log(f'WARNING: image hook cleanup failed: {type(exc).__name__}: {exc}')
        finally:
            pipe = None
            self._safe_release_memory()

    def _finish_image_hooks(self):
        # Complete cleanup before committing an output. On hook failure, discard
        # the resident pipeline but retain the already-produced valid PIL image.
        pipe = self._model
        try:
            if pipe is not None and hasattr(pipe, 'maybe_free_model_hooks'):
                pipe.maybe_free_model_hooks()
        except Exception as exc:
            self._log(f'WARNING: post-inference hook cleanup failed; dropping pipeline: {type(exc).__name__}: {exc}')
            self._model = None
            self._memory_mode = None
        finally:
            pipe = None
            if self._model is None: self._safe_release_memory()

    def unload(self):
        self._enhancer = self._processor = None
        self._loaded_node = None
        self._release_image()
        self._log('Unloaded image model and prompt enhancer; allocator cleanup attempted.')

    def _discard_output(self, path):
        if path is None: return
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            self._log(f'WARNING: could not remove incomplete output {path}: {exc}')

    @staticmethod
    def _torch():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this extension venv. Run Repair and inspect its Torch/device verification logs. CPU/MPS inference is not supported by this adapter.')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This pinned runtime requires a CUDA device with BF16 support.')
        return torch

    def _load_image(self, memory_mode='offload', cancel_event=None):
        image_root = self._image_root()
        if self._model is not None and self._memory_mode == memory_mode: return
        self._release_image()
        self._offline()
        self._check_cancelled(cancel_event)
        self._log(f'Constructing image pipeline from shared weights {image_root}; BF16 SDPA, memory={memory_mode}.')
        pipe = None
        try:
            torch = self._torch()
            from diffusers import QwenImage21Pipeline
            pipe = QwenImage21Pipeline.from_pretrained(str(image_root), torch_dtype=torch.bfloat16,
                                                      local_files_only=True)
            self._check_cancelled(cancel_event)
            if memory_mode == 'offload': pipe.enable_model_cpu_offload()
            else: pipe.to('cuda')
            pipe.set_progress_bar_config(disable=True)
            self._model = pipe
            self._memory_mode = memory_mode
            self._loaded_node = self._node()
        except BaseException:
            self._model = pipe = None
            self._safe_release_memory()
            raise
        self._log('Image pipeline construction complete.')

    def _load_enhancer(self, cancel_event=None):
        if self._enhancer is not None: return
        self._offline()
        self._check_cancelled(cancel_event)
        root = self.model_dir / 'prompt_enhancer'
        self._log(f'Constructing community reduced-refusal prompt enhancer from {root}.')
        try:
            torch = self._torch()
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self._processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
            self._enhancer = AutoModelForImageTextToText.from_pretrained(
                str(root), local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda').eval()
            self._check_cancelled(cancel_event)
            self._loaded_node = self._node()
        except BaseException:
            self._enhancer = self._processor = None
            self._safe_release_memory()
            raise
        self._log('Prompt enhancer construction complete; image pipeline is not resident.')

    def load(self):
        if self.is_loaded() and self._loaded_node == self._node(): return
        try:
            self.unload()
            self._validate_assets()
            if self._node().endswith('-enhanced'): self._load_enhancer()
            else: self._load_image()
        except BaseException:
            self.unload()
            self._log('Load failed; inspect the following traceback and Repair/download guidance.')
            traceback.print_exc(file=sys.stderr)
            raise

    def _parameters(self, values):
        if not isinstance(values, dict): raise ValueError('Generation parameters must be an object.')
        result = {k: values.get(k, v) for k, v in DEFAULTS.items()}
        for key in ('prompt', 'negative_prompt'):
            if not isinstance(result[key], str): raise ValueError(f'{key} must be text.')
        if not result['prompt'].strip(): raise ValueError('A non-empty prompt is required.')
        result['prompt'] = result['prompt'].strip()
        for key, low, high in [('width', 256, 2048), ('height', 256, 2048), ('output_resolution', 256, 2048),
                               ('steps', 1, 100), ('seed', 0, 2**32 - 1), ('enhancer_max_tokens', 128, 32768)]:
            raw = result[key]
            try:
                parsed = int(raw)
                if isinstance(raw, bool) or float(raw) != parsed or not low <= parsed <= high: raise ValueError()
            except (ValueError, TypeError, OverflowError): raise ValueError(f'{key} must be an integer in [{low}, {high}].') from None
            if key in ('width', 'height', 'output_resolution') and parsed % 32:
                raise ValueError(f'{key} must be a multiple of 32; dimensions are never silently rounded.')
            result[key] = parsed
        try: scale = float(result['true_cfg_scale'])
        except (TypeError, ValueError): raise ValueError('true_cfg_scale must be a finite number.') from None
        if not math.isfinite(scale) or not 1 <= scale <= 10: raise ValueError('true_cfg_scale must be in [1, 10].')
        result['true_cfg_scale'] = scale
        enabled = str(result['use_kv_cache']).lower()
        if enabled not in ('true', 'false'): raise ValueError('use_kv_cache must be true or false.')
        result['use_kv_cache'] = enabled == 'true'
        if result['memory_mode'] not in ('offload', 'cuda'): raise ValueError('Invalid memory_mode.')
        if result['enhancer_thinking'] not in ('off', 'on'): raise ValueError('enhancer_thinking must be off or on.')
        return result

    def _references(self, primary, params):
        if not self._node().startswith('edit'):
            # The text runner supplies a dummy 1x1 image. It is not a reference.
            if params.get('extra_image_paths'): raise ValueError('Use an Edit node to supply image references.')
            return []
        extra = params.get('extra_image_paths', [])
        if not isinstance(extra, (list, tuple)): raise ValueError('extra_image_paths must be a list.')
        if len(extra) > 9: raise ValueError('A maximum of ten references is supported: primary plus nine extra slots.')
        values = [(1, primary)]
        for index, item in enumerate(extra, 2):
            if item is None or item == '': continue
            if not isinstance(item, (str, Path)): raise ValueError(f'Reference slot {index} must be a local image path.')
            values.append((index, item))
        from PIL import Image, ImageOps
        images = []
        for slot, value in values:
            try:
                if isinstance(value, (bytes, bytearray)):
                    if not value: raise ValueError('empty image')
                    source = io.BytesIO(value)
                elif isinstance(value, (str, Path)):
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        workspace = os.environ.get('WORKSPACE_DIR')
                        if not workspace: raise ValueError('WORKSPACE_DIR is required for relative references')
                        path = Path(workspace) / path
                    if not path.is_file(): raise ValueError(f'Image file does not exist: {path}')
                    source = path
                else: raise ValueError('primary reference must be image bytes or a local path')
                with Image.open(source) as im:
                    im.load()
                    oriented = ImageOps.exif_transpose(im)
                    decoded = oriented.convert('RGBA' if ('A' in oriented.getbands() or 'transparency' in oriented.info) else 'RGB')
                images.append(decoded)
            except Exception as exc:
                raise ValueError(f'Cannot decode reference slot {slot}: {exc}') from exc
            self._log(f'Reference slot {slot} -> <image{len(images)}> ({decoded.width}x{decoded.height}, {decoded.mode}).')
        return images

    def _enhance(self, prompt, references, params, report, cancel_event):
        self._release_image()  # Never co-reside main pipeline and PE.
        mode, budget = params['enhancer_thinking'], params['enhancer_max_tokens']
        thinking = mode == 'on'
        report(12, f'Loading community prompt enhancer; mode={mode}, budget={budget}')
        self._load_enhancer(cancel_event)
        model, processor = self._enhancer, self._processor
        inputs = output = None
        try:
            from PIL import Image
            torch = self._torch()
            from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
            system = (self.model_dir / 'prompt_enhancer' / 'system_prompt.txt').read_text(encoding='utf-8')
            pe_images = []
            for original in references:
                image = original.copy().convert('RGBA')
                white = Image.new('RGBA', image.size, 'white'); white.alpha_composite(image)
                image = white.convert('RGB')
                if image.width * image.height > 1024 * 1024:
                    scale = math.sqrt(1024 * 1024 / (image.width * image.height))
                    image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
                pe_images.append(image)
            content = [{'type': 'image', 'image': im} for im in pe_images]
            content.append({'type': 'text', 'text': prompt})
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]}, {'role': 'user', 'content': content}]
            report(20, 'Encoding prompt-enhancer text and ordered references')
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors='pt', enable_thinking=thinking).to(model.device)
            if 'mm_token_type_ids' not in inputs and hasattr(processor, 'create_mm_token_type_ids'):
                inputs['mm_token_type_ids'] = processor.create_mm_token_type_ids(inputs['input_ids'])
            prompt_length = inputs['input_ids'].shape[1]
            processors = LogitsProcessorList()
            if not references: processors.append(PresencePenalty(1.5, prompt_length))
            owner = self
            started = time.monotonic()
            class CancelAndProgress(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    owner._check_cancelled(cancel_event)
                    count = input_ids.shape[1] - prompt_length
                    if count == 1 or count % 128 == 0:
                        report(22 + min(10, int(10 * count / params['enhancer_max_tokens'])),
                               f'Enhancing prompt: {count} generated tokens ({time.monotonic()-started:.0f}s)')
                    if not thinking and count > 0 and count % 32 == 0:
                        suffix = processor.tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True)
                        try: parse_enhancement(suffix, len(references), thinking=False)
                        except ValueError: pass  # Incomplete or invalid first object; final parse reports the error.
                        else: return True
                    return False
            torch.manual_seed(params['seed'])
            report(22, f'Generating enhanced prompt; mode={mode}, input tokens={prompt_length}, budget={budget}')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=params['enhancer_max_tokens'], do_sample=True,
                        temperature=1.0, top_p=0.95, top_k=20, logits_processor=processors,
                        stopping_criteria=StoppingCriteriaList([CancelAndProgress()]), pad_token_id=processor.tokenizer.eos_token_id)
            self._check_cancelled(cancel_event)
            generated_count = output.shape[1] - prompt_length
            text = processor.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
            result = parse_enhancement(text, len(references), thinking=thinking)
            self._log(f'Enhancement complete ({generated_count} tokens). Aspect suggestion: {result["wh_ratio"] or result["ratio_follow"]}; requested {params["width"]}x{params["height"]} is unchanged.')
            report(35, 'Prompt enhancement complete; releasing enhancer memory')
            return result
        finally:
            self._enhancer = self._processor = None
            model = processor = inputs = output = None
            self._safe_release_memory()

    @staticmethod
    def _torch_generator(seed):
        import torch
        return torch.Generator(device='cpu').manual_seed(seed)

    @contextmanager
    def _stage_hooks(self, report, cancel_event):
        originals = []
        def wrap(obj, name, pct, label, only_once=False):
            if obj is None or not hasattr(obj, name): return
            original = getattr(obj, name)
            seen = False
            def wrapped(*args, **kwargs):
                nonlocal seen
                self._check_cancelled(cancel_event)
                if not (only_once and seen): report(pct, label)
                seen = True
                result = original(*args, **kwargs)
                self._check_cancelled(cancel_event)
                return result
            originals.append((obj, name, original))
            setattr(obj, name, wrapped)
        try:
            wrap(self._model, 'encode_prompt', 45, 'Encoding image-model prompt and reference vision features')
            wrap(getattr(self._model, 'vae', None), 'encode', 46, 'Encoding reference image latents')
            wrap(getattr(self._model, 'transformer', None), 'forward', 55, 'Denoising: transformer prefill', True)
            wrap(getattr(self._model, 'vae', None), 'decode', 93, 'Decoding generated image')
            yield
        finally:
            for obj, name, original in reversed(originals): setattr(obj, name, original)

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None) -> Path:
        last_pct = -1
        final = temporary = None
        def report(percent, label):
            nonlocal last_pct
            percent = max(last_pct, min(100, int(percent)))
            last_pct = percent
            self._log(label)
            if progress_cb: progress_cb(percent, label)
        try:
            self._check_cancelled(cancel_event)
            report(0, 'Validating parameters and reference images')
            p = self._parameters(params)
            references = self._references(image_bytes, params)
            if self._loaded_node is not None and self._loaded_node != self._node(): self.unload()
            report(3, 'Validating complete pinned checkpoint assets')
            self._validate_assets(cancel_event)
            prompt = p['prompt']
            enhanced = None
            if self._node().endswith('-enhanced'):
                enhanced = self._enhance(prompt, references, p, report, cancel_event)
                prompt = enhanced['rewritten_prompt']
            self._check_cancelled(cancel_event)
            report(38, 'Loading image pipeline (local-only BF16 SDPA)')
            if self._model is None or self._memory_mode != p['memory_mode']:
                self._load_image(p['memory_mode'], cancel_event)
            self._check_cancelled(cancel_event)
            if p['negative_prompt'] and p['true_cfg_scale'] == 1:
                self._log('Negative prompt is inactive at true_cfg_scale=1; set CFG above 1 to use it.')
            self._log(f'Inference: refs={len(references)}, seed={p["seed"]}, steps={p["steps"]}, size={p["width"]}x{p["height"]}, reference resolution={p["output_resolution"]}, KV cache={p["use_kv_cache"]}.')
            def step_callback(pipe, index, timestep, callback_kwargs):
                self._check_cancelled(cancel_event)
                report(55 + int(36 * (index + 1) / p['steps']), f'Denoising step {index + 1}/{p["steps"]}')
                return callback_kwargs
            report(42, 'Preparing image-model text and reference encoding')
            with self._stage_hooks(report, cancel_event):
                result = self._model(prompt=prompt, image=references or None,
                    negative_prompt=p['negative_prompt'] or None, true_cfg_scale=p['true_cfg_scale'],
                    width=p['width'], height=p['height'], output_resolution=p['output_resolution'],
                    num_inference_steps=p['steps'], generator=self._torch_generator(p['seed']),
                    use_kv_cache=p['use_kv_cache'], output_type='pil',
                    callback_on_step_end=step_callback, callback_on_step_end_tensor_inputs=['latents'])
            self._check_cancelled(cancel_event)
            image = result.images[0]
            if image.size != (p['width'], p['height']):
                raise RuntimeError(f'Pipeline output dimensions {image.size} differ from requested {(p["width"], p["height"])}.')
            self._finish_image_hooks()
            self._check_cancelled(cancel_event)
            report(96, 'Writing and validating atomic PNG output')
            from PIL import Image, PngImagePlugin
            self.outputs_dir = Path(self.outputs_dir).absolute()
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            final = self.outputs_dir / f'qwen-image-2-1-{token}.png'
            temporary = self.outputs_dir / f'.qwen-image-2-1-{token}.png.tmp'
            metadata = PngImagePlugin.PngInfo()
            record = {'node': self._node(), 'original_prompt': p['prompt'], 'effective_prompt': prompt,
                      'parameters': p, 'reference_count': len(references), 'enhancement': enhanced,
                      'image_revision': self._lock['components']['image']['revision']}
            metadata.add_text('qwen_image_2_1', json.dumps(record, ensure_ascii=False))
            with temporary.open('xb') as output:
                image.save(output, format='PNG', pnginfo=metadata)
                output.flush(); os.fsync(output.fileno())
            with Image.open(temporary) as check: check.verify()
            self._check_cancelled(cancel_event)
            os.replace(temporary, final)
            self._check_cancelled(cancel_event)
            report(100, f'Image ready: {final.name}')
            self._check_cancelled(cancel_event)
            return final
        except GenerationCancelled:
            self._discard_output(final)
            self._log('Generation cancelled; no successful output is delivered.')
            self.unload()
            raise
        except Exception:
            self._discard_output(final)
            self._log('Generation failed; image/PE state will be released. Full diagnostic traceback follows.')
            traceback.print_exc(file=sys.stderr)
            self.unload()
            raise
        finally:
            self._discard_output(temporary)
