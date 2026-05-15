from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence


OPENAI_COMPAT_BACKENDS = {"vllm_local", "sglang_local"}


def _normalise_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def infer_llm_model_family(model_id: str, configured: str = "auto") -> str:
    value = str(configured or "auto").strip().lower()
    if value != "auto":
        return value
    lowered = str(model_id or "").lower()
    if "qwen3.6" in lowered or "qwen3_6" in lowered or "qwen3-6" in lowered:
        return "qwen3_6"
    return "causal_lm"


def load_chat_processor_and_model(
    *,
    model_id: str,
    token: str | None,
    torch_dtype: Any = None,
    device: Any = None,
    model_family: str = "auto",
    device_map_auto: bool = True,
    quantization: str | None = None,
) -> Any:
    from transformers import AutoModelForCausalLM, AutoProcessor

    family = infer_llm_model_family(model_id, model_family)
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model_kwargs: Dict[str, Any] = {"token": token}
    quantization = _normalise_optional_string(quantization)
    if quantization:
        if quantization.lower() not in {"nf4", "bnb_nf4", "bitsandbytes_nf4", "4bit", "bnb_4bit"}:
            raise ValueError(f"Unsupported Transformers quantization: {quantization}")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Transformers bitsandbytes fallback requires a transformers build with BitsAndBytesConfig."
            ) from exc
        bnb_kwargs: Dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        if torch_dtype is not None and str(torch_dtype) != "auto":
            bnb_kwargs["bnb_4bit_compute_dtype"] = torch_dtype
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            **bnb_kwargs,
        )
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device is not None and str(device).startswith("cuda") and device_map_auto:
        model_kwargs["device_map"] = "auto"

    if family == "qwen3_6":
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.6 chat checkpoints require a transformers version with "
                "AutoModelForImageTextToText. Use the vllm backend or upgrade transformers."
            ) from exc
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    elif family == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    else:
        raise ValueError(f"Unsupported LLM model family: {family}")

    if device is not None and not str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()
    return processor, model, family


def append_vllm_common_args(
    cmd: list[str],
    *,
    dtype: str | None,
    trust_remote_code: bool,
    tokenizer_mode: str | None = None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    enable_prefix_caching: bool | None = None,
) -> None:
    if dtype is not None and str(dtype).strip() and str(dtype).strip() != "auto":
        cmd.extend(["--dtype", str(dtype).strip()])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    tokenizer_mode = _normalise_optional_string(tokenizer_mode)
    if tokenizer_mode:
        cmd.extend(["--tokenizer-mode", tokenizer_mode])
    if max_num_batched_tokens is not None and int(max_num_batched_tokens) > 0:
        cmd.extend(["--max-num-batched-tokens", str(int(max_num_batched_tokens))])
    if max_num_seqs is not None and int(max_num_seqs) > 0:
        cmd.extend(["--max-num-seqs", str(int(max_num_seqs))])
    if enable_prefix_caching is True:
        cmd.append("--enable-prefix-caching")


def get_visible_gpu_info() -> Any:
    try:
        import torch
    except Exception:
        return SimpleNamespace(count=0, names=[], memory_gb=[], capabilities=[])
    if not torch.cuda.is_available():
        return SimpleNamespace(count=0, names=[], memory_gb=[], capabilities=[])
    names: List[str] = []
    memory_gb: List[float] = []
    capabilities: List[tuple[int, int]] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        names.append(str(torch.cuda.get_device_name(idx)))
        memory_gb.append(float(props.total_memory) / 1024**3)
        capabilities.append(tuple(torch.cuda.get_device_capability(idx)))
    return SimpleNamespace(count=len(names), names=names, memory_gb=memory_gb, capabilities=capabilities)


def infer_model_params_b(model_id: str, configured: Any = None) -> float | None:
    if configured is not None and str(configured).strip() and str(configured).strip().lower() not in {"auto", "none", "null"}:
        return float(configured)
    lowered = str(model_id or "").lower()
    # Prefer the largest explicit "<n>B" marker in names like 35B-A3B; memory has to load all weights.
    matches = [float(x) for x in re.findall(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z])", lowered)]
    if matches:
        return max(matches)
    return None


def choose_auto_backend(
    *,
    model_id: str,
    model_params_b: Any = None,
    allow_sglang: bool = False,
) -> Dict[str, Any]:
    gpu = get_visible_gpu_info()
    params_b = infer_model_params_b(model_id, model_params_b)
    joined_names = " ".join(gpu.names).lower()
    min_mem = min(gpu.memory_gb) if gpu.memory_gb else 0.0
    total_mem = sum(gpu.memory_gb) if gpu.memory_gb else 0.0
    is_hopper_or_newer = any(name in joined_names for name in ("h100", "h200", "b100", "b200"))
    is_a100 = "a100" in joined_names
    bf16_weight_gb = params_b * 2.0 if params_b is not None else None

    selected: Dict[str, Any] = {
        "backend": "transformers",
        "reason": "no_cuda_or_unknown_model_size",
        "tensor_parallel_size": 1,
        "quantization": None,
        "kv_cache_dtype": "auto",
        "gpu_count": gpu.count,
        "gpu_names": gpu.names,
        "gpu_memory_gb": gpu.memory_gb,
        "model_params_b": params_b,
    }
    if gpu.count <= 0:
        selected["reason"] = "no_visible_cuda_device"
        return selected
    if bf16_weight_gb is None:
        selected.update({"backend": "vllm", "reason": "cuda_available_model_size_unknown"})
        return selected
    if min_mem >= bf16_weight_gb * 1.25:
        selected.update({"backend": "vllm", "reason": "bf16_fits_single_gpu"})
        return selected
    if gpu.count > 1 and total_mem >= bf16_weight_gb * 1.35:
        selected.update(
            {
                "backend": "vllm",
                "reason": "bf16_fits_tensor_parallel",
                "tensor_parallel_size": gpu.count,
            }
        )
        return selected
    if is_hopper_or_newer:
        selected.update(
            {
                "backend": "vllm",
                "reason": "hopper_or_newer_fp8_fallback",
                "quantization": "fp8",
                "kv_cache_dtype": "fp8",
                "tensor_parallel_size": min(gpu.count, 2),
            }
        )
        return selected
    if allow_sglang and ("qwen" in str(model_id).lower() or "moe" in str(model_id).lower()):
        selected.update({"backend": "sglang_local", "reason": "sglang_enabled_for_qwen_or_moe"})
        return selected
    if is_a100 or min_mem <= 48:
        selected.update({"backend": "transformers_bnb", "reason": "single_or_low_memory_ampere_nf4_fallback", "quantization": "nf4"})
        return selected
    selected.update({"backend": "transformers_bnb", "reason": "conservative_nf4_fallback", "quantization": "nf4"})
    return selected


def resolve_backend_name(
    *,
    configured_backend: str,
    model_id: str,
    model_params_b: Any = None,
    allow_sglang: bool = False,
) -> Dict[str, Any]:
    backend = str(configured_backend or "transformers").strip()
    if backend != "auto":
        return {"backend": backend, "reason": "configured"}
    return choose_auto_backend(model_id=model_id, model_params_b=model_params_b, allow_sglang=allow_sglang)


def build_openai_chat_client(
    *,
    model_id: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "EMPTY",
    backend: str = "vllm_local",
) -> Any:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenAI-compatible LLM backend requires the `httpx` package.") from exc
    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
    return SimpleNamespace(
        backend=str(backend),
        base_url=f"http://{host}:{int(port)}",
        api_key=str(api_key),
        model_id=str(model_id),
        client=client,
        close=client.close,
    )


def _render_chat_prompt(ctx: Any, messages: List[Dict[str, str]], *, enable_thinking: bool = False) -> str:
    tokenizer = getattr(ctx, "tokenizer", None)
    if tokenizer is None and hasattr(ctx, "llm"):
        tokenizer = ctx.llm.get_tokenizer()
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        rendered = []
        for message in messages:
            role = str(message.get("role", "user")).strip()
            content = str(message.get("content", "") or "")
            rendered.append(f"{role}: {content}")
        rendered.append("assistant:")
        return "\n".join(rendered)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_vllm_offline_chat_model(
    *,
    model_id: str,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    quantization: str | None = None,
    kv_cache_dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
    tokenizer_mode: str | None = None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    enable_prefix_caching: bool | None = None,
    additional_config: Dict[str, Any] | None = None,
) -> Any:
    try:
        from vllm import LLM
    except ModuleNotFoundError as exc:
        raise RuntimeError("stage.runtime.backend=vllm requires the `vllm` package in the active environment.") from exc
    kwargs: Dict[str, Any] = {
        "model": str(model_id),
        "tensor_parallel_size": int(tensor_parallel_size),
        "dtype": str(dtype or "auto"),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": bool(trust_remote_code),
        "enforce_eager": bool(enforce_eager),
    }
    quantization = _normalise_optional_string(quantization)
    kv_cache_dtype = str(kv_cache_dtype or "auto").strip()
    if quantization:
        kwargs["quantization"] = quantization
    if kv_cache_dtype and kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    if max_model_len is not None and int(max_model_len) > 0:
        kwargs["max_model_len"] = int(max_model_len)
    tokenizer_mode = _normalise_optional_string(tokenizer_mode)
    if tokenizer_mode:
        kwargs["tokenizer_mode"] = tokenizer_mode
    if max_num_batched_tokens is not None and int(max_num_batched_tokens) > 0:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if max_num_seqs is not None and int(max_num_seqs) > 0:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    if enable_prefix_caching is not None:
        kwargs["enable_prefix_caching"] = bool(enable_prefix_caching)
    if additional_config:
        kwargs["additional_config"] = dict(additional_config)
    llm = LLM(**kwargs)
    return SimpleNamespace(backend="vllm", model_id=str(model_id), llm=llm, tokenizer=llm.get_tokenizer(), vllm_kwargs=kwargs)


def decode_openai_chat_completion(
    ctx: Any,
    *,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    headers = {"Authorization": f"Bearer {ctx.api_key}"}
    payload = {
        "model": ctx.model_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    resp = ctx.client.post(f"{ctx.base_url}/v1/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("OpenAI-compatible chat response did not include choices")
    message = choices[0].get("message", {})
    return str(message.get("content", "") or "").strip()


def decode_vllm_chat_completion(
    ctx: Any,
    *,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool = False,
) -> str:
    return decode_vllm_chat_completions(
        ctx,
        messages_batch=[messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )[0]


def decode_vllm_chat_completions(
    ctx: Any,
    *,
    messages_batch: Sequence[List[Dict[str, str]]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool = False,
) -> List[str]:
    try:
        from vllm import SamplingParams
    except ModuleNotFoundError as exc:
        raise RuntimeError("stage.runtime.backend=vllm requires the `vllm` package in the active environment.") from exc
    prompts = [_render_chat_prompt(ctx, messages, enable_thinking=enable_thinking) for messages in messages_batch]
    sampling = SamplingParams(
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    outputs = ctx.llm.generate(prompts, sampling)
    texts: List[str] = []
    for output in outputs:
        completions = getattr(output, "outputs", []) or []
        if not completions:
            texts.append("")
        else:
            texts.append(str(getattr(completions[0], "text", "") or "").strip())
    return texts
