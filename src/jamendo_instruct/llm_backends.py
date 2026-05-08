from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List


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
) -> Any:
    from transformers import AutoModelForCausalLM, AutoProcessor

    family = infer_llm_model_family(model_id, model_family)
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model_kwargs: Dict[str, Any] = {"token": token}
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
                "AutoModelForImageTextToText. Use the vllm_local backend or upgrade transformers."
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
) -> None:
    if dtype is not None and str(dtype).strip() and str(dtype).strip() != "auto":
        cmd.extend(["--dtype", str(dtype).strip()])
    if trust_remote_code:
        cmd.append("--trust-remote-code")


def build_openai_chat_client(
    *,
    model_id: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "EMPTY",
) -> Any:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenAI-compatible vLLM backend requires the `httpx` package.") from exc
    return SimpleNamespace(
        backend="vllm_local",
        base_url=f"http://{host}:{int(port)}",
        api_key=str(api_key),
        model_id=str(model_id),
        client=httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)),
    )


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
