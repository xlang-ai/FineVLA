"""API client for calling the LLM Judge (GT extraction + caption alignment)."""

import json
import random
import re
import time
from typing import Optional

import httpx
from openai import OpenAI

from .config import DEFAULT_MODEL, DEFAULT_BASE_URL, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES
from .models import GTExtractionResult, CaptionExtractionResult, AlignmentResult, CapabilityJudgment, DirectAlignmentResult

# Rate limit backoff cap (seconds)
_RATE_LIMIT_BACKOFF_CAP = 120

# Shared httpx client for Gemini native protocol (reused across calls, thread-safe)
_shared_httpx_client: Optional[httpx.Client] = None

def _get_httpx_client() -> httpx.Client:
    """Get or create a shared httpx client (thread-safe, connection pooling)."""
    global _shared_httpx_client
    if _shared_httpx_client is None or _shared_httpx_client.is_closed:
        _shared_httpx_client = httpx.Client(
            timeout=httpx.Timeout(600.0, connect=60.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
    return _shared_httpx_client


def _is_gpt_reasoning_model(model: str) -> bool:
    """Detect GPT reasoning models (o1, o3, o4, gpt-5 etc.)."""
    m = model.lower()
    return (
        m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
        or "gpt-5" in m or "gpt5" in m
    )


def _is_qwen_thinking_model(model: str) -> bool:
    """Detect Qwen models that support thinking/reasoning (qwen3, qwen3.5 etc.)."""
    m = model.lower()
    return "qwen3" in m


def _is_gemini_native_model(model: str) -> bool:
    """Detect models that require Gemini native protocol (contents/parts) via DashScope.

    Includes:
      - vertex_ai.* / ai_studio.* prefixed models (always native)
      - gemini-robotics-* models
      - gemini-3+ models that need native protocol for thinking support
    """
    m = model.lower()
    if m.startswith("vertex_ai.") or m.startswith("ai_studio."):
        return True
    if "gemini-robotics" in m:
        return True
    bare = m.split(".", 1)[-1] if "." in m else m
    if bare.startswith("gemini-3") or bare.startswith("gemini-4"):
        return True
    return False


def create_client(api_key: str, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Create OpenAI client."""
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_gemini_native(
    client: OpenAI,
    system_prompt: str,
    user_text: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sample_id: str = "unknown",
    enable_thinking: bool = False,
) -> str:
    """Call DashScope using Gemini native protocol (contents/parts).

    Used for vertex_ai.*, ai_studio.*, gemini-robotics-*, gemini-3+ models.
    Uses raw httpx POST since the request format is Gemini-native, not OpenAI messages.
    """
    body = {
        "model": model,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
    }

    if system_prompt:
        body["system_instruction"] = {
            "parts": [{"text": system_prompt}]
        }

    # Non-vertex_ai/ai_studio models need this flag to opt into native protocol
    if not (model.lower().startswith("vertex_ai.") or model.lower().startswith("ai_studio.")):
        body["dashscope_extend_params"] = {
            "using_native_protocol": True,
        }

    # Thinking config for gemini-3+ models
    model_suffix = model.lower().split(".", 1)[-1] if "." in model.lower() else model.lower()
    supports_thinking = (
        ("gemini-3" in model_suffix or "gemini-4" in model_suffix)
        and "image-preview" not in model_suffix
    )
    if supports_thinking:
        if enable_thinking:
            body["generationConfig"] = {
                "thinkingConfig": {
                    "thinkingLevel": "high",
                    "includeThoughts": False,
                }
            }
        else:
            body["generationConfig"] = {
                "temperature": temperature,
                "thinkingConfig": {
                    "thinkingLevel": "low",
                    "includeThoughts": False,
                }
            }
    elif "gemini-2.5" in model_suffix:
        body["generationConfig"] = {
            "temperature": temperature,
            "thinkingConfig": {
                "thinkingBudget": -1,
                "includeThoughts": False,
            }
        }

    api_key = client.api_key
    base_url = str(client.base_url).rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    http_client = _get_httpx_client()
    non_rate_limit_failures = 0
    attempt = 0

    while True:
        try:
            resp = http_client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            content = ""

            if "choices" in data:
                choices = data["choices"]
                if not choices:
                    raise Exception("Empty choices in response")
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
            elif "candidates" in data:
                candidates = data["candidates"]
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    content = "".join(p["text"] for p in parts if "text" in p)
            else:
                raise Exception(f"Unexpected response format: {json.dumps(data)[:500]}")

            time.sleep(0.5)
            return (content or "").strip()

        except Exception as e:
            attempt += 1
            error_str = str(e)
            is_rate_limit = (
                "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "rate" in error_str.lower()
            )

            if is_rate_limit:
                base_backoff = min(2 ** min(attempt, 8) * 3, _RATE_LIMIT_BACKOFF_CAP)
                jitter = random.uniform(0, base_backoff * 0.3)
                backoff_time = base_backoff + jitter
                if attempt <= 3:
                    print(f"\n[{sample_id}] Rate-limited (attempt {attempt}, backoff={backoff_time:.1f}s)")
            else:
                non_rate_limit_failures += 1
                backoff_time = 0.5 * non_rate_limit_failures
                if attempt <= 3:
                    print(f"\n[{sample_id}] API failed (attempt {attempt}, {non_rate_limit_failures}/{max_retries}): {e}")
                if non_rate_limit_failures >= max_retries:
                    print(f"\n[{sample_id}] Failed after {attempt} attempts: {e}")
                    return ""

            time.sleep(backoff_time)

    return ""


def call_api(
    client: OpenAI,
    system_prompt: str,
    user_payload: dict,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sample_id: str = "unknown",
    enable_thinking: bool = False,
) -> str:
    """Call LLM API with retry logic. Returns raw response text."""
    user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)

    # Gemini native protocol path (vertex_ai.*, gemini-3+, etc.)
    if _is_gemini_native_model(model):
        return _call_gemini_native(
            client, system_prompt, user_text, model,
            temperature=temperature, max_retries=max_retries,
            sample_id=sample_id, enable_thinking=enable_thinking,
        )

    # OpenAI-compatible path
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    for attempt in range(max_retries):
        try:
            if "gemini" in model.lower():
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "dashscope_extend_params": {"provider": "b"},
                        "stream": False,
                    },
                )
            elif _is_gpt_reasoning_model(model):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={
                        "reasoning_effort": "high",
                        "max_completion_tokens": 65536,
                        "stream": False,
                    },
                )
            elif enable_thinking:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={
                        "enable_thinking": True,
                        "max_completion_tokens": 65536,
                        "stream": False,
                    },
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "max_completion_tokens": 65536,
                        "stream": False,
                    },
                )

            time.sleep(0.5)
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[{sample_id}] Failed after {max_retries} attempts: {e}")
                return ""
            if attempt < 3:
                print(f"\n[{sample_id}] API call failed (attempt {attempt + 1}): {e}")
            time.sleep(0.5)

    return ""


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response, handling code fences and thinking blocks."""
    text = text.strip()
    if not text:
        return None

    # Strip thinking blocks
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try extracting from code fences
    code_blocks = re.findall(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_blocks:
        for block in code_blocks:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _normalize_extraction_keys(parsed: dict, target_key: str) -> dict:
    """Normalize extraction output: map atomic_facts/gt_atomic_facts/caption_atomic_facts -> target_key."""
    for cr in parsed.get("capability_results", []):
        for src in ("atomic_facts", "gt_atomic_facts", "caption_atomic_facts"):
            if src in cr and src != target_key:
                cr[target_key] = cr.pop(src)
                break
    return parsed


def parse_gt_extraction(response_text: str, sample_id: str) -> Optional[GTExtractionResult]:
    """Parse and validate GT extraction response (supports unified extraction.txt output)."""
    parsed = _extract_json_from_text(response_text)
    if parsed is None:
        return None
    try:
        parsed["sample_id"] = sample_id
        _normalize_extraction_keys(parsed, "gt_atomic_facts")
        return GTExtractionResult.model_validate(parsed)
    except Exception:
        return None


def parse_caption_extraction(response_text: str, sample_id: str) -> Optional[CaptionExtractionResult]:
    """Parse and validate caption extraction response (supports unified extraction.txt output)."""
    parsed = _extract_json_from_text(response_text)
    if parsed is None:
        return None
    try:
        parsed["sample_id"] = sample_id
        _normalize_extraction_keys(parsed, "caption_atomic_facts")
        return CaptionExtractionResult.model_validate(parsed)
    except Exception:
        return None


def parse_alignment(response_text: str, sample_id: str) -> Optional[AlignmentResult]:
    """Parse and validate Phase 2 alignment response (legacy, all-at-once)."""
    parsed = _extract_json_from_text(response_text)
    if parsed is None:
        return None
    try:
        parsed["sample_id"] = sample_id  # Ensure consistency
        return AlignmentResult.model_validate(parsed)
    except Exception:
        return None


def parse_direct_alignment(response_text: str, sample_id: str) -> Optional[DirectAlignmentResult]:
    """Parse and validate direct alignment response (Method B)."""
    parsed = _extract_json_from_text(response_text)
    if parsed is None:
        return None
    try:
        parsed["sample_id"] = sample_id
        return DirectAlignmentResult.model_validate(parsed)
    except Exception:
        return None


def _fix_misplaced_omissions(parsed: dict) -> dict:
    """Fix LLM putting omissions inside alignments list.

    Some LLMs write omissions as alignment entries with caption_fact_id=null
    and label="omission". Move them to the omissions list before validation.
    """
    alignments = parsed.get("alignments", [])
    if not alignments:
        return parsed

    clean_alignments = []
    extra_omissions = []
    for a in alignments:
        cap_fid = a.get("caption_fact_id")
        label = a.get("label", "")
        if cap_fid is None or label == "omission":
            extra_omissions.append({
                "gt_fact_id": a.get("gt_fact_id", ""),
                "note": a.get("note", ""),
            })
        else:
            clean_alignments.append(a)

    if extra_omissions:
        parsed["alignments"] = clean_alignments
        parsed.setdefault("omissions", []).extend(extra_omissions)

    return parsed


def parse_capability_judgment(
    response_text: str, sample_id: str, capability: str,
    slot: Optional[str] = None,
) -> Optional[CapabilityJudgment]:
    """Parse LLM response for a per-capability or per-slot alignment call."""
    parsed = _extract_json_from_text(response_text)
    if parsed is None:
        return None
    try:
        parsed = _fix_misplaced_omissions(parsed)
        parsed["sample_id"] = sample_id
        parsed["capability"] = capability
        if slot is not None:
            parsed["slot"] = slot
        return CapabilityJudgment.model_validate(parsed)
    except Exception:
        return None
