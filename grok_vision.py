"""Small xAI/Grok image-understanding adapter for semantic object discovery.

Core tracking does not depend on this module. Grok is opt-in and receives
only sampled JPEG stills, never a continuous video stream.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

API_URL = "https://api.x.ai/v1/responses"
DEFAULT_MODEL = "grok-4.7"
MAX_JPEG_BYTES = 20 * 1024 * 1024

_PROMPT = """You are the semantic vision layer for a physical-object memory app.
Inspect this single room or desk image and list SEARCH-WORTHY MOVABLE PHYSICAL OBJECTS a person may later ask to find.

Rules:
- Ignore people, faces, body parts, walls, ceilings, floors and large fixed furniture.
- Prefer concrete movable possessions: phone, keys, mouse, bottle, notebook, bag, charger, headphones, remote, glasses, books, tools, clothing, etc.
- Do not invent brands or ownership.
- If two objects look similar, keep them separate when their positions differ.
- bbox is normalized [x, y, width, height] in 0..1 relative to the image. Use null if you cannot localize confidently.
- confidence is visual confidence from 0 to 1, not proof that this is the same physical instance seen in another image.
- instance_signature must be a short conservative visible-description signature useful for matching later views.
- location_hint is a short spatial phrase visible in this image.
- aliases should contain common search names, not speculative brands.

Return ONLY JSON with keys scene and objects. Every object must contain:
name, category, description, instance_signature, aliases, location_hint, confidence, bbox.
Return at most 20 objects. If nothing useful is visible, return an empty objects list.
"""


class GrokError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticObject:
    name: str
    category: str
    description: str
    instance_signature: str
    aliases: tuple[str, ...]
    location_hint: str
    confidence: float
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class SemanticScan:
    scene: str
    objects: tuple[SemanticObject, ...]
    model: str


class GrokVision:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float = 120.0):
        self.api_key = (api_key if api_key is not None else os.getenv("XAI_API_KEY", "")).strip()
        self.model = (model or os.getenv("CTRLF_GROK_MODEL") or DEFAULT_MODEL).strip()
        self.timeout = float(timeout)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def analyze_jpeg(self, jpeg: bytes, context: str = "") -> SemanticScan:
        if not self.configured:
            raise GrokError("Grok is not configured. Set XAI_API_KEY before starting the app.")
        if not jpeg:
            raise GrokError("No image data was provided.")
        if len(jpeg) > MAX_JPEG_BYTES:
            raise GrokError("Image is larger than xAI's 20 MiB image-input limit.")

        encoded = base64.b64encode(jpeg).decode("ascii")
        prompt = _PROMPT
        context = " ".join(str(context).split())[:500]
        if context:
            prompt += f"\\nUser-provided scan context: {context}"

        body = {
            "model": self.model,
            "store": False,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
                    {"type": "input_text", "text": prompt},
                ],
            }],
        }
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "real-world-ctrl-f/0.3",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1000]
            raise GrokError(f"xAI request failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GrokError(f"Could not reach xAI: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise GrokError("xAI returned a response that was not valid JSON.") from exc

        text = _extract_output_text(payload)
        try:
            parsed = _parse_json_object(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise GrokError("Grok returned an object inventory that could not be parsed safely.") from exc
        return _validate_scan(parsed, self.model)


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    for item in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        raise GrokError("xAI response did not contain output text.")
    return "\\n".join(parts)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _clean_text(value: Any, fallback: str = "", maximum: int = 180) -> str:
    text = " ".join(str(value or fallback).split())
    return text[:maximum]


def _validate_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    if x + w > 1.02 or y + h > 1.02:
        return None
    return max(0.0, x), max(0.0, y), min(1.0 - x, w), min(1.0 - y, h)


def _validate_scan(payload: dict[str, Any], model: str) -> SemanticScan:
    scene = _clean_text(payload.get("scene"), "room scan", 140)
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise GrokError("Grok object inventory did not include an objects list.")
    objects: list[SemanticObject] = []
    for raw in raw_objects[:20]:
        if not isinstance(raw, dict):
            continue
        name = _clean_text(raw.get("name"), "", 60)
        if len(name) < 2:
            continue
        category = _clean_text(raw.get("category"), "object", 50)
        description = _clean_text(raw.get("description"), name, 180)
        signature = _clean_text(raw.get("instance_signature"), name, 100).casefold().replace(" ", "_")
        signature = re.sub(r"[^\\w.-]+", "_", signature).strip("_") or re.sub(r"\\W+", "_", name.casefold())
        aliases: list[str] = []
        for alias in raw.get("aliases", []) if isinstance(raw.get("aliases"), list) else []:
            alias = _clean_text(alias, "", 60)
            if len(alias) >= 2 and alias.casefold() not in {name.casefold(), *(a.casefold() for a in aliases)}:
                aliases.append(alias)
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        objects.append(SemanticObject(
            name=name,
            category=category,
            description=description,
            instance_signature=signature,
            aliases=tuple(aliases[:6]),
            location_hint=_clean_text(raw.get("location_hint"), "", 100),
            confidence=confidence,
            bbox=_validate_bbox(raw.get("bbox")),
        ))
    return SemanticScan(scene=scene, objects=tuple(objects), model=model)
