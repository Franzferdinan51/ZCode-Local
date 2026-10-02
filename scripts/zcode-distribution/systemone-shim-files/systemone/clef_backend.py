"""Clef backend: judge locally with Cloudflare's clef / clef-flash weights.

Clef (``Cloudflare/clef``, 27B) and Clef-Flash (``Cloudflare/clef-flash``,
9B) are Apache-2.0 multimodal decision models post-trained from Qwen3.8-27B
and Qwen3.5-9B. Each release is a sharded backbone plus a small joint schema
head (``joint_head.safetensors``) and a ``joint_schema_model.py`` whose
``systemone(model, processor, request)`` answers a Jev/SystemOne
``POST /v1/systemone`` body with the same response body — no text is
generated, no output is parsed, one logit per allowed option per question.

This backend loads a release through that same module (imported from the
downloaded snapshot, never from ``sys.path``) and maps its answers onto
SystemOne answers with the shared reference confidence:

- choice answers: ``{choice, confidence, probabilities}`` keyed by option
  name, in criteria order.
- score answers: probabilities keyed by level INDEX (``"0"``-``"N"``) with
  a ``{"index": level}`` legend — remapped positionally here.
- noul answers: ``{"noul": P(true)}`` only.
- ``usage`` is ``{input_tokens, output_tokens: 0}`` (decisions emit no
  completion tokens); the input count rides along in ``_meta``.

Media: images may be PIL images, http(s) URLs, data URLs, or local paths
(they are decoded to PIL before the forward pass); videos must be frame
arrays (numpy). Anything undecodable is reported in
``_meta["media_dropped"]`` with reasons and never fails the request.

Wire notes (joint_schema_model.py, Cloudflare/clef, verified 2026-10-02):
- ``load_release_model(path, device="cuda", dtype=torch.bfloat16)`` takes
  either a repo id or a local snapshot dir; a dir skips re-download.
- ``encode_record`` defaults to ``max_length=16384``; over-limit states
  are truncated state-side (schema always fits or raises loudly).
- ``question_options`` sorts choice criteria for encoding, but
  ``systemone_answer`` keys probabilities in criteria order — so the
  caller's option order is preserved end to end.

Heavy deps (torch/transformers/hf_hub/safetensors/pillow) import lazily:
this module stays importable on a slim install, and ``ClefBackend()``
raises a helpful ImportError naming ``systemone[clef]`` when they are
missing. Loading ``joint_schema_model.py`` executes the release's own
modeling code (the repo is tagged ``custom-code``); pin
``CLEF_REVISION`` (or ``SYSTEMONE_REVISION``) in deploys that need
reproducible, reviewable code.

Env: CLEF_MODEL_ID (default Cloudflare/clef-flash — the runnable one;
  Cloudflare/clef wants datacentre VRAM), CLEF_REVISION,
  CLEF_DEVICE (default auto: cuda > mps > cpu), CLEF_DTYPE
  (default bfloat16), CLEF_MAX_LENGTH (default 16384),
  CLEF_IMAGE_TIMEOUT (default 30s), CLEF_IMAGE_MAX_MB (default 64).
"""

from __future__ import annotations

import base64
import binascii
import importlib.util
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Sequence

#: Release small enough for workstation GPUs; the 27B needs ~55GB+ VRAM.
DEFAULT_MODEL_ID = "Cloudflare/clef-flash"

_DTYPE_ALIASES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16", "half": "float16",
    "float32": "float32", "fp32": "float32", "float": "float32",
}


class ClefError(Exception):
    """Clef load, media, or contract failure (message is sanitized)."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        self.hint = hint
        super().__init__(f"{message} {hint}".strip() if hint else message)


def _pil_image(item: Any, *, timeout: float, max_bytes: int) -> Any:
    """Decode one image reference to a PIL image.

    Accepts PIL images (passthrough), http(s) URLs (downloaded),
    ``data:image/...;base64,...`` URLs, and local file paths.
    Raises ClefError when the reference cannot be decoded.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ClefError(
            "image inputs need pillow, which is not installed",
            hint="pip install 'systemone[clef]' to add it",
        ) from None
    if isinstance(item, Image.Image):
        return item
    blob: bytes | None = None
    if isinstance(item, str) and item.startswith(("http://", "https://")):
        from .patterns import require_http_url

        url = require_http_url(item, what="Clef image URL")
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 -- scheme enforced above via patterns.require_http_url; nosemgrep
                blob = resp.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise ClefError(
                f"image download failed (HTTP {exc.code})",
                hint=f"checked {url}",
            ) from None
        except Exception as exc:
            raise ClefError(
                f"could not download image: {type(exc).__name__}",
                hint=f"checked {url}",
            ) from None
        if len(blob) > max_bytes:
            raise ClefError(
                f"image exceeds the {max_bytes // (1024 * 1024)} MB cap",
                hint="raise CLEF_IMAGE_MAX_MB or downscale the image",
            )
    elif isinstance(item, str) and item.startswith("data:"):
        header, _, payload = item.partition(",")
        if ";base64" not in header or not payload:
            raise ClefError("unsupported data URL (need base64 image data)")
        mime = header[5:].split(";")[0]
        if mime and not mime.startswith("image/"):
            raise ClefError(f"unsupported data URL media type {mime!r}")
        try:
            blob = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise ClefError("image data URL is not valid base64") from None
        if len(blob) > max_bytes:
            raise ClefError(
                f"image exceeds the {max_bytes // (1024 * 1024)} MB cap",
                hint="raise CLEF_IMAGE_MAX_MB or downscale the image",
            )
    else:
        try:
            path = Path(str(item)).expanduser()
            if len(path.name) == 0 or not path.is_file():
                raise ClefError(f"image path not found: {item!r:.120}")
            if path.stat().st_size > max_bytes:
                raise ClefError(
                    f"image exceeds the {max_bytes // (1024 * 1024)} MB cap",
                    hint="raise CLEF_IMAGE_MAX_MB or downscale the image",
                )
            blob = path.read_bytes()
        except ClefError:
            raise
        except OSError as exc:
            raise ClefError(
                f"could not read image {item!r:.120}",
                hint=f"{type(exc).__name__}: pass a URL, data URL, or path",
            ) from None
    try:
        import io

        image = Image.open(io.BytesIO(blob))
        image.load()
        return image
    except Exception:
        raise ClefError("bytes did not decode as an image") from None


def _video_frames(item: Any) -> Any:
    """Validate one video reference: frame arrays pass through as-is."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a base dependency
        raise ClefError("video inputs need numpy") from None
    if isinstance(item, np.ndarray):
        return item
    if isinstance(item, (list, tuple)) and item and all(
        isinstance(frame, np.ndarray) for frame in item
    ):
        return list(item)
    raise ClefError(
        f"unsupported video reference {type(item).__name__}",
        hint="pass frame arrays (numpy); URLs/paths are not decoded",
    )


class ClefBackend:
    """Drop-in local engine over Cloudflare clef / clef-flash weights."""

    backend_name = "clef"

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        max_length: int | None = None,
        image_timeout: float | None = None,
        image_max_mb: float | None = None,
    ) -> None:
        from .patterns import resolve_revision

        self.model_id = (
            model_id or os.environ.get("CLEF_MODEL_ID") or DEFAULT_MODEL_ID
        ).strip()
        self.model_name = self.model_id
        self.revision = resolve_revision(
            (revision or os.environ.get("CLEF_REVISION") or "").strip() or None
        )
        self.device_name = (
            device or os.environ.get("CLEF_DEVICE") or "auto"
        ).strip().lower()
        dtype_name = (dtype or os.environ.get("CLEF_DTYPE") or "bfloat16").strip().lower()
        if dtype_name not in _DTYPE_ALIASES:
            raise ClefError(
                f"unknown CLEF_DTYPE {dtype_name!r}",
                hint="want one of: bfloat16, float16, float32",
            )
        self.dtype_name = _DTYPE_ALIASES[dtype_name]
        max_length_env = (os.environ.get("CLEF_MAX_LENGTH") or "").strip()
        try:
            self.max_length = (
                max_length
                if max_length is not None
                else (int(max_length_env) if max_length_env else 16384)
            )
        except ValueError:
            raise ClefError(
                f"bad CLEF_MAX_LENGTH {max_length_env!r}",
                hint="want an integer token budget",
            ) from None
        if self.max_length <= 0:
            raise ClefError(
                f"bad CLEF_MAX_LENGTH {self.max_length}",
                hint="want a positive integer token budget",
            )
        timeout_env = (os.environ.get("CLEF_IMAGE_TIMEOUT") or "").strip()
        try:
            self.image_timeout = (
                image_timeout
                if image_timeout is not None
                else (float(timeout_env) if timeout_env else 30.0)
            )
        except ValueError:
            raise ClefError(
                f"bad CLEF_IMAGE_TIMEOUT {timeout_env!r}",
                hint="want a number of seconds",
            ) from None
        max_mb_env = (os.environ.get("CLEF_IMAGE_MAX_MB") or "").strip()
        try:
            max_mb = (
                image_max_mb
                if image_max_mb is not None
                else (float(max_mb_env) if max_mb_env else 64.0)
            )
        except ValueError:
            raise ClefError(
                f"bad CLEF_IMAGE_MAX_MB {max_mb_env!r}",
                hint="want a number of megabytes",
            ) from None
        if max_mb <= 0:
            raise ClefError(
                f"bad CLEF_IMAGE_MAX_MB {max_mb}",
                hint="want a positive number of megabytes",
            )
        self.image_max_bytes = int(max_mb * 1024 * 1024)
        self._model: Any = None
        self._processor: Any = None
        self._csm: Any = None
        self._load()

    # -- loading --------------------------------------------------------

    def _load(self) -> None:
        try:
            import torch
        except ImportError:
            raise ClefError(
                "the clef engine needs torch, which is not installed",
                hint="pip install 'systemone[clef]' to add it",
            ) from None
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ClefError(
                "the clef engine needs huggingface_hub, which is not installed",
                hint="pip install 'systemone[clef]' to add it",
            ) from None
        device = self._resolve_device(torch)
        try:
            snap = snapshot_download(
                self.model_id,
                revision=self.revision,
                allow_patterns=[
                    "*.safetensors", "*.json", "tokenizer*",
                    "chat_template.jinja", "processor_config.json",
                    "joint_schema_model.py",
                ],
            )
        except Exception as exc:
            raise ClefError(
                f"could not download {self.model_id}: {type(exc).__name__}",
                hint=f"{exc}; check CLEF_MODEL_ID/CLEF_REVISION and network",
            ) from None
        joint_path = Path(snap) / "joint_schema_model.py"
        if not joint_path.is_file():
            raise ClefError(
                f"{self.model_id} has no joint_schema_model.py",
                hint="CLEF_MODEL_ID must be a Clef release (clef/clef-flash)",
            )
        spec = importlib.util.spec_from_file_location(
            "systemone_clef_joint_schema", joint_path
        )
        if spec is None or spec.loader is None:
            raise ClefError(
                "could not load joint_schema_model.py from the snapshot",
                hint=f"checked {joint_path}",
            )
        csm = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(csm)
        except ImportError as exc:
            raise ClefError(
                f"the clef release needs {exc.name or 'a dependency'}, "
                "which is not installed",
                hint="pip install 'systemone[clef]' to add it",
            ) from None
        dtype = getattr(torch, self.dtype_name, None)
        if dtype is None:  # pragma: no cover - guarded by __init__ validation
            raise ClefError(f"torch has no dtype {self.dtype_name!r}")
        try:
            model, processor = csm.load_release_model(
                snap, device=device, dtype=dtype)
        except Exception as exc:
            raise ClefError(
                f"could not load {self.model_id}: {type(exc).__name__}",
                hint=f"{exc}",
            ) from None
        self._model = model
        self._processor = processor
        self._csm = csm

    def _resolve_device(self, torch: Any) -> str:
        if self.device_name not in ("auto", "cuda", "mps", "cpu"):
            raise ClefError(
                f"unknown CLEF_DEVICE {self.device_name!r}",
                hint="want one of: auto, cuda, mps, cpu",
            )
        if self.device_name != "auto":
            return self.device_name
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def health(self) -> bool:
        """True once the release is loaded (load happens in __init__)."""
        return self._model is not None

    # -- question rendering ---------------------------------------------

    @staticmethod
    def _to_clef(q: Dict[str, Any]) -> Dict[str, Any]:
        name = q.get("name", "q")
        qtype = q.get("type")
        if qtype == "choice":
            options = list(q.get("options") or [])
            if len(options) < 2:
                raise ClefError(f"question {name!r} needs >= 2 options")
            return {
                "type": "choice",
                "instructions": q.get("prompt") or f"Choose the best option ({name})",
                # Insertion order is the caller's option order; Clef keys
                # answer probabilities in criteria order (systemone_answer).
                "criteria": {
                    str(o.get("name", o) if isinstance(o, dict) else o): (
                        o.get("description") if isinstance(o, dict) else None
                    ) for o in options
                },
            }
        if qtype == "score":
            levels = [
                str(lv.get("name", lv) if isinstance(lv, dict) else lv)
                for lv in (q.get("levels") or [])
            ]
            if len(levels) < 2:
                raise ClefError(f"question {name!r} needs >= 2 levels")
            return {
                "type": "score",
                "instructions": q.get("prompt") or f"Rate the level ({name})",
                "criteria": levels,
            }
        if qtype == "noul":
            statement = q.get("statement") or q.get("prompt") or name
            return {"type": "noul", "instructions": statement}
        raise ClefError(f"question {name!r}: unknown type {qtype!r}")

    @staticmethod
    def _from_clef(
        name: str,
        qtype: str,
        answer: Dict[str, Any],
        labels: Sequence[str],
    ) -> Dict[str, Any]:
        from .patterns import (
            choice_confidence,
            noul_confidence,
            score_confidence,
            validate_choice,
            validate_distribution,
        )

        if qtype == "choice":
            probs = {str(k): float(v) for k, v in
                     (answer.get("probabilities") or {}).items()}
            choice = answer.get("choice") or (
                max(probs, key=probs.get) if probs else None)
            if choice is None:
                raise ClefError(f"question {name!r}: empty choice answer")
            validate_choice({"choice": choice, "probabilities": probs},
                            list(probs))
            return {"type": "choice", "choice": choice,
                    "probabilities": probs,
                    "confidence": choice_confidence(list(probs.values()))}
        if qtype == "score":
            raw = {str(k): float(v) for k, v in
                   (answer.get("probabilities") or {}).items()}
            # Clef keys by level index; remap positionally via the legend
            # when present, else by the caller's level order.
            legend = answer.get("legend") or {}
            if all(str(i) in raw for i in range(len(labels))):
                dist = {labels[i]: raw[str(i)] for i in range(len(labels))}
            elif set(raw) == set(labels):
                dist = {lv: raw[lv] for lv in labels}
            elif legend and set(raw) == set(legend):
                inv = {str(v): str(k) for k, v in legend.items()}
                dist = {lv: raw[inv[lv]] for lv in labels if lv in inv}
                if set(dist) != set(labels):
                    raise ClefError(
                        f"question {name!r}: legend does not cover levels")
            else:
                raise ClefError(
                    f"question {name!r}: score probabilities match "
                    "neither level indices nor names")
            level = max(dist, key=dist.get) if dist else None
            if level is None:
                raise ClefError(f"question {name!r}: empty score answer")
            validate_distribution(dist, list(dist), level)
            return {"type": "score", "level": level, "distribution": dist,
                    "score": sum(i * p for i, p in enumerate(dist.values())),
                    "confidence": score_confidence(list(dist.values())),
                    "legend": {lv: legend.get(str(i), lv)
                               for i, lv in enumerate(labels)} if legend
                    else {lv: lv for lv in labels}}
        # noul: Clef answers carry "noul" = P(true), and nothing else.
        try:
            p = float(answer.get("noul", answer.get("probability", 0.5)))
        except (TypeError, ValueError):
            raise ClefError(f"question {name!r}: bad noul answer") from None
        p = max(0.0, min(1.0, p))
        return {"type": "noul", "probability": p, "answer": p >= 0.5,
                "confidence": noul_confidence(p)}

    # -- judging --------------------------------------------------------

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer through the release's own ``systemone()`` forward pass.

        Text, JSON, images, and video states all judge jointly. Media that
        cannot be decoded is reported in ``_meta["media_dropped"]`` (with
        reasons) and the request still judges on the rest.
        """
        questions = list(questions)
        if not questions:
            raise ClefError("questions must be non-empty")
        pil_images: list = []
        drop_reasons: list = []
        for item in list(images or []):
            try:
                pil_images.append(_pil_image(
                    item, timeout=self.image_timeout,
                    max_bytes=self.image_max_bytes))
            except ClefError as exc:
                drop_reasons.append(f"image: {exc}")
        frames: list = []
        for item in list(videos or []):
            try:
                frames.append(_video_frames(item))
            except ClefError as exc:
                drop_reasons.append(f"video: {exc}")
        request: Dict[str, Any] = {
            "model": self.model_id,
            "state": state,
            "questions": {q.get("name", f"q{i}"): self._to_clef(q)
                          for i, q in enumerate(questions)},
        }
        if pil_images:
            request["images"] = pil_images
        if frames:
            request["videos"] = frames
        types = {q.get("name", f"q{i}"): q.get("type")
                 for i, q in enumerate(questions)}
        labels: Dict[str, list] = {}
        for i, q in enumerate(questions):
            nm = q.get("name", f"q{i}")
            if q.get("type") == "score":
                labels[nm] = [
                    str(lv.get("name", lv) if isinstance(lv, dict) else lv)
                    for lv in (q.get("levels") or [])]
            else:
                labels[nm] = []
        t0 = time.perf_counter()
        try:
            body = self._csm.systemone(
                self._model, self._processor, request,
                max_length=self.max_length)
        except ValueError as exc:
            raise ClefError(f"the release refused the request: {exc}") from None
        except Exception as exc:
            raise ClefError(
                f"the forward pass failed: {type(exc).__name__}",
                hint=f"{exc}",
            ) from None
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        raw = body.get("answers") or {}
        answers: Dict[str, Any] = {}
        for name, qtype in types.items():
            ans = raw.get(name)
            if not isinstance(ans, dict):
                raise ClefError(f"question {name!r}: missing answer")
            answers[name] = self._from_clef(name, qtype, ans, labels[name])
        meta: Dict[str, Any] = {"model": self.model_name, "backend": "clef",
                                "n_questions": len(questions),
                                "latency_ms": latency_ms}
        if isinstance(body.get("usage"), dict):
            meta["usage"] = dict(body["usage"])
            try:
                meta["input_tokens"] = int(body["usage"].get("input_tokens", 0))
            except (TypeError, ValueError):
                pass
        dropped_images = len(list(images or [])) - len(pil_images)
        dropped_videos = len(list(videos or [])) - len(frames)
        if dropped_images or dropped_videos:
            meta["media_dropped"] = {"images": dropped_images,
                                     "videos": dropped_videos,
                                     "reasons": drop_reasons}
        answers["_meta"] = meta
        return answers
