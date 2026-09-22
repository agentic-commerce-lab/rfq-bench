"""Thin HTTP client for a local Laya server.

Talks to the sidecar shape from the laya-integration skill:
``POST {base_url}{predict_path}`` with a JSON body ``{"state": ..., "questions": ...}``
returning Laya's ``predict`` result (``{"answers": {...}, "usage": {...}}``).

Uses only the standard library (no new dependency) and serializes calls with a lock:
one GPU serves one forward pass at a time, so concurrent calls only interleave badly.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

from rfq_bench.agent.laya_settings import LayaSettings


class LayaError(RuntimeError):
    """Raised when the Laya server is unreachable or returns an unusable response."""


class LayaClient:
    """Loopback HTTP client returning Laya's parsed ``predict`` result."""

    def __init__(self, settings: LayaSettings) -> None:
        self._settings = settings
        self._url = settings.base_url.rstrip("/") + "/" + settings.predict_path.lstrip("/")
        self._lock = threading.Lock()

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Send one decision request and return the result dict.

        Raises :class:`LayaError` on a transport failure or non-JSON reply, so the
        run's retry/backoff can re-attempt a transiently-down server.
        """
        body = json.dumps({"state": state, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 (loopback, developer-controlled URL)
            self._url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with (
                self._lock,
                urllib.request.urlopen(req, timeout=self._settings.timeout) as resp,  # noqa: S310
            ):
                raw = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LayaError(f"Laya server unreachable at {self._url}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LayaError(f"Laya returned non-JSON from {self._url}") from exc
        if not isinstance(data, dict):
            raise LayaError(f"Laya returned a non-object response from {self._url}")
        return data
