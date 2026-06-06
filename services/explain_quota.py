from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import DefaultDict, List

from fastapi import Request


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


class ExplainQuotaTracker:
    """In-memory quota for successful LLM explains per IP (resets on restart)."""

    def __init__(self, max_per_ip: int, window_seconds: float) -> None:
        self.max_per_ip = max_per_ip
        self.window_seconds = window_seconds
        self._successes: DefaultDict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.max_per_ip > 0

    def _prune(self, ip: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._successes[ip] = [t for t in self._successes[ip] if t > cutoff]

    def is_exhausted(self, ip: str) -> bool:
        if not self.enabled:
            return False
        now = time.time()
        with self._lock:
            self._prune(ip, now)
            return len(self._successes[ip]) >= self.max_per_ip

    def record_success(self, ip: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            self._prune(ip, now)
            self._successes[ip].append(now)
