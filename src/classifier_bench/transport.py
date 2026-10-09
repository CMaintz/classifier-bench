"""One stdlib HTTP transport for every classifier: no retries, an explicit timeout."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from .classifiers import Request


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


class TransportTimeout(Exception):
    """The deadline passed before a full response arrived."""


Transport = Callable[[Request, float], Response]


def _is_timeout(err: BaseException) -> bool:
    reason = getattr(err, "reason", None)
    return isinstance(err, (TimeoutError, socket.timeout)) or isinstance(reason, (TimeoutError, socket.timeout))


def urllib_transport(request: Request, timeout_s: float) -> Response:
    """POST once. HTTP errors come back as a Response; timeouts raise TransportTimeout."""
    data = json.dumps(request.body).encode("utf-8")
    req = urllib.request.Request(request.url, data=data, method="POST", headers=request.headers)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - fixed https endpoints
            return Response(resp.status, dict(resp.headers.items()), resp.read())
    except urllib.error.HTTPError as err:
        return Response(err.code, dict(err.headers.items()) if err.headers else {}, err.read())
    except (urllib.error.URLError, OSError) as err:
        if _is_timeout(err):
            raise TransportTimeout(str(err)) from err
        raise
