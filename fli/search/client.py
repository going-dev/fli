"""HTTP client implementation with impersonation, rate limiting and retry functionality.

This module provides a robust HTTP client that handles:

- User agent impersonation (to mimic a browser)
- Rate limiting (10 requests per second, *globally* across threads)
- Automatic retries with exponential backoff
- Thread-safe session management (one ``curl_cffi`` session per worker thread)
- Error handling

Threading model
---------------

``curl_cffi.requests.Session`` wraps a libcurl handle which is not safe
to share across threads. We keep one session per worker thread using
``threading.local``; the rate-limit budget is shared globally via
:class:`~fli.search._concurrency.TokenBucketRateLimiter` so concurrent
callers cooperate cleanly under Google's 10 req/sec ceiling.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from fli.search._concurrency import TokenBucketRateLimiter
from fli.search.exceptions import (
    SearchClientError,
    SearchConnectionError,
    SearchHTTPError,
    SearchTimeoutError,
)

# ``curl_cffi`` adds ~100ms to import time on first load — we only need
# it once an HTTP request actually fires, so import lazily on first use.
# ``TYPE_CHECKING`` makes the annotation visible to static checkers
# without paying the import cost at runtime.
if TYPE_CHECKING:
    from curl_cffi import requests as _curl_requests

    Response = _curl_requests.Response
    Session = _curl_requests.Session
else:
    Response = "Any"
    Session = "Any"

# Module-level singleton client + lock guarding its lazy initialisation.
# ``get_client()`` uses double-checked locking so concurrent first callers
# can't each construct an independent ``Client`` (each with its own
# ``TokenBucketRateLimiter`` — that would silently double the global
# request budget).
client: Client | None = None
_client_lock = threading.Lock()

# Google's published ceiling.
DEFAULT_CALLS_PER_SECOND = 10

# Request timeout in seconds.  Override with the FLI_TIMEOUT env var.
DEFAULT_TIMEOUT: float = 60.0
_env_timeout = os.environ.get("FLI_TIMEOUT")
if _env_timeout is not None:
    try:
        REQUEST_TIMEOUT: float = float(_env_timeout)
    except ValueError:
        msg = f"FLI_TIMEOUT must be a number of seconds, got: {_env_timeout!r}"
        raise ValueError(msg) from None
    if REQUEST_TIMEOUT <= 0:
        raise ValueError(f"FLI_TIMEOUT must be a positive number, got: {_env_timeout!r}")
else:
    REQUEST_TIMEOUT = DEFAULT_TIMEOUT


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds, got: {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got: {raw!r}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got: {raw!r}")
    return value


# Connection-establishment timeout. A proxy or host we can't connect to quickly
# is doomed, not slow, so fail it fast instead of letting it sit on a worker.
CONNECT_TIMEOUT: float = _env_seconds("FLI_CONNECT_TIMEOUT", 5.0)

# No-progress (low-speed) abort. libcurl aborts a transfer that averages fewer
# than LOW_SPEED_LIMIT bytes/sec over LOW_SPEED_TIME seconds. This kills a
# stalled connection that will never deliver while leaving a slow-but-streaming
# response alone, something a total timeout cannot distinguish. LOW_SPEED_TIME
# must sit ABOVE the upstream's worst no-byte think-time (time-to-first-byte) or
# it will clip valid slow responses, so it defaults off (0) and is enabled per
# deployment once tuned. REQUEST_TIMEOUT stays the absolute backstop.
LOW_SPEED_LIMIT: int = _env_int("FLI_LOW_SPEED_LIMIT", 1)
LOW_SPEED_TIME: int = int(_env_seconds("FLI_LOW_SPEED_TIME", 0.0))

# Retries cover a transient transport blip, not a doomed call. Bound both the
# attempt count and the total wall time so a retry can never multiply how long
# we hold a worker (see _is_retryable_transport_error for which errors qualify).
MAX_ATTEMPTS: int = max(1, _env_int("FLI_MAX_ATTEMPTS", 2))
RETRY_DEADLINE: float = _env_seconds("FLI_RETRY_DEADLINE", REQUEST_TIMEOUT)

# Per-egress-IP request pacing. RPC scraping needs far fewer requests than the
# browser, so we keep each proxy IP slow and let aggregate throughput come from
# rotating across many IPs rather than hammering one. Each proxy is held to one
# request per PER_PROXY_INTERVAL seconds plus up to PER_PROXY_JITTER of jitter,
# which also serialises the parallel expansion workers that share a proxy.
PER_PROXY_INTERVAL: float = float(os.environ.get("FLI_PER_PROXY_INTERVAL", "1.0"))
PER_PROXY_JITTER: float = float(os.environ.get("FLI_PER_PROXY_JITTER", "1.0"))


class _ProxyPacer:
    """Serialises requests per proxy key to a minimum interval plus jitter."""

    def __init__(self) -> None:
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                ready_at = self._next_allowed.get(key, 0.0)
                if now >= ready_at:
                    self._next_allowed[key] = (
                        now + PER_PROXY_INTERVAL + random.uniform(0, PER_PROXY_JITTER)
                    )
                    return
                sleep_for = ready_at - now
            time.sleep(sleep_for)


_proxy_pacer = _ProxyPacer()


def _is_retryable_transport_error(exc: BaseException) -> bool:
    """Decide whether a failed request is worth one bounded retry.

    Retry only an unclassified transport hiccup (e.g. a mid-stream reset),
    never a timeout, a refused/unreachable host, an HTTP error, or an upstream
    gRPC rejection. Retrying those either can't help (a dead proxy refuses
    again, a 4xx/429 won't change on an instant retry) or just multiplies the
    time we hold a worker on a call that already failed, which is the exact
    stall we are trying to avoid. The connect and low-speed timeouts already
    fail the doomed cases fast; this leaves a single bounded retry for a blip.
    """
    return type(exc) is SearchClientError


# Shared retry policy for get/post: a bounded attempt count AND a hard total-time
# ceiling (whichever trips first), applied only to retryable transport errors.
_RETRY = {
    "retry": retry_if_exception(_is_retryable_transport_error),
    "stop": stop_after_attempt(MAX_ATTEMPTS) | stop_after_delay(RETRY_DEADLINE),
    "wait": wait_exponential(multiplier=0.5, max=2.0),
    "reraise": True,
}


class Client:
    """HTTP client with built-in rate limiting, retry and user agent impersonation functionality.

    Sessions are kept per-thread because ``curl_cffi.requests.Session`` is
    not thread-safe — concurrent ``post``/``get`` calls from different
    threads each get their own libcurl handle. The shared
    :class:`TokenBucketRateLimiter` enforces the global 10 req/sec budget
    across all of them.
    """

    DEFAULT_HEADERS = {
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    }

    def __init__(
        self,
        calls_per_second: int = DEFAULT_CALLS_PER_SECOND,
        proxy: str | None = None,
    ):
        """Initialise the rate limiter, per-thread sessions, and optional proxy.

        ``proxy`` (an ``http(s)://`` or ``socks5://`` URL) is applied to every
        request this client makes, including the parallel expansion workers,
        which all share this instance.
        """
        self._sessions = threading.local()
        self._rate_limiter = TokenBucketRateLimiter(calls=calls_per_second, period=1.0)
        self._proxy = proxy

    def _session(self) -> Session:
        """Return this thread's ``Session``, creating it on first use."""
        session = getattr(self._sessions, "session", None)
        if session is None:
            # Deferred import: ``curl_cffi`` is heavy (~100ms cold) and
            # not needed for CLI flows that never hit the network, so
            # only pull it in on the first real request.
            from curl_cffi import CurlOpt
            from curl_cffi import requests as _requests

            # Connect + no-progress options apply to every request on this
            # session: fail fast when we can't connect, or when a transfer
            # stalls, without capping a slow-but-progressing response.
            # REQUEST_TIMEOUT (passed per request) stays the absolute backstop.
            curl_options = {CurlOpt.CONNECTTIMEOUT: int(CONNECT_TIMEOUT)}
            if LOW_SPEED_TIME > 0:
                curl_options[CurlOpt.LOW_SPEED_LIMIT] = LOW_SPEED_LIMIT
                curl_options[CurlOpt.LOW_SPEED_TIME] = LOW_SPEED_TIME
            session = _requests.Session(curl_options=curl_options)
            session.headers.update(self.DEFAULT_HEADERS)
            self._sessions.session = session
        return session

    def __del__(self):
        """Best-effort cleanup of the main-thread session (others die with their thread)."""
        session = getattr(self._sessions, "session", None) if hasattr(self, "_sessions") else None
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 — destruction-time best effort
                pass

    # ------------------------------------------------------------------
    # Request entry points
    # ------------------------------------------------------------------

    @retry(**_RETRY)
    def get(self, url: str, **kwargs: Any) -> Response:
        """Make a rate-limited GET request with automatic retries."""
        if self._proxy is not None:
            _proxy_pacer.wait(self._proxy)
            kwargs.setdefault("proxy", self._proxy)
        else:
            self._rate_limiter.acquire()
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        try:
            response = self._session().get(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            raise _wrap_request_error("GET", url, e) from e

    @retry(**_RETRY)
    def post(self, url: str, **kwargs: Any) -> Response:
        """Make a rate-limited POST request with automatic retries."""
        if self._proxy is not None:
            _proxy_pacer.wait(self._proxy)
            kwargs.setdefault("proxy", self._proxy)
        else:
            self._rate_limiter.acquire()
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        try:
            response = self._session().post(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            raise _wrap_request_error("POST", url, e) from e


def _wrap_request_error(method: str, url: str, exc: BaseException) -> SearchClientError:
    """Map curl-cffi / network errors into our typed ``SearchClientError`` family.

    The CLI surfaces these as short user-facing messages and writes the
    underlying traceback to a log file, so the message here should read
    well on its own.
    """
    # If a typed error somehow escapes the request body (e.g. a future
    # change in ``_session()``), keep its original type instead of
    # downgrading it to the generic fallback below.
    if isinstance(exc, SearchClientError):
        return exc

    # Imported lazily — ``curl_cffi.requests.exceptions`` triggers the
    # full curl-cffi load, which we otherwise defer until first request.
    from curl_cffi.requests import exceptions as curl_exc

    host = _host_from_url(url)
    if isinstance(exc, curl_exc.Timeout):
        return SearchTimeoutError(
            f"Timed out talking to Google Flights ({host}). "
            "The service may be slow or unreachable from your network — "
            "check your connection and try again."
        )
    if isinstance(exc, curl_exc.ConnectionError):
        return SearchConnectionError(
            f"Could not reach Google Flights ({host}). "
            "Check your internet connection or DNS and try again."
        )
    if isinstance(exc, curl_exc.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f" (HTTP {status})" if status else ""
        return SearchHTTPError(
            f"Google Flights returned an error response{suffix}. "
            "The request may be malformed, rate-limited, or blocked.",
            status_code=status,
        )
    # Anything else (including bare CurlError) — keep a clean message but
    # preserve the original via ``__cause__`` so logs still show details.
    return SearchClientError(
        f"{method} request to Google Flights ({host}) failed: {exc.__class__.__name__}"
    )


def _host_from_url(url: str) -> str:
    """Best-effort host extraction for error messages."""
    try:
        from urllib.parse import urlparse

        return urlparse(url).hostname or url
    except Exception:  # noqa: BLE001 — never let logging fail the request path
        return url


def get_client() -> Client:
    """Get or create a shared HTTP client instance.

    Returns:
        Singleton instance of the HTTP client

    """
    # Double-checked locking: the fast path is a single read (no lock
    # taken once the client is initialised). Only the first concurrent
    # callers serialise through ``_client_lock`` to ensure exactly one
    # ``Client`` (and therefore one ``TokenBucketRateLimiter``) ever
    # exists per process.
    global client
    if client is None:
        with _client_lock:
            if client is None:
                client = Client()
    return client
