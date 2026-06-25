"""Typed errors raised by the search client.

These exist so the CLI (and library consumers) can react to network
failures with a clear, user-facing message instead of a raw curl-cffi
traceback. They are intentionally light wrappers — the original
exception is kept as ``__cause__`` for logging.
"""

from __future__ import annotations


class SearchClientError(Exception):
    """Base class for errors talking to the Google Flights backend."""


class SearchTimeoutError(SearchClientError):
    """The request to Google Flights timed out before any data arrived."""


class SearchConnectionError(SearchClientError):
    """A network/DNS issue prevented us from reaching Google Flights."""


class SearchHTTPError(SearchClientError):
    """Google Flights returned a non-2xx HTTP response."""

    def __init__(self, message: str, *, status_code: int | None = None):
        """Store the HTTP status alongside the message for richer logging."""
        super().__init__(message)
        self.status_code = status_code


# Canonical gRPC status codes (0-16) carried by Google's ErrorResponse envelope.
# Used only to label the exception message; the numeric code is authoritative.
_GRPC_CODE_NAMES = {
    0: "OK",
    1: "CANCELLED",
    2: "UNKNOWN",
    3: "INVALID_ARGUMENT",
    4: "DEADLINE_EXCEEDED",
    5: "NOT_FOUND",
    6: "ALREADY_EXISTS",
    7: "PERMISSION_DENIED",
    8: "RESOURCE_EXHAUSTED",
    9: "FAILED_PRECONDITION",
    10: "ABORTED",
    11: "OUT_OF_RANGE",
    12: "UNIMPLEMENTED",
    13: "INTERNAL",
    14: "UNAVAILABLE",
    15: "DATA_LOSS",
    16: "UNAUTHENTICATED",
}


class GoogleFlightsUpstreamError(SearchClientError):
    """Google answered HTTP 200 with a structured ErrorResponse envelope.

    Instead of flight data, the response carries a gRPC status code (e.g.
    13 = INTERNAL). This is an upstream rejection, not a parse failure and not
    a genuine "no results" (which still decodes to ``None``/empty). The numeric
    code is surfaced via ``grpc_code`` so a caller can apply its own retry or
    alerting policy rather than fli hard-coding one: a default retry during a
    broad outage would turn a fleet of clients into a retry storm and deepen
    the upstream's gating of everyone.

    Attributes:
        grpc_code: The gRPC status code from the envelope (e.g. 13 = INTERNAL),
            or ``None`` if it could not be extracted.
        type_url: The protobuf type URL Google attached, when present.

    """

    def __init__(self, grpc_code: int | None, type_url: str | None = None):
        """Build the error from the gRPC status code and optional type URL."""
        self.grpc_code = grpc_code
        self.type_url = type_url
        name = _GRPC_CODE_NAMES.get(grpc_code) if grpc_code is not None else None
        if grpc_code is None:
            label = "unknown"
        else:
            label = f"{grpc_code} {name}" if name else str(grpc_code)
        super().__init__(
            f"Google Flights rejected the request with an ErrorResponse "
            f"envelope (gRPC status {label}) instead of flight data. This is "
            f"an upstream error, not an empty result."
        )
