r"""Parsing helpers for Google Flights' FlightsFrontendService wire format.

The Service returns JSONP-flavoured responses of the form::

    )]}'\n\n
    <chunk1_byte_len>\n
    [["wrb.fr", null, "<inner JSON string>"]]
    <chunk2_byte_len>\n
    [["wrb.fr", null, "<inner JSON string>"]]
    ...

`GetShoppingResults` and `GetCalendarGraph` happen to emit a single chunk so
the legacy parsers in this package could get away with `lstrip(")]}'")`.
`GetBookingResults` emits two chunks, so we need a proper multi-chunk reader.

Important quirk: the length headers count UTF-8 **bytes**, not Python string
characters. When the response contains any non-ASCII characters (which it
sometimes does — airport names, airline names) the offsets diverge, so the
reader must operate over the byte representation of the body.

This module centralises that reader and exposes :func:`iter_wrb_chunks` which
yields the decoded inner JSON of each ``wrb.fr`` chunk.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from fli.search.exceptions import GoogleFlightsUpstreamError

logger = logging.getLogger(__name__)

_PREFIX = b")]}'"


def iter_wrb_chunks(body: str | bytes) -> Iterator[Any]:
    """Yield the inner JSON object of every ``wrb.fr`` chunk in ``body``.

    Robust to single-chunk responses with no length headers (the older
    ``GetShoppingResults`` / ``GetCalendarGraph`` shape) — those are parsed
    by falling back to a single JSON load over the trimmed body.
    """
    if isinstance(body, str):
        raw = body.encode("utf-8")
    else:
        raw = body

    raw = raw.lstrip()
    if raw.startswith(_PREFIX):
        raw = raw[len(_PREFIX) :]
    raw = raw.lstrip()

    if not raw:
        return

    # Fast path: no length headers (legacy single-chunk responses).
    if not (b"0" <= raw[:1] <= b"9"):
        try:
            outer = json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Failed to decode single-chunk wrb.fr body as JSON", exc_info=True)
            return
        yield from _chunks_from_outer(outer)
        return

    cursor = 0
    while cursor < len(raw):
        # Read the decimal length prefix terminated by \n.
        end = raw.find(b"\n", cursor)
        if end == -1:
            break
        try:
            length = int(raw[cursor:end])
        except ValueError:
            logger.warning(
                "Malformed length header at offset %d; truncating chunk stream",
                cursor,
            )
            break
        # Google's length header counts the leading newline after the header
        # AND the trailing newline that separates this chunk from the next.
        # We've already consumed the leading newline (it terminated the header),
        # so we read `length - 1` bytes which gives JSON + trailing \n.
        cursor = end + 1
        chunk_bytes = max(length - 1, 0)
        payload = raw[cursor : cursor + chunk_bytes]
        cursor += chunk_bytes
        try:
            outer = json.loads(payload.strip().decode("utf-8"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Discarding malformed wrb.fr chunk", exc_info=True)
            continue
        yield from _chunks_from_outer(outer)


def _grpc_error_in_frame(frame: Any) -> tuple[int, str | None] | None:
    """Detect Google's structured ErrorResponse envelope in a ``wrb.fr`` frame.

    A success frame is ``["wrb.fr", null, "<json>"]`` with the payload at index
    2. When Google rejects the request it still answers HTTP 200 but replaces
    that payload with an error block, e.g.::

        ["wrb.fr", null, null, null, null,
         [13, null, [["type.googleapis.com/travel.frontend.flights.ErrorResponse", ...]]]]

    The leading int (``13`` here) is a gRPC status code (13 = INTERNAL). Returns
    ``(grpc_code, type_url)`` when such a block is present, else ``None``.
    ``bool`` (an ``int`` subclass) and code ``0`` (OK) are deliberately not
    treated as errors so neither trips a false positive.
    """
    if not isinstance(frame, list):
        return None
    for element in frame:
        if not (isinstance(element, list) and len(element) >= 3):
            continue
        code = element[0]
        if type(code) is not int or code == 0:
            continue
        try:
            type_url = element[2][0][0]
        except (IndexError, TypeError):
            type_url = None
        if isinstance(type_url, str) and type_url.endswith(".ErrorResponse"):
            return code, type_url
    return None


def _chunks_from_outer(outer: Any) -> Iterator[Any]:
    """Walk a top-level chunk list and yield decoded inner-JSON payloads.

    Raises :class:`~fli.search.exceptions.GoogleFlightsUpstreamError` when a
    frame carries Google's structured ErrorResponse envelope (an HTTP-200
    upstream rejection). Without this, the error frame's payload slot is null,
    so the reader would skip it and the caller would see a silent empty result
    indistinguishable from "no flights".
    """
    if not isinstance(outer, list):
        return
    for row in outer:
        if not isinstance(row, list):
            continue
        error = _grpc_error_in_frame(row)
        if error is not None:
            grpc_code, type_url = error
            logger.warning(
                "Google Flights returned an ErrorResponse (gRPC %s, %s)",
                grpc_code,
                type_url,
            )
            raise GoogleFlightsUpstreamError(grpc_code, type_url=type_url)
        if len(row) < 3 or row[0] != "wrb.fr":
            continue
        inner = row[2]
        if not isinstance(inner, str) or not inner:
            continue
        try:
            yield json.loads(inner)
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to decode wrb.fr inner JSON payload", exc_info=True)
            continue


def parse_first_wrb_payload(body: str | bytes) -> Any:
    """Return the inner JSON of the first ``wrb.fr`` chunk, or None."""
    for chunk in iter_wrb_chunks(body):
        return chunk
    return None
