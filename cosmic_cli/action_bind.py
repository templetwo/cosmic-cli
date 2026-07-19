"""Canonical action-binding records for PAUSE / policy hashes.

Invariant: one approval authorizes exactly one concrete mutation, once.
Binding records must not collide when field bytes contain separators
(Claude v3 EDIT collision: newline-joined path/old/new).
"""

from __future__ import annotations

import hashlib
from typing import Optional


SCHEMA = "cosmic.mutation.v1"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _field(name: str, value: str) -> str:
    """Length-delimited field: name=<byte_len>:<utf-8-bytes>."""
    raw = value.encode("utf-8")
    return f"{name}={len(raw)}:{value}"


def _field_bytes(name: str, data: bytes) -> str:
    return f"{name}_sha={_sha256_hex(data)}:{len(data)}"


def bind_write(*, path: str, content: str) -> str:
    """Canonical WRITE/CREATE binding over full content bytes."""
    body = content.encode("utf-8")
    return "\n".join(
        [
            SCHEMA,
            "action=WRITE",
            _field("path", path),
            _field_bytes("post", body),
        ]
    )


def bind_edit(
    *,
    path: str,
    pre_content: str,
    old: str,
    new: str,
    post_content: str,
) -> str:
    """Canonical EDIT binding: path + pre/post hashes + old/new lengths.

    Uses hashes+lengths for content identity so raw newlines in old/new
    cannot forge another edit's record.
    """
    pre_b = pre_content.encode("utf-8")
    post_b = post_content.encode("utf-8")
    old_b = old.encode("utf-8")
    new_b = new.encode("utf-8")
    return "\n".join(
        [
            SCHEMA,
            "action=EDIT",
            _field("path", path),
            _field_bytes("pre", pre_b),
            _field_bytes("post", post_b),
            f"old_sha={_sha256_hex(old_b)}:{len(old_b)}",
            f"new_sha={_sha256_hex(new_b)}:{len(new_b)}",
        ]
    )


def bind_mkdir(*, path: str) -> str:
    return "\n".join([SCHEMA, "action=MKDIR", _field("path", path)])


def action_sha256(binding_record: str) -> str:
    return _sha256_hex(binding_record.encode("utf-8"))
