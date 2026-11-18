"""Confirmation tokens: proof that a write was previewed, with exactly these arguments.

An MCP write tool runs in two calls. The first call (no token) changes nothing: it returns
a preview of the change and a token. The client's model shows the person the preview, and
only after they agree calls the tool again with the same arguments plus the token.

The token is signed with a secret only this server knows (an HMAC), so it can't be made
up and it only fits the arguments it was issued for: change anything (another priority, a
different ticket) and it no longer matches, so the person must see the new preview. The
server keeps no list of tokens: the signature itself is the proof, so this also works when
every HTTP request is handled independently.

The token's random part (the nonce) becomes the write's idempotency key, `mcp:<nonce>`.
Sending the confirmed call twice (a client retry, a timeout) still writes once.
"""

import hashlib
import hmac
import json
import secrets
from typing import Any


class Confirmations:
    def __init__(self, secret: bytes | None = None):
        # A new secret per server process: tokens from before a restart stop working, and
        # the client simply asks for a new preview.
        self._secret = secret or secrets.token_bytes(32)

    def issue(self, tool: str, arguments: dict[str, Any]) -> str:
        """A token for running `tool` with these (validated, normalised) arguments."""
        nonce = secrets.token_hex(8)
        return f"{nonce}.{self._sign(tool, nonce, arguments)}"

    def check(self, token: str, tool: str, arguments: dict[str, Any]) -> str | None:
        """The token's nonce if this server issued it for exactly this call, else None."""
        nonce, _, signature = token.partition(".")
        if not nonce or not signature:
            return None
        # compare_digest takes the same time however many characters match, so the
        # response time doesn't help anyone guess a valid signature one character at a time.
        if hmac.compare_digest(signature, self._sign(tool, nonce, arguments)):
            return nonce
        return None

    def _sign(self, tool: str, nonce: str, arguments: dict[str, Any]) -> str:
        # sort_keys: the same arguments always give the same text, whatever their order.
        message = json.dumps([tool, nonce, arguments], sort_keys=True, separators=(",", ":"))
        return hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()[:32]
