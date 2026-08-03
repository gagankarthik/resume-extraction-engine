"""
Upload authorisation for the extraction engine.

WHY THIS EXISTS

The engine runs behind a Lambda Function URL with `authorization_type = "NONE"`,
because the browser posts resumes to it directly — the multi-agent pipeline
takes 30-90 seconds, which does not fit inside the 30-second timeout that any
CloudFront/API Gateway proxy in front of it would impose. Function URLs allow
the long request; they also mean AWS performs no authentication of its own.

Without the check below, the endpoint is open to the internet: anyone who
learns the URL can run a ten-agent GPT pipeline on the account's API key.

WHAT THE CHECK IS

The frontend verifies the caller's Cognito session and then mints a short-lived
HS256 JWT with a secret shared only between the two services. The browser sends
that token with the upload. It is deliberately not a Cognito token: this service
has no reason to know about the user pool, and a ten-minute upload ticket is a
much smaller thing to leak than a session.

Verification is stdlib `hmac` rather than a JWT library, because the Lambda zip
is built from requirements.txt and this is a well-specified 30-line format —
not worth a dependency in a package that already ships pdfplumber and openai.

FAILING CLOSED

If EXTRACTION_SHARED_SECRET is unset the service refuses uploads. It does not
fall back to allowing them: an unauthenticated deployment is the exact failure
this module exists to prevent, and silent degradation is how it went unnoticed
before. For local development set EXTRACTION_ALLOW_ANONYMOUS=true, which is
explicit, greppable, and says what it does.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# Named so it cannot collide with anything AWS reserves on a Function URL.
TOKEN_HEADER = "X-Extraction-Token"

# The token must say it was minted for this service, so a token issued for
# some other service sharing the secret cannot be replayed here.
AUDIENCE = "resume-extraction-engine"

# Upper bound on how long a ticket may live, enforced here rather than trusted
# from the issuer. A minted token claiming a one-year expiry is rejected.
MAX_LIFETIME_SECONDS = 15 * 60

# Tolerance for clock drift between the frontend's compute and this Lambda.
CLOCK_SKEW_SECONDS = 60


def _b64url_decode(segment: str) -> bytes:
    """Decode a base64url segment, restoring the padding JWT strips."""
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _reject(reason: str) -> HTTPException:
    """
    One opaque message outward, the real reason in the log.

    Telling a caller which part of their token failed is free reconnaissance.
    """
    logger.warning("[auth] upload rejected: %s", reason)
    return HTTPException(status_code=401, detail="Upload authorisation failed or expired.")


def verify_upload_token(token: str, secret: str) -> dict:
    """
    Check an HS256 JWT and return its claims. Raises HTTPException on any fault.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise _reject("not three JWT segments")

    header_seg, payload_seg, signature_seg = parts

    try:
        header = json.loads(_b64url_decode(header_seg))
        claims = json.loads(_b64url_decode(payload_seg))
        signature = _b64url_decode(signature_seg)
    except (ValueError, TypeError) as exc:
        raise _reject(f"undecodable token: {exc}") from exc

    if not isinstance(claims, dict) or not isinstance(header, dict):
        raise _reject("header or claims not an object")

    # Pin the algorithm. Accepting whatever the token names is how "alg": "none"
    # and RS256-key-confusion attacks work.
    if header.get("alg") != "HS256":
        raise _reject(f"unexpected alg {header.get('alg')!r}")

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{header_seg}.{payload_seg}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(expected, signature):
        raise _reject("signature mismatch")

    # Signature is good; the claims can now be trusted enough to read.
    if claims.get("aud") != AUDIENCE:
        raise _reject(f"wrong audience {claims.get('aud')!r}")

    now = time.time()

    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        raise _reject("missing exp")
    if now > exp + CLOCK_SKEW_SECONDS:
        raise _reject("expired")

    iat = claims.get("iat")
    if not isinstance(iat, (int, float)):
        raise _reject("missing iat")
    if iat > now + CLOCK_SKEW_SECONDS:
        raise _reject("issued in the future")
    if exp - iat > MAX_LIFETIME_SECONDS:
        raise _reject("lifetime longer than this service permits")

    return claims


async def require_upload_token(
    x_extraction_token: str | None = Header(default=None),
) -> dict | None:
    """
    FastAPI dependency guarding the upload route.

    Returns the token claims so the caller can be logged; the `sub` is the
    frontend's Cognito subject, which is what ties an extraction to a person.
    """
    secret = os.getenv("EXTRACTION_SHARED_SECRET", "").strip()

    if not secret:
        if os.getenv("EXTRACTION_ALLOW_ANONYMOUS", "").strip().lower() == "true":
            logger.warning(
                "[auth] EXTRACTION_ALLOW_ANONYMOUS=true — uploads are UNAUTHENTICATED. "
                "This must never be set in a deployed environment."
            )
            return None
        logger.error(
            "[auth] EXTRACTION_SHARED_SECRET is not set, so no upload can be authorised. "
            "Set it here and in the frontend, or set EXTRACTION_ALLOW_ANONYMOUS=true locally."
        )
        raise HTTPException(
            status_code=503,
            detail="The extraction service is not accepting uploads: authorisation is not configured.",
        )

    if not x_extraction_token:
        raise _reject(f"no {TOKEN_HEADER} header")

    return verify_upload_token(x_extraction_token.strip(), secret)
