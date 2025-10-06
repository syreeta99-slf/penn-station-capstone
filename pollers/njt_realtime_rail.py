#!/usr/bin/env python3
"""
NJT realtime rail poller
- Treats "Daily token limit reached" as a non-fatal outcome (exit 0)
- Uses consistent logging
- Writes/reads token at ~/.njt/token.json
"""

from __future__ import annotations
import os
import sys
import json
import time
import enum
import logging
import pathlib
import datetime as dt
from typing import Tuple, Optional

import requests

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [njt] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S+0000]"
)
log = logging.getLogger("njt")

# -----------------------------
# Config
# -----------------------------
ENV = os.getenv("NJT_ENV", "prod")
BASE = os.getenv("NJT_BASE", "https://raildata.njtransit.com")
CLIENT_ID = os.getenv("NJT_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("NJT_CLIENT_SECRET", "")
TOKEN_DIR = pathlib.Path(os.getenv("NJT_TOKEN_DIR", str(pathlib.Path.home() / ".njt")))
TOKEN_PATH = TOKEN_DIR / "token.json"
TIMEOUT = 30

# Optionally a small "warm" endpoint
WARM_ENDPOINT = "/RailDataWS/Authentication/ValidateToken"

# -----------------------------
# Token status classification
# -----------------------------
class TokenStatus(enum.Enum):
    OK = enum.auto()            # Token available/valid
    DAILY_LIMIT = enum.auto()   # Daily token issuance limit reached (non-fatal)
    UNAVAILABLE = enum.auto()   # Network/other error (fatal)

# -----------------------------
# Helpers
# -----------------------------
def _utc_midnight_next() -> dt.datetime:
    now = dt.datetime.utcnow()
    nxt = (now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1))
    return nxt

def _save_token(token: str) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps({"token": token, "saved_utc": dt.datetime.utcnow().isoformat()}) + "\n")

def _read_token() -> Optional[str]:
    if not TOKEN_PATH.exists():
        return None
    try:
        data = json.loads(TOKEN_PATH.read_text().strip() or "{}")
        return data.get("token")
    except Exception:
        return None

def _log_env():
    log.info(f"ENV={ENV}  BASE={BASE}")
    log.info(f"token path={TOKEN_PATH}")

# -----------------------------
# Token acquisition
# -----------------------------
def _request_new_token() -> Tuple[TokenStatus, Optional[str]]:
    """
    Hit NJT auth; map 'daily limit' to DAILY_LIMIT without raising.
    Expected responses:
      - 200 with token in JSON
      - 429/400-ish with a body containing 'Daily token limit reached'
    """
    url = f"{BASE}/RailDataWS/Authentication/Authorize"
    payload = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        txt_lower = (r.text or "").lower()

        if r.status_code == 200:
            try:
                token = r.json().get("access_token") or r.json().get("token")
            except Exception:
                # Some implementations return plain text token
                token = r.text.strip()
            if token:
                return TokenStatus.OK, token

        # Known daily-limit signal (status or body text)
        if "daily token limit reached" in txt_lower or r.status_code in (429,):
            log.info("Daily token limit reached; no new token until next UTC day.")
            return TokenStatus.DAILY_LIMIT, None

        # Any other non-200 response
        log.error(f"Token endpoint unexpected status={r.status_code} body={r.text[:300]}")
        return TokenStatus.UNAVAILABLE, None

    except requests.RequestException as e:
        log.error(f"Token request failed: {e}")
        return TokenStatus.UNAVAILABLE, None

def get_token() -> Tuple[TokenStatus, Optional[str]]:
    """
    Try existing token; if absent, try to fetch one.
    DAILY_LIMIT => not an error; return (DAILY_LIMIT, None)
    """
    existing = _read_token()
    if existing:
        return TokenStatus.OK, existing

    status, token = _request_new_token()
    if status is TokenStatus.OK and token:
        _save_token(token)
        return TokenStatus.OK, token

    return status, None

# -----------------------------
# Warm path (lightweight check)
# -----------------------------
def warm(token: Optional[str]) -> bool:
    """
    Return True to keep CI green unless there's a real error.
    If token is None due to DAILY_LIMIT, we simply skip work.
    """
    if not token:
        log.info("No token available this run; skipping warm.")
        return True

    try:
        url = f"{BASE}{WARM_ENDPOINT}"
        hdrs = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=hdrs, timeout=TIMEOUT)
        if r.status_code == 200:
            log.info("Warm check ok.")
            return True
        # If token is invalid/expired, remove local file so next run can re-issue post-midnight
        if r.status_code in (401, 403):
            log.warning(f"Warm check unauthorized ({r.status_code}); clearing cached token.")
            try:
                TOKEN_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            return True  # Non-fatal; next run will attempt re-issue
        log.warning(f"Warm check unexpected status={r.status_code}; proceeding.")
        return True
    except requests.RequestException as e:
        log.error(f"Warm check failed: {e}")
        # Network/infra error—return False only if you want CI to fail on infra issues.
        return False

# -----------------------------
# Main poller work (placeholder)
# -----------------------------
def run_poller(token: Optional[str]) -> bool:
    """
    If token is None because of DAILY_LIMIT, we should skip and return True (non-fatal).
    Replace with your real polling logic.
    """
    if token is None:
        log.info("No token available this run; skipping poll.")
        return True

    # ---- Example polling call(s) ----
    try:
        hdrs = {"Authorization": f"Bearer {token}"}
        # Example endpoint (replace with actual)
        url = f"{BASE}/RailDataWS/Trains/GetAll"
        r = requests.get(url, headers=hdrs, timeout=TIMEOUT)
        if r.status_code == 200:
            # process r.json() ...
            log.info("Poll succeeded.")
            return True
        if r.status_code in (401, 403):
            log.warning(f"Poll unauthorized ({r.status_code}); clearing cached token.")
            try:
                TOKEN_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            return True  # Non-fatal; we'll re-issue next UTC day
        log.error(f"Poll unexpected status={r.status_code}: {r.text[:300]}")
        return False
    except requests.RequestException as e:
        log.error(f"Poll failed: {e}")
        return False

# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    _log_env()

    status, token = get_token()

    if status is TokenStatus.OK:
        ok = warm(token)
        if not ok:
            sys.exit(1)
        ok = run_poller(token)
        sys.exit(0 if ok else 1)

    if status is TokenStatus.DAILY_LIMIT:
        # Uniform, friendly message; keep step green
        log.info("No token available this run; skipping.")
        reset = _utc_midnight_next().strftime("%Y-%m-%d %H:%M:%S UTC")
        log.info(f"Next token window: {reset}")
        sys.exit(0)

    # UNAVAILABLE or true error
    log.error("Token unavailable due to error; failing this run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
