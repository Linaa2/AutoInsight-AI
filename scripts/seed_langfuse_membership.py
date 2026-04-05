#!/usr/bin/env python3
"""scripts/seed_langfuse_membership.py — Post-startup project membership seed.

WHY THIS EXISTS
---------------
LangFuse v2's ``LANGFUSE_INIT_*`` bootstrap variables create the
organisation, project, user, and *organisation* membership — but they do
**not** insert a row into the ``project_memberships`` table.  Without that
row every project-backed UI page (dashboard, users, datasets …) returns:

    TRPCError: User is not a member of this project

This script fills that gap.  It runs after ``docker compose up -d`` in both
``make langfuse-up`` and ``make langfuse-reset``, so the fix is durable
across fresh volumes.

WHAT IT DOES
------------
1. Waits (up to ``MAX_WAIT`` seconds) for the LangFuse DB to be reachable
   *and* for LANGFUSE_INIT_* to have created at least one user.
2. Runs a single idempotent SQL ``INSERT … ON CONFLICT DO NOTHING`` that
   links every user to every project inside their organisation wherever the
   ``project_memberships`` row is absent.
3. Exits 0 on success.  On a timeout or DB error it prints a warning and
   still exits 0 — a missing membership is bad UX but not a hard startup
   failure.

IDEMPOTENCY
-----------
Safe to run multiple times.  The query uses ``ON CONFLICT (project_id,
user_id) DO NOTHING`` so re-running after a successful seed is a no-op.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_LANGFUSE = REPO_ROOT / ".env.langfuse"
COMPOSE_FILE = REPO_ROOT / "docker-compose.langfuse.yml"

# How long to wait for the DB + INIT_* run to complete before giving up.
MAX_WAIT_SECONDS = 90

# ── SQL ─────────────────────────────────────────────────────────────────────

# Insert missing project_memberships for ALL org members that don't already
# have a row for a project inside their org.  Uses ON CONFLICT so it is safe
# to run repeatedly.
SEED_SQL = """\
INSERT INTO project_memberships (project_id, user_id, org_membership_id, role)
SELECT
    proj.id               AS project_id,
    u.id                  AS user_id,
    om.id                 AS org_membership_id,
    'OWNER'::"Role"       AS role
FROM users u
JOIN organization_memberships om ON om.user_id = u.id
JOIN projects proj ON proj.org_id = om.org_id
ON CONFLICT (project_id, user_id) DO NOTHING;
"""

# ── Helpers ─────────────────────────────────────────────────────────────────


def _psql(sql: str) -> tuple[int, str, str]:
    """Run ``sql`` via ``docker compose exec langfuse-db psql`` and return (rc, stdout, stderr).

    Using ``docker compose exec`` (rather than a hardcoded container name)
    makes this robust to different project / directory names.
    """
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "exec",
            "-T",  # no pseudo-TTY (safe for scripted use)
            "langfuse-db",
            "psql",
            "-U",
            "langfuse",
            "-d",
            "langfuse",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def _wait_for_users(max_wait: int = MAX_WAIT_SECONDS) -> bool:
    """Wait until at least one user row exists in the DB (INIT_* finished)."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        rc, stdout, _ = _psql("SELECT COUNT(*) FROM users;")
        if rc == 0:
            try:
                # psql output looks like:  ⎡ count ⎤ / ⎢  1   ⎥
                count = int(
                    next(
                        ln.strip()
                        for ln in stdout.splitlines()
                        if ln.strip() and ln.strip().lstrip("-").isdigit()
                    )
                )
            except (IndexError, ValueError):
                count = 0
            if count > 0:
                return True
        time.sleep(3)
    return False


# ── ANSI helpers ─────────────────────────────────────────────────────────────


def _ok(msg: str) -> None:
    print(f"  \033[32m✔\033[0m  {msg}")


def _info(msg: str) -> None:
    print(f"  \033[36m·\033[0m  {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33m!\033[0m  {msg}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("\033[1;36m🔗  LangFuse — seeding project memberships\033[0m")
    print("\033[90m" + "─" * 60 + "\033[0m")

    _info(f"Waiting for DB + LANGFUSE_INIT_* (up to {MAX_WAIT_SECONDS}s) …")
    if not _wait_for_users():
        _warn(
            f"Timed out after {MAX_WAIT_SECONDS}s — no users found in DB. "
            "Project membership seed skipped. "
            "Run `make langfuse-up` again once LangFuse has started."
        )
        sys.exit(0)  # non-fatal — don't block the make target

    rc, _stdout, stderr = _psql(SEED_SQL)
    if rc != 0:
        _warn(f"project_memberships seed query failed: {stderr.strip()}")
        sys.exit(0)  # non-fatal

    _ok("project_memberships seed complete — users can now access their projects.")
    print()


if __name__ == "__main__":
    main()
