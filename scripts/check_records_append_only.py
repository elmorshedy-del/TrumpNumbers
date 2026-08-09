#!/usr/bin/env python3
"""Fail the build if a commit deleted or rewrote part of the record.

The records under `data/record/` and `data/live_forecasts.jsonl` are the only
data in this project that cannot be rebuilt. Order-book depth exists once and
never again. The public schedule is revised after the fact, so what it said in
advance is only knowable from a copy taken in advance. `posts_ledger.jsonl`
holds first-seen times and deletion history, which a rebuild from the mirror
cannot return. `live_forecasts.jsonl` is a prospective register whose whole
claim rests on nothing being amended after the outcome is known.

Everything else here is derived. Delete `data/exports/`, drop the cached
database, and the next pass rebuilds both from the source. Delete a line from
the ledger and it is gone.

That asymmetry is invisible in a diff. A tidy-up that prunes "stale" data files,
a rebase that drops a commit, an agent asked to shrink the repository — each is
a normal-looking change that costs something unrecoverable, and none of them
announce it. So the invariant is checked mechanically rather than remembered:

    every protected file that existed at the base commit must still exist at
    the head commit, with its previous bytes as an exact prefix.

Appending is allowed. Rewriting history is not. Deleting is not. The one
exception is the rotation `record.py` performs on closed periods: `x.jsonl`
becomes `x.jsonl.gz`, which is a deletion whose content survives, so it is
allowed exactly when the gzip decompresses to something that still carries the
old bytes as a prefix.

    python scripts/check_records_append_only.py --base <sha> [--head <sha>]

Exits non-zero, and names the file and the rule it broke, on any violation.
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys

# Paths whose contents may only ever grow. Anything not listed here is derived
# and may be rewritten freely — that is the whole distinction this file exists
# to enforce, so adding a path here should mean "this cannot be regenerated",
# not "this looks important".
PROTECTED_PREFIXES = ("data/record/", "data/live_forecasts.jsonl")

NULL_SHA = "0" * 40


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def _tracked(commit: str) -> set[str]:
    out = _git("ls-tree", "-r", "--name-only", commit).decode()
    return {p for p in out.splitlines() if p.startswith(PROTECTED_PREFIXES)}


def _blob(commit: str, path: str) -> bytes | None:
    try:
        return _git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None


def _logical(path: str, raw: bytes | None) -> bytes | None:
    """Decompressed content, so a rotation compares like with like."""
    if raw is None:
        return None
    if not path.endswith(".gz"):
        return raw
    try:
        return gzip.decompress(raw)
    except OSError:
        return raw


def violations(base: str, head: str) -> list[str]:
    found = []
    for path in sorted(_tracked(base)):
        before = _logical(path, _blob(base, path))
        if before is None:
            continue

        after = _logical(path, _blob(head, path))
        if after is None and not path.endswith(".gz"):
            # The one legal way for a protected file to disappear: record.py
            # gzipped a period that has closed and can no longer grow.
            after = _logical(f"{path}.gz", _blob(head, f"{path}.gz"))
            if after is None:
                found.append(
                    f"{path}: deleted. Records are append-only — if this file has "
                    f"closed, rotate it to {path}.gz rather than removing it."
                )
                continue

        if after is None:
            found.append(f"{path}: deleted. Rotated archives are never removed.")
            continue

        if not after.startswith(before):
            kept = len(before)
            found.append(
                f"{path}: rewritten. The first {kept} bytes must still be the first "
                f"{kept} bytes — lines may be appended, never edited, reordered or "
                f"dropped. Add a correcting entry instead of changing an old one."
            )
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="commit the change starts from")
    p.add_argument("--head", default="HEAD", help="commit the change ends at")
    args = p.parse_args(argv)

    # A branch's first push reports an all-zero "before"; there is no prior
    # state to protect, so there is nothing to check.
    if not args.base or args.base == NULL_SHA:
        print("no base commit to compare against — skipping")
        return 0

    try:
        _git("cat-file", "-e", f"{args.base}^{{commit}}")
    except subprocess.CalledProcessError:
        print(f"base commit {args.base} not present in this checkout — skipping")
        return 0

    found = violations(args.base, args.head)
    if not found:
        print(f"records intact: nothing under {', '.join(PROTECTED_PREFIXES)} was lost")
        return 0

    print("\nThis change destroys data that cannot be rebuilt:\n", file=sys.stderr)
    for v in found:
        print(f"  - {v}", file=sys.stderr)
    print(
        "\nWhy this is blocked: these files are the only record of things that "
        "existed once — order-book depth at a moment, what the schedule said "
        "before the day happened, when a post was first seen, what was forecast "
        "before the week closed. Nothing regenerates them.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
