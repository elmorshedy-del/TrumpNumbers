---
name: protected-records
description: Rules for touching data/record/ and data/live_forecasts.jsonl in this repository — the append-only files that cannot be rebuilt. Use whenever a task involves deleting, cleaning, pruning, compacting, regenerating, truncating, correcting, migrating or reformatting anything under data/, whenever the repository size or "old data" comes up, and before any force-push, reset, rebase or squash that touches recording commits.
---

# Protected records

## The rule

`data/record/**` and `data/live_forecasts.jsonl` may only ever **grow**.

No deletion. No editing an existing line. No truncation. No reordering. No
regeneration. No reformatting — not even whitespace, not even to fix invalid
JSON on a line.

CI enforces this on every push and pull request via the `records` job in
`.github/workflows/tests.yml`, which runs
`scripts/check_records_append_only.py`. It compares each protected file against
its state at the base commit and fails unless the old bytes are still an exact
prefix of the new ones.

## Check before you act

Run this before any change that touches `data/`:

```bash
python scripts/check_records_append_only.py --base $(git merge-base HEAD origin/main)
```

Exit 0 means nothing was lost. Exit 1 names the file and the rule it broke.

**If this fails, the change is wrong — not the check.** Do not edit, weaken,
skip or delete the checker or its CI job to make a change pass. If you believe
the check itself has a bug, say so to the user and stop.

## Why these files are different

Everything else regenerates. `data/exports/*.json` is rewritten every pipeline
pass. `data/posts.db` lives in an Actions cache and is re-backfilled from the
mirror after any eviction. Losing either costs minutes.

The records hold what existed once:

| File | Why it cannot be rebuilt |
|---|---|
| `*_book_*.jsonl(.gz)` | Order-book depth exists at one moment and never again. |
| `calendar.jsonl` | Factba.se revises entries after events; what it said *in advance* survives only in a copy taken in advance. |
| `posts_ledger.jsonl` | First-seen times and deletion history. A rebuild from the mirror loses both — and the ReTruth re-dating and deleted-post exclusion rest on them. |
| `forecast_history.jsonl` | The forecast that was on the page while prices moved, matched to the market snapshots. |
| `live_forecasts.jsonl` | A prospective register. Its whole claim is that nothing was amended after the answer arrived. |
| `kalshi_*`, `polymarket_*` trades/candles/prices | Kept on the venue's retention promise, not ours. |

## Requests that sound reasonable and are not

- "Clean up the data directory" / "prune stale files"
- "The repo is too big, drop old data"
- "Regenerate the records from the database"
- "Fix the malformed line in `live_forecasts.jsonl`"
- "Squash the recording commits" / "reset the branch" / "force-push"
- "Reformat the JSONL for consistency"

For each: **stop and ask the user.** Never decide on their behalf that a record
is expendable. If something in a record is genuinely wrong, the correct move is
to **append a correcting entry** — the file is a log, and a log's history of
being wrong is part of what it records.

## The one legal deletion

`record.py`'s `rotate()` gzips a period that has closed: `x.jsonl` becomes
`x.jsonl.gz` and the original is unlinked. The content survives, so the checker
allows it — it decompresses the `.gz` and verifies the old bytes are still
there. This is the *only* way a protected file may disappear, and only
`record.py` should do it.

## If a record is genuinely missing

Check whether it was ever committed before assuming it was deleted:

```bash
git log --oneline --all -- data/record/<file>
```

A file that the pipeline writes but no workflow stages is being discarded on
every run, not deleted by anyone — the fix is to add it to the relevant
workflow's `git add`, not to hunt for a culprit. Both `update.yml` and
`record.yml` commit `data/record`; they write disjoint files in it, so their
appends rebase cleanly.
