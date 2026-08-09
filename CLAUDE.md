# Working in this repository

Read this before changing anything under `data/`.

## The one rule that matters

**`data/record/` and `data/live_forecasts.jsonl` are append-only. Never delete,
edit, truncate, reorder, regenerate, compact or rewrite them.**

This is enforced by CI (`scripts/check_records_append_only.py`, run by the
`records` job in `.github/workflows/tests.yml`), which compares every protected
file against its previous state and fails on anything that is not a pure
append. If that job goes red, the change is destroying data — fix the change,
never the check.

### Why, in one paragraph

Everything else in this project is derived and regenerates itself. Delete
`data/exports/`, drop the cached `data/posts.db`, and the next pipeline pass
rebuilds both from the source in minutes. The records are the exception: they
hold things that existed exactly once and no longer exist anywhere else.
Order-book depth at a given minute is gone the minute after. Factba.se revises
the public schedule after events happen, so what it said *in advance* survives
only in the copy taken in advance. `posts_ledger.jsonl` holds first-seen times
and deletion history, which a rebuild from the mirror cannot return — and the
ReTruth re-dating and the deleted-post exclusion both rest on that evidence.
`live_forecasts.jsonl` is a prospective register whose entire claim is that
nothing was amended once the answer was known; a single edited line makes every
row in it unfalsifiable.

### What this rules out

Requests that sound reasonable and are not:

- "Clean up / prune / compact the data directory" — no.
- "Regenerate the records from the database" — impossible; the database is the
  thing that loses this information.
- "Fix the bad line in `live_forecasts.jsonl`" — append a correcting entry.
  Never edit the original.
- "Squash or rebase away the recording commits" — no. Their content is the point.
- "The repository is too large, remove old data" — the archives are already
  gzipped on rotation. Raise it with the user rather than deleting anything.
- "Reset the branch / force-push to drop these commits" — no.

If a task seems to require any of the above, stop and ask the user. Do not
decide on their behalf that a record is expendable.

### The one legal deletion

`record.py` rotates a closed period: `x.jsonl` becomes `x.jsonl.gz` and the
original is unlinked. That is allowed because the content survives. CI verifies
the gzip still carries the old bytes. Nothing else may remove a protected file.

## How the pipeline is laid out

Two workflows write data, on separate schedules:

- `.github/workflows/update.yml` — every 30 min: poll, rebuild the forecast,
  deploy the site. Nightly at 08:10 UTC it also reconciles and re-runs the full
  backtest. Commits `data/live_forecasts.jsonl` and `data/record`, plus
  `data/exports` on the nightly pass.
- `.github/workflows/record.yml` — hourly order-book snapshots, daily deep pull.
  Commits `data/record`.

`daemon.py` runs the same functions on the same cadences locally. It is not a
second implementation — keep the two in step when changing either.

### Ordering inside a pass

`one_pass()` in `daemon.py` runs the derived exports through `best_effort()` and
then calls `refresh_forecast()` unguarded, deliberately:

- The feed, diagnostics and backtest are rebuilt next pass. Failing one costs
  half an hour of staleness, so it must not abort the pass.
- `refresh_forecast()` is the only step that writes anything irreplaceable. The
  register keeps the **first** forecast per (week, cut), so a cut that no pass
  ever reaches is a permanent hole. If it fails, the job **should** go red: that
  leaves the last good deploy up and the reason in the run log.

Do not "tidy" this by wrapping everything uniformly in either direction.

## Exports must parse in a browser

Every page loads its data through one `fetch().json()`. Python writes `NaN` for
a missing float by default and reads it back happily; `JSON.parse` rejects it,
which blanks the whole page with no console error and no failed request. So
`_write()` in `truthforecast/pipeline.py` passes `allow_nan=False`, and
`_json_safe()` maps every flavour of missing (`float('nan')`, `np.nan`, `pd.NA`,
`NaT`) onto `null` first. Keep both. A failed export is visible; an export that
will not parse is not.

## Tests

`python -m pytest tests/ -q`. `tests/test_regressions.py` carries one test per
defect ever found here, each written to pin the behaviour rather than the
implementation. Add to it when you fix something; do not delete from it.
