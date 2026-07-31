# Trump Truth Social post tracker & weekly forecast

Tracks how often Donald Trump posts on Truth Social, shows the posts, and projects where the
current Monday–Sunday total will land — with ~20 statistical models backtested against each other
using proper scoring rules.

Python pipeline + a static site. No build step, no server dependency, no API keys.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python daemon.py --once     # backfill (first run), then build every export
python daemon.py            # run forever: poll, forecast, nightly backtest, serve :8000
```

Then open <http://localhost:8000/site/index.html>.

---

## What it found

From the full archive (34,000+ posts, Feb 2022 → present; models fitted on the second term from
2025-01-20):

**It is nowhere near Poisson.** Variance runs ~16× the mean. A Poisson process at this rate would
put 95% of days between 10 and 28 posts; the actual range is 0 to 168. That failure is reported
rather than hidden, because it is the finding that motivates every other model: either the rate
isn't fixed, or the posts aren't independent.

**It is heavy-tailed.** The busiest day is 11× the median, skewness 3.6, and the busiest 1% of days
carry 6% of all posts. So every figure is a median with an *asymmetric* interval, never `x ± y`.

**There is almost no day-to-day memory.** Once the weekly cycle is removed, autocorrelation is
inside the noise band at every lag (lag-1 ≈ 0.11, lag-7 falls from 0.027 to 0.007). Whatever makes
posts cluster works *within* a day, not across days.

**Self-excitation is real but very fast.** The Hawkes fit gives a branching ratio around 0.56 —
each post triggers ~0.56 further posts — with a kernel half-life of about a minute. That produces
the rapid-fire bursts you can see in the feed, but it barely moves the *weekly* total, which is
consistent with the flat autocorrelation.

**Weeks are genuinely hard to forecast.** The best models beat climatology by only a few percent of
CRPS. That is the honest headline. Anyone quoting a confident weekly number is quoting noise.

**Most models are overconfident.** Across 53 weeks, nearly every model's 80% interval contained the
truth only 70–77% of the time. Poisson and ZIP are the worst (39%, with intervals a third of the
width they need) — the direct consequence of variance being locked to the mean.

## How it works

```
truthforecast/
  ingest/     cursor-walk client for trumpstruth.org, parsers, SQLite store
  series.py   posts -> daily/weekly counts (US Eastern days)
  diagnostics.py  dispersion, tails, deseasonalized ACF/PACF, regimes, bursts
  models/     the model zoo (base contract, baselines, counts, dynamics, ml, registry)
  backtest/   walk-forward harness, proper scoring rules, leaderboard
  forecast.py current-week projection
  pipeline.py orchestration -> data/exports/*.json
site/         static HTML/CSS/vanilla JS reading those JSON files
daemon.py     the self-updating local runner
```

### Data source

Truth Social's own API returns 403 to datacenter IPs, so posts are read from
[trumpstruth.org](https://trumpstruth.org), a complete public mirror whose `robots.txt` permits
crawling. Its listing pages paginate with a `cursor` parameter that is plain base64 JSON
(`{"status_created_at": "...", "_pointsToNextItems": true}`), so a cursor can be *synthesized* for
any date — the whole archive backfills in ~360 requests at 1 req/sec.

Two gotchas the code handles, both of which silently corrupt the series if missed:

- **Nested posts.** A ReTruth embeds the original as a nested `.status` block; a typical page has
  159 `.status` nodes for 100 real posts. Only top-level blocks (those with `data-status-url`) count.
- **ReTruth timestamps are the *original's*.** A ReTruth of a month-old post is stored under the
  month-old date, by both the feed and the listing. So ReTruths are ingested and labelled but
  **excluded from the modelled series** — counting them would credit activity to days with none.

Timestamps are stored in UTC and bucketed into **US Eastern** days. The listing renders Eastern
wall-clock with no zone label; `tests/test_ingest.py` pins the offset in both EDT and EST, so a site
change can't quietly shift the whole archive by an hour.

### The model zoo

| Tier | Models |
|---|---|
| Baselines | last-week, trailing-4w median, day-of-week climatology (×2) |
| Resampling | block bootstrap over whole week-remainders, plus an independent-day control |
| Count GLM | Poisson, Negative Binomial (±trend), ZIP, ZINB |
| Shrinkage | empirical-Bayes / James–Stein day-of-week |
| Dynamics | INGARCH, log-SARIMA |
| Regime | Poisson-HMM, NegBin-HMM |
| Self-exciting | Hawkes (exponential kernel, fitted on post-level times) |
| ML | LightGBM quantile regression |
| Ensembles | linear opinion pool, quantile averaging |

Every model returns a **full predictive distribution**, which is what makes them comparable.
The list is pre-registered in `models/registry.py` and the leaderboard shows all of it, including
the models that lost badly — a board showing only winners is a target painted after the shot.

### Grading

Walk-forward, expanding origin, one-day embargo, with a final stretch of weeks held out. For every
historical week and every weekday cut, each model forecasts that week's total from what was
knowable then.

Scored only with **proper** rules — CRPS (primary, in posts), log score, pinball, Brier for
thresholds — plus PIT-based calibration and interval width for sharpness. Accuracy is deliberately
absent: a model can be more accurate on average while being systematically overconfident, and
overconfidence is the failure that stays invisible until the tail event arrives.

Two caveats the numbers themselves demand:
- Each week is scored at seven cuts, so forecasts aren't independent. PIT p-values are a strong
  directional signal about over/under-confidence, not exact significance.
- Gaps of a few hundredths of CRPS between neighbouring models are inside the noise of ~50 weeks.
  Read the groupings, not the ordering.

## What it can't tell you

It cannot tell you **why** he posted a lot on a given day. Clustering has two explanations that look
nearly identical in daily counts — an outside shock hitting everything at once, versus events
triggering each other — and separating them is a live research problem, not a textbook exercise.
Both are fitted; neither is asserted. The burst keywords describe what was in a heavy day; they do
not explain it.

Tail probabilities are the softest numbers here and are flagged as such in the UI.

## Commands

```bash
python daemon.py --once            # one full pass (backfills on first run)
python daemon.py --backtest        # force the expensive backtest now
python daemon.py --serve-only      # just serve existing exports
python -m truthforecast.pipeline --backfill 2022-02-01
python -m truthforecast.pipeline --poll --forecast
python -m pytest tests/ -q
```

Config lives in `truthforecast/config.py`. The one setting worth thinking about is the modelling
window (`TTF_WINDOW_START`): which slice of history you fit is a bigger lever on the answer than
which model you pick, so it is a visible config value rather than an accident.

### Running it unattended

```bash
# systemd --user
systemctl --user enable --now trump-truth-forecast.service   # ExecStart=…/.venv/bin/python daemon.py

# or cron, if you'd rather not keep a process alive
*/15 * * * * cd /path/to/repo && .venv/bin/python daemon.py --once
0 4 * * *    cd /path/to/repo && .venv/bin/python daemon.py --backtest
```

SQLite holds all state and the poll is idempotent, so restarts and overlapping runs are safe.

## Note

This is a descriptive forecasting exercise about public posting volume. It makes no claims about
anyone's intentions and does not attribute causes to bursts.
