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

Then open <http://localhost:8000/>. The local server mounts the exports at
`/data/exports/` — the same layout the deployed bundle has, so both resolve data through the
same relative path and "works locally" cannot mean a different URL shape than the one users get.

---

## What it found

From the full archive (34,000+ posts, Feb 2022 → present; models fitted on the second term from
2025-01-20):

**It is nowhere near Poisson.** Variance runs ~16× the mean. A Poisson process at this rate would
put 95% of days between 10 and 28 posts; the actual range is 0 to 168. That failure is reported
rather than hidden, because it is the finding that motivates every other model: either the rate
isn't fixed, or the posts aren't independent.

And it is not the calendar in disguise, which is the obvious objection: a single fixed rate is
refuted by the weekly cycle alone. Grant a Poisson model everything it normally knows and it still
fails — rates fitted per weekday leave χ²/df of **14.6**, adding a causal 28-day level leaves
**15.7**, against 1.0 for a well-specified Poisson model. The overdispersion belongs to the
process, not to the model's ignorance about Mondays.

**It is heavy-tailed.** The busiest day is 11× the median, skewness 3.6, and the busiest 1% of days
carry 6% of all posts. So every figure is a median with an *asymmetric* interval, never `x ± y`.

**There is almost no day-to-day memory.** Once the weekly cycle is removed, autocorrelation is
inside the noise band at every lag (lag-1 ≈ 0.11, lag-7 falls from 0.027 to 0.007). Whatever makes
posts cluster works *within* a day, not across days.

**Self-excitation is real but very fast.** The Hawkes fit gives a branching ratio around 0.56 —
each post triggers ~0.56 further posts — with a kernel half-life of about a minute. That produces
the rapid-fire bursts you can see in the feed, but it barely moves the *weekly* total, which is
consistent with the flat autocorrelation.

**Weeks are genuinely hard to forecast.** The best model removes **3.0%** of climatology's CRPS.
That is the honest headline. Anyone quoting a confident weekly number is quoting noise. It is also
what most of the leaderboard's ordering is made of: the top twelve models span 18.24 to 18.83 CRPS,
which is inside the noise of 53 weeks.

**Most of a model's score is the calendar, not the model.** CRPS for the leading model falls from
**28.9** on Monday, with nothing of the week observed, to **8.8** on Saturday. Averaging across
cuts mostly measures how much of the week the forecaster had already been told.

**Every model is overconfident.** Across 53 weeks, not one model's 80% interval reached 80%: the
best manages 75%, the bulk sit between 57% and 73%, and Poisson and ZIP manage **27%** with
intervals a third of the width they need — the direct consequence of locking variance to the mean.
This is the most transferable thing here. "This week lands near 150" expires on Sunday; "forcing
variance to equal the mean on a heavy-tailed count process produces intervals that are wrong two
times in three" does not.

**The regime changes are an artifact.** The changepoint search was told to find four breaks, and
methods of that kind return exactly the number they are given — they will partition a straight
line. Told to justify each break instead, the same search finds **none**. The window still makes
sense; the four dates never did.

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
.github/      the same cadences on GitHub Actions, publishing to Pages
REVIEW.md     what a fresh read found: what was wrong, what is unstated, what to build next
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

Cuts run from **-1** (Monday, nothing of the week observed — the hardest forecast, and the one on
display for a seventh of all time) to **5** (Saturday complete). The fully-observed Sunday cut is
excluded: by then every model returns the answer exactly, scoring 0 CRPS with an interval that
covers by construction, and averaging that free win in would flatter coverage and shrink CRPS for
every model alike.

The live headline picks its models from the **development** leaderboard, never from the holdout.
A holdout can answer "how does the chosen procedure do on weeks nobody looked at" only while
nothing is chosen with it; selecting on eight weeks would spend the one clean sample on a choice
and then report the result as if it were still out-of-sample.

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
python daemon.py --reconcile       # re-read the last 14 days (deletions, corrections)
python daemon.py --serve-only      # just serve existing exports
python -m truthforecast.pipeline --backfill 2022-02-01
python -m truthforecast.pipeline --poll --forecast
python -m truthforecast.pipeline --reconcile 30
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
0 4 * * *    cd /path/to/repo && .venv/bin/python daemon.py --backtest --reconcile
```

SQLite holds all state and the poll is idempotent, so restarts and overlapping runs are safe.

The poll is defined by **coverage**, not by cadence: it walks the listing back until it reaches
the point the last completed poll got to, and only advances that watermark when the walk actually
closes the gap. This matters more than it sounds. Reading just the newest page and trusting it
means any interruption longer than a page of posts writes *quiet days* into the series rather than
gaps — and a quiet day is indistinguishable from a real one, so every model downstream learns from
it and nothing in the output looks wrong. Bursts are the whole subject here; a hundred posts inside
one interval is the event the project exists to measure, not an edge case.

## Hosting it on GitHub

`.github/workflows/update.yml` is the same daemon on GitHub's scheduler: poll and rebuild twice
an hour, reconcile and re-rank every model nightly, publish `site/` plus the exports to GitHub
Pages. The archive lives in the Actions cache rather than in git — it is a 25 MB binary that
changes every pass — and losing it costs one automatic backfill, because the mirror is the source
of truth and the cache is only a shortcut.

Three things about that platform are worth knowing before trusting what it publishes:

- **`schedule:` only fires on the default branch.** On any other branch the workflow exists and
  never runs by itself.
- **Scheduled runs are best-effort.** They queue under load and can be skipped entirely; "every
  30 minutes" really means "usually within the hour".
- **Private repositories bill Actions minutes and need a paid plan for Pages.** Public ones get
  both free. Two runs an hour is ~1,500 minutes a month, which does not fit the free private
  allowance — on a private repo, drop to hourly or accept the bill.

None of that is a problem as long as nobody reads a stale page as a live one, so every export
carries its own build time, the site states how old its data is, and it says so loudly when the
updater has stopped. Freshness is a claim the page has to earn, not one it makes by existing.

To turn it on: **Settings → Pages → Source: GitHub Actions**, then either wait for the schedule or
run the workflow manually from the Actions tab.

## Getting notified when he posts

`.github/workflows/notify.yml` polls every five minutes and pushes new posts to whichever channel
you configure. Set one of these as a repository secret (**Settings → Secrets and variables →
Actions**) and it starts working; anything unset is skipped.

| secret | channel |
|---|---|
| `NTFY_TOPIC` | [ntfy.sh](https://ntfy.sh) — install the app, subscribe to the topic, no account needed |
| `TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID` | a Telegram bot |
| `DISCORD_WEBHOOK` / `SLACK_WEBHOOK` | channel webhooks |
| `WEBHOOK_URL` | anything else — receives the raw JSON |

The first run adopts whatever is already stored and sends nothing, so a fresh database cannot page
you about last week. After that it is keyed on the mirror's ingest order rather than post
timestamps — a ReTruth carries the *original* author's date, so a timestamp watermark would either
miss it or replay months of history. A failed send does not advance the watermark: the next run
retries, which can duplicate an alert on a channel that did work. For a notifier that is the right
way round.

Bursts are handled: past five posts in one pass it sends one summary line instead of five
notifications, which is the difference between an alert and an attack on your phone.

**What "instant" is actually worth here.** Nothing downstream can be fresher than the mirror's own
crawl of Truth Social, and GitHub's scheduler queues under load:

| running it as | latency |
|---|---|
| `notify.yml` on Actions | 5-minute cron, plus GitHub's queueing — realistically 5–20 minutes |
| `TTF_POLL_SECONDS=60 python daemon.py` on any always-on machine | about a minute |
| the ceiling | whatever trumpstruth.org's own crawl lag is |

Every alert carries the post's own timestamp, so the lag is visible rather than something you have
to take on trust. If you want the fast version, run the daemon on a machine that stays up — the
poll is two requests, so a 60-second loop is nothing.

Run exactly one notifier. Two processes with separate databases keep separate watermarks and will
both alert you. The secrets are referenced only by `notify.yml`, so the site workflow cannot
double-send; a local daemon with the same environment variables set will.

```bash
python -m truthforecast.notify --dry-run   # print what would be sent
python -m truthforecast.notify --reset     # forget the watermark; next run adopts the present
```

### The fast version

Run it as a process instead of a cron slot. GitHub's scheduler floors out at five minutes and
queues under load; a loop you own does not.

```bash
docker run -d --restart=always \
  -e TTF_NTFY_TOPIC=your-topic-name \
  -e TTF_POLL_SECONDS=30 \
  -v truthforecast-data:/app/data \
  -p 8000:8000 truthforecast
```

The volume holds the archive and the coverage watermark, so a restart resumes rather than
re-walking 34,000 posts. At `TTF_POLL_SECONDS=30` the poll costs two requests every thirty
seconds and the only remaining delay is the mirror's.

### Why it cannot be instant

Four links, and only one of them is ours:

    Trump posts -> Truth Social publishes -> trumpstruth.org crawls -> we poll -> your phone
                                             \_____ unmeasured _____/   \__ 30s to 20min __/

The diagnostics page now measures the whole chain — post timestamp to the moment this pipeline
stored it, ReTruths excluded — so the number stops being a guess as soon as the poller has seen
enough live posts. Everything below the mirror is a floor we cannot get under by polling harder.

**The X API does not help, for two independent reasons.** It is the wrong data: Truth Social is
where he posts, and this project measures Truth Social volume — an X-based series would be a
different, much sparser thing. And real-time on X means the filtered stream, which is Pro tier at
$5,000/month minimum; the $200/month Basic tier has no streaming at all, and both tiers closed to
new signups during the June 2026 move to pay-per-use.

**Nor do the X accounts that relay Truth Social posts.** They exist, and watching one instead is a
reasonable-sounding idea, but it moves in the wrong direction. Those accounts are themselves
polling Truth Social or a mirror and republishing — their posts quote the original's timestamp,
which is what reposting looks like. Going through one means inheriting its crawl lag, plus its
posting interval, plus your own polling of X, and paying X for the privilege. You cannot be faster
than your source's source. It is also a dependency you cannot see into: when a relay account gets
suspended, rate-limited, or quietly stops, the silence is indistinguishable from "he did not post"
— the exact failure this project's own poller had to be rebuilt to avoid.

The useful move is upstream, not downstream.

### Reading Truth Social directly

Truth Social is a Mastodon fork, so the account exposes a standard RSS feed at
`https://truthsocial.com/@realDonaldTrump.rss`. That removes the mirror from the chain entirely —
one link instead of three. The catch is that it refuses datacenter traffic; from Google Cloud every
endpoint returns 403, which is why the archive uses the mirror in the first place. From a home
connection it may just work, and that depends on where you are rather than on anything in this
repo, so the code asks rather than assumes:

```bash
python -m truthforecast.ingest.direct --probe    # can this machine read it?
python -m truthforecast.ingest.direct --watch    # poll it every 30s and alert
```

The probe reports the status code, what `robots.txt` says, and how old the newest post in the feed
is. `--watch` refuses to run if robots.txt disallows the path.

This path feeds **alerts only** and never the modelled series. The archive is keyed on the mirror's
post id, which a direct read does not have, so merging the two could store the same post twice
under different keys — and a double-counted day is precisely the silent corruption the rest of this
codebase is built to prevent. Speed is worth having; it is not worth the series.

**Genuinely instant exists and is not for us.** Trump Media launched **Truth API** on 1 August
2026: licensed, millisecond delivery of posts from top Truth Social accounts, sold to
high-frequency trading firms. Reported pricing is $100,000 per month, or $60,000 on a three-year
contract. That is the actual price of "instant", and it tells you what the latency is worth to the
people who pay for it. Anything cheaper is polling something that is itself polling.

## Note

This is a descriptive forecasting exercise about public posting volume. It makes no claims about
anyone's intentions and does not attribute causes to bursts.
