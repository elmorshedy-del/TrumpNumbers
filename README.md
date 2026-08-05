# Trump Truth Social post tracker & weekly forecast

Tracks how often Donald Trump posts on Truth Social, shows the posts, and projects where the
current **Sunday–Saturday** total will land — the same week, post types and exclusions the Kalshi
weekly market resolves on, so the number is comparable to the one people quote. ~20 statistical
models are backtested against each other using proper scoring rules.

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

**It is nowhere near Poisson.** Variance runs ~15× the mean. A Poisson process at this rate would
put 95% of days between 10 and 28 posts; the actual range is 0 to 166. That failure is reported
rather than hidden, because it is the finding that motivates every other model: either the rate
isn't fixed, or the posts aren't independent.

And it is not the calendar in disguise, which is the obvious objection: a single fixed rate is
refuted by the weekly cycle alone. Grant a Poisson model everything it normally knows and it still
fails — rates fitted per weekday leave χ²/df of **14.3**, adding a causal 28-day level leaves
**15.4**, against 1.0 for a well-specified Poisson model. The overdispersion belongs to the
process, not to the model's ignorance about Mondays.

**It is heavy-tailed.** The busiest day is 11× the median, skewness 3.5, and the busiest 1% of days
carry 6% of all posts. So every figure is a median with an *asymmetric* interval, never `x ± y`.

**The day in progress is worth conditioning on, and the clock is not.** Today's posts are neither
ignored nor scaled up — both score worse, in opposite halves of the day. Instead each model's own
view of today is conditioned on how today has actually gone, using the days whose posting looked
most like it at this exact moment, in continuous elapsed time. That last part matters more than it
sounds: conditioning on the *hour* made the projection a step function that held flat for an hour
and then jumped, and handing the day between two different estimators at midnight moved the
headline by **12 posts** at the stroke of a clock with no information arriving. At elapsed time
zero the conditioning is now the identity, so the day a model draws at 23:59 is the day it draws at
00:00, and the projection is continuous across the boundary by construction.

**There is almost no day-to-day memory.** Once the weekly cycle is removed, autocorrelation is
inside the noise band at every lag (lag-1 ≈ 0.11, lag-7 falls from 0.027 to 0.007). Whatever makes
posts cluster works *within* a day, not across days.

**Self-excitation is real but very fast.** The Hawkes fit gives a branching ratio around 0.56 —
each post triggers ~0.56 further posts — with a kernel half-life of about a minute. That produces
the rapid-fire bursts you can see in the feed, but it barely moves the *weekly* total, which is
consistent with the flat autocorrelation.

**Weeks are genuinely hard to forecast.** The best model removes **2.3%** of climatology's CRPS.
That is the honest headline. Anyone quoting a confident weekly number is quoting noise. It is also
what most of the leaderboard's ordering is made of: the top twelve models span 17.84 to 18.28 CRPS,
which is inside the noise of 52 weeks.

**Choosing a model costs more than having a good one is worth.** The site's own selection rule —
take the top three of the leaderboard and average them — is now scored week by week like any other
forecaster, picking only from weeks that had already closed. It loses about **0.5 CRPS (2.5%)** to
the model that turns out best on those weeks. That gap is the price of not knowing in advance which
model will win, and it is *larger than the 2.3% the winning model gains over climatology*. Four
different selection rules, from "trust one model" to "blend five", sit within 0.11 CRPS of each
other, so there was never anything to tune. This is the most useful thing on the models page and it
was invisible until the deployed rule was put on the board it selects from.

**Most of a model's score is the calendar, not the model.** CRPS for the leading model falls from
**28.0** on the week's opening day, with nothing observed, to **7.6** on its last full day. Averaging across
cuts mostly measures how much of the week the forecaster had already been told.

**Almost every model is overconfident.** Across 52 weeks, coverage of the nominal 80% interval
runs from **0.34** to **0.81**: only the top two reach 80%, and Poisson and ZIP manage **34%** with
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
  partial.py  the day in progress, conditioned in continuous time
  combine.py  how several predictive distributions become the one on the page
  forecast.py current-week projection
  live.py     the append-only record of what was forecast before the fact
  record.py   what the markets said while the week was open (Kalshi, Polymarket, the schedule)
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
  month-old date, by both the feed and the listing. Counting them as filed would credit activity to
  days with none, so they are **re-dated from the mirror's ingest order** before being counted —
  the ids either side of a ReTruth bracket the moment it actually appeared (`redate_retruths`).

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

Cuts run from **-1** (the week's opening day with nothing observed — the hardest forecast, and the
one on display for a seventh of all time) to **5** (six days complete). The fully-observed final cut is
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

Three caveats the numbers themselves demand:
- Each week is scored at seven cuts, so forecasts aren't independent. PIT p-values are a strong
  directional signal about over/under-confidence, not exact significance.
- Gaps of a few hundredths of CRPS between neighbouring models are inside the noise of ~50 weeks.
  Read the groupings, not the ordering.
- Every draw is seeded from `(model, week, cut)`, so the board reproduces exactly on unchanged
  data. That is not tidiness: re-running under a different seed moves individual models by up to
  **0.05 CRPS** and reorders the top five, so anything smaller than that is the random number
  generator rather than a result. Set `TTF_SEED` to measure it yourself.

Skill is also reported **at each cut**, not only on average. A model's average CRPS is mostly a
statement about the calendar — 28.0 with nothing observed against 7.6 with six days banked — so
dividing by climatology at the *same* cut is what isolates the model. The same split is worth
applying to calibration: the leading model's headline 80% coverage is an average of **0.71 early in
the week and 0.89 late**. It is not calibrated at any cut; it is wrong in two directions that
happen to cancel.

### The number on the front page is scored too

The tracker does not show a model, it shows a rule, and a combination is not entitled to its
members' scores. That rule is replayed week by week under the name `headline-top3`, choosing its
models using only weeks that had already closed, and it appears on the models page with its own
CRPS and coverage — alongside the three alternatives it had never been measured against.

### The prospective record

Everything above is retrospective. However careful the embargoes, the *choices* — which models,
which window, which convention — were all made with the whole series visible, so a backtest is an
upper bound on live performance and cannot be anything else.

So every forecast the pipeline publishes is appended to `data/live_forecasts.jsonl` the moment it
is built, committed, and scored once the week closes. One line per (week, cut), and it is the
**first** forecast made at that cut that stands: a register you may keep amending until the answer
arrives is not a register. The site reports how many weeks it has and declines to draw conclusions
from four.

The file is not in the repository — it appears on the first scheduled run. Seeding it with
forecasts made during development, by code still being changed against a database being rebuilt,
would put the least checkable rows at the very start of a record whose only value is that it can be
checked.

```bash
python -m truthforecast.live --score   # grade the weeks that have closed
python -m truthforecast.live --list    # the raw register
```

### The market record

The forecast can be rebuilt from the mirror at any time. What can never be rebuilt is what the
**markets** believed while the week was still open — and that is the only data that can ever
answer the question this project keeps walking into: are those markets priced well? Backtests
against climatology say whether a model has skill; only a record of prices says whether that
skill was worth anything after fees.

`truthforecast/record.py` collects it as it happens, under one rule: *record whatever cannot be
reconstructed later, at the finest granularity that is cheap, and decide what it is for
afterwards.* Kalshi's weekly post-count market — and its deleted-post market, which prices the
exact quantity this pipeline's counting convention subtracts — gets hourly order-book snapshots
plus daily pulls of 1-minute candlesticks, the full trade tape and lifecycle metadata.
Polymarket's parallel post-count markets (same process, *offset* windows) get the same
treatment, and matter twice over: their public history endpoint returned nothing for closed
markets when checked, so for that venue the snapshot **is** the record — and two venues pricing
one process is the cheapest test of market rationality there is. When they disagree, at least
one of them is wrong.

Two non-market streams ride along. The president's public schedule (Factba.se's feed) is
recorded point-in-time: entries are appended when first seen, because the published archive is
revised as pool reports land, and a schedule feature fed to any future backtest must know what
was knowable the evening before, not what the record was amended to say afterwards. The first
run seeds the whole archive and marks every entry `retrospective` for exactly that reason. And
each pipeline pass appends its own headline to `forecast_history.jsonl` — the live register
keeps the *first* forecast per (week, cut), which is the right rule for a register and too
coarse to line up against a price that moves all day.

Everything is append-only JSONL under `data/record/`, committed to git the way the live register
is, with closed days' book files and closed months' candle and trade files gzipped in place.
The full ten-week backfill plus the calendar archive lands around 15 MB; steady state adds a few
hundred kilobytes a day, most of it the hourly books. `record.yml` runs it on GitHub's schedule,
dependency-free — the recorder is standard-library only — and the first daily run performs the
whole backfill by itself, because every incremental pull starts from each market's open when it
has no state to say otherwise.

```bash
python -m truthforecast.record --snapshot   # order books, both venues, right now
python -m truthforecast.record --daily      # candles, trades, metadata, calendar, rotation
```

What is deliberately **not** recorded, because it can be fetched retroactively forever: news
archives, election calendars, court dockets, equity prices. A recorder that hoards the
reconstructible spends its byte budget insuring the wrong risk — and the risk being insured
here is specific: the order book at 11pm during a burst exists once, and never again.

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
python -m truthforecast.record --snapshot     # market order books, both venues
python -m truthforecast.record --daily        # candles, trades, calendar, rotation
python -m pytest tests/ -q
```

Config lives in `truthforecast/config.py`. Two settings decide what the headline number *means*.

**The counting convention** (`CountConvention`) defaults to the Kalshi weekly market: Sunday 00:00
through Saturday 23:59 ET, counting Truths, ReTruths and Quote Truths, excluding deleted posts. It
was previously Monday–Sunday, originals only, deleted included — which is a cleaner measure of
original authorship but is not what any market resolves on, and the gap is not small: on the week
of 26 July the two conventions read 166 and 97. Set `TTF_WEEK_ANCHOR=MON`,
`TTF_INCLUDE_RETRUTHS=0`, `TTF_INCLUDE_DELETED=1` to get the original behaviour back.

Counting ReTruths is only honest because they are re-dated first. The mirror files a ReTruth under
the *original* author's timestamp, so counting them as filed would credit activity to days on which
nothing happened. The mirror's ingest order brackets the real reshare moment, which recovers it to
within hours — 72% of stored ReTruth dates agree to within a day, and the other 28% are precisely
the ones the stored date gets wrong, one by 420 days.

**The modelling window** (`TTF_WINDOW_START`): which slice of history you fit is a bigger lever on
the answer than which model you pick, so it is a visible config value rather than an accident.

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

That click can be automated, but it needs a credential the built-in `GITHUB_TOKEN` deliberately
is not: enabling Pages is a repository-admin operation. If you would rather not click, create a
token and store it as the repository secret `PAGES_TOKEN`:

- **fine-grained**, scoped to this repository only, with *Pages: read and write* and
  *Administration: read and write* (fine-grained tokens use the GitHub App permission model, and
  the action documents `administration:write` + `pages:write` for Apps);
- or a **classic** token with the `repo` scope, which is much broader — prefer the fine-grained one;
- set the shortest expiry offered.

The workflow's first run then enables Pages itself. **Delete the secret and revoke the token once
it has worked** — it is a one-time setup step, and a standing admin-scoped credential is a poor
trade for it. With no `PAGES_TOKEN` present the step is skipped entirely and nothing changes.

Never put a token in a file in the repository. This one is public, and committed credentials are
harvested by bots within seconds of the push — being deleted in a later commit does not help,
because the value stays in the git history and in every clone already taken.

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
