# Review

A read of this project with fresh eyes: first whether the code does what it says, then whether
what it says is worth saying. Findings are ordered by how much they move a number someone might
act on. Each says what it is, what the evidence was, and whether it is fixed here or left alone
deliberately.

The short version: the engineering is careful and the statistical instincts are better than most
of what gets published on this subject — proper scoring rules, a pre-registered model list, a
leaderboard that shows its losers, a headline that admits the models barely beat climatology.
Three defects were undermining exactly the parts the project is proudest of, and one of them was
inverting the meaning of its own holdout. All three are fixed. The rest is a list of things that
are true, unstated, and worth stating.

---

## 1. The evaluation scored a free win and skipped the hardest case

**Was:** `BACKTEST.cut_days = (0, 1, 2, 3, 4, 5, 6)`.

Cut 6 means the week is fully observed. Every model then returns the answer exactly: CRPS 0, an
interval of zero width that covers by construction, and a PIT value drawn from pure uniform noise.
Confirmed in the shipped `backtest.json` — `crps_by_cut` is `0.0` at cut 6 for all twenty models,
without exception.

One seventh of every published figure was therefore that free win:

| published | what it was over the cuts that carry information |
|---|---|
| `hier-dow-eb` coverage80 **0.771** | **0.733** |
| `poisson-glm` coverage80 **0.388** | **0.286** |
| `hier-dow-eb` CRPS **14.11** posts | **16.5** posts |
| `hier-dow-eb` width80 **51.5** posts | **60.1** posts |

Those are arithmetic, not estimates: every cut carries exactly 53 rows, and the degenerate one
contributes a known 0 CRPS, a known 0 width and a guaranteed cover.

The README's honest headline — "nearly every model's 80% interval contained the truth only 70–77%
of the time" — was too *generous* to the models. Skill scores survive, because they are ratios and
the dilution cancels out of them.

Meanwhile cut **-1** — Monday, nothing of the week observed — was never scored at all, although
`ForecastTask` supports it and the live site is in exactly that state for a seventh of every week.
The backtest was scoring a situation that never occurs and skipping the hardest one that does.

**Fixed:** `cut_days = (-1, 0, 1, 2, 3, 4, 5)`, which is precisely the set of situations the site
is ever in. `coverage80_by_cut` is now exported alongside `crps_by_cut`, so an average over cuts
can no longer hide a model that is fine on Saturday and badly overconfident on Monday.

Re-run on the current archive, the board reads: best CRPS **18.24** (was 14.11), best coverage80
**0.752** against a nominal 0.80, `poisson-glm` at **0.272**, and CRPS for the leading model
falling from **28.9** on Monday to **8.8** on Saturday. That last spread is the useful new number —
most of a model's average score was always a statement about how much of the week it had already
been told.

## 2. The holdout was being used to choose

**Was:** `pipeline.load_ranking()` read `leaderboard_holdout` first and fell back to `leaderboard`.
`forecast.headline()` takes the top three of that ranking and averages their quantiles into the
number on the front page.

So the eight-week holdout was selecting the live models — the one thing a holdout cannot do and
remain a holdout — while `report.py` printed, in the same file, *"The holdout leaderboard is the
honest number."* After selection it is not an honest number about anything; it is a training set
of eight weeks reported as if untouched.

It is also the noisiest board available, and the reordering shows it. Dev rank → holdout rank:

    hier-dow-eb   #1 → #7        hawkes      #14 → #3
    negbin-hmm    #3 → #6        ingarch      #8 → #2
    zinb-glm     #12 → #8        empirical-dow #11 → #14

across a CRPS spread of 14.11 to 14.53 — a 3% band over 53 weeks. That is a shuffle, not a
ranking. The consequence was concrete: the site was leading with `hawkes`, which sits 14th of 20
on the fuller sample with the second-worst calibration on the board (coverage 0.607 against a
nominal 0.80).

The fresh run reproduces it exactly, which is the point — this is not a one-off draw. On the
corrected board `hawkes` is 14th of 20 with the *worst* coverage on the leaderboard (0.523), and
the eight-week holdout still ranks it **2nd**. `empirical-dow-8w`, 16th on the development board,
comes 4th on the holdout. Eight weeks cannot tell these models apart, and it was being asked to.

**Fixed:** selection comes from the development leaderboard. The holdout is still computed, still
displayed, and now actually means what the notes say it means.

## 3. LightGBM was tested on a feature vector it had never been trained on

**Was:** `QuantileGBM.sample_remaining` built its prediction row from `self._last_feats[-1]` — the
features of the final day of *history* — for every cut, while `_build_training` anchors each
training row on the cut day itself. With the one-day embargo, history always ends on a Saturday.
So every prediction, at every cut, handed the model a Saturday row whose lag and rolling features
stop before the target week began, while training had taught it rows anchored on all seven
weekdays with lags reaching into the week.

**Fixed:** the row is rebuilt as of the cut, extending the series with the days of the target week
already known, which is what training assumed. One residual skew remains and is not worth
contorting the code for: at cut −1 the embargo leaves the anchor on Saturday where training used
Sunday.

**And it barely mattered — which is worth more than the fix.** A/B under an identical protocol,
same data, same cuts, same reference model:

    before (features of the last history day)   CRPS 22.43   skill −0.194
    after  (features as of the cut)             CRPS 21.98   skill −0.170

The bug was real and cost about 0.45 posts of CRPS, a fifth of the model's gap to climatology. It
was not the reason the model loses. `lgbm-quantile` is still 19th of 20 with 44% coverage on an
80% interval, and the reason is that it is trained on about 370 rows derived from 53 weeks — a
gradient booster asked to learn a heavy tail from a sample that contains almost none of it. The
tempting write-up here was "ML was sabotaged by a plumbing bug"; the measurement says otherwise,
and the measurement took five minutes.

## 4. The poll could invent quiet days, silently, forever

**Was:** `poll()` fetched the RSS feed (100 newest) plus the first listing page (100), and stopped.
No gap detection, no watermark, no way to notice.

The failure mode is the bad kind. Miss a stretch of posts and the series does not record a gap —
it records **low counts**, which are indistinguishable from real quiet days. Every model
downstream fits them faithfully, the diagnostics report them as tail behaviour, and nothing
anywhere looks wrong. The archive is only self-correcting if something re-reads it, and nothing
did.

On a 15-minute local daemon this was mostly latent. Hosted on GitHub's scheduler it is not: those
runs queue under load and can be skipped outright. And the specific event that breaks a 200-post
window is a burst — the busiest day in the record is 168 posts, the fitted Hawkes kernel has a
half-life of about a minute, and bursts are the entire subject of the project. The one thing the
poll had to survive was the one thing it was built to lose.

**Fixed:** the poll is now defined by coverage rather than cadence. It walks the listing back to
the point the last *completed* poll reached (plus two hours of overlap) and only advances that
watermark when the walk actually got there — an interrupted catch-up leaves the watermark alone
and the next run picks the gap back up. `walk_back` now reports how far it got, so "stopped early"
can no longer be mistaken for "finished". A normal pass is one or two pages; a three-day outage
costs about a minute of extra requests.

## 5. Nothing ever looked at the past again

**Was:** a poll only ever reads what is new. But posts get deleted and the mirror marks them,
and timestamps get sharpened from minute to second precision when a post reaches the feed after
the listing. Neither arrives on its own. The stored history drifted away from what the source
currently says, invisibly, because nothing re-checked.

**Fixed:** `reconcile(days=14)` re-reads the recent past, nightly.

## 6. Four regime changes, because the search was told to find four

**Was:** `structural_breaks` calls `Binseg().predict(n_bkps=min(4, len(weekly)//20))`. That method
returns exactly the number of breakpoints it is given. It will partition a straight line into four
segments just as willingly as a step function. The site displayed the result under the heading
"Has the process changed?", which is a question the code was never allowed to answer "no" to.

A penalized search on the same signal — one that has to pay for each break it claims, so zero is
an available answer — finds **zero**. The four dates on the page are a partition of noise.

This cuts both ways, and both are worth saying. The site was overstating its evidence for regime
change; and the project's own config choice (`TTF_WINDOW_START = 2025-01-20`, chosen because
"fitting across a structural break describes neither regime") turns out to be *better* supported
than its diagnostics page suggested, since there is no detectable break inside the window to fit
across.

**Fixed:** `n_breaks_penalized` is computed, exported and shown, and the forced segmentation is now
labelled as forced. `window_vs_breaks` reports whether any surviving break falls inside the
modelling window.

## 7. The day in progress was thrown away

**Was:** `week_progress` excluded today from `observed`, and the models drew a whole fresh day in
its place. At 9pm on a Friday that had produced 4 posts, the forecast discarded the 4 and simulated
~19. The site displayed "today so far" in a tile immediately beside a projection that ignored it.

It is worth being precise about the failure, because the obvious description is wrong. **It was not
biased.** Measured across every day since the window start, the error from discarding today
averages 0.00 at every hour. It was *uninformative*: CRPS flat at 8.53 from midnight to midnight,
never learning anything from a day already 92% decided. It reads high only when today is quiet,
low when today is loud.

The docstring shows the choice was deliberate — "treating a half-finished day as a finished one
biases every projection downwards" — and that reasoning is correct at 00:01 and wrong by evening.
What is missing is any measurement of the cost, and the backtest structurally cannot supply one:
it scores forecasts only at day boundaries, on whole observed days, so the intraday state the live
site occupies for 23 hours out of 24 is never evaluated.

**Two obvious fixes are both worse.** Walk-forward CRPS, expanding origin, strictly past-only data:

| hour ET | discard *(was)* | scale by elapsed share | Poisson-Gamma conjugate | **empirical** |
|---|---|---|---|---|
| 06:00 | 8.53 | 31.50 | 13.77 | **7.73** |
| 09:00 | 8.53 | 17.96 | 11.25 | **7.18** |
| 12:00 | 8.53 | 12.28 | 8.88 | **6.05** |
| 15:00 | 8.53 | 8.96 | 6.76 | **5.03** |
| 18:00 | 8.53 | 5.85 | 4.67 | **3.73** |
| 21:00 | 8.53 | 2.78 | 2.20 | **1.91** |

Scaling `posts_so_far / fraction_elapsed` is four times worse than doing nothing at 6am — dividing
by a small fraction multiplies the morning's noise — and does not overtake the old behaviour until
mid-afternoon. Its failure case is stark: nothing posted by noon predicts a total of zero, across a
record of 556 days containing no zero days at all.

The Poisson-Gamma conjugate update, the textbook answer, also loses to discarding until noon. It
models the day as binomial thinning of a fixed intraday clock, so an empty morning is strong
evidence of a quiet day. His mornings do not work that way: on the 44 days with nothing posted by
noon, the median finish was 6, the mean 10.1, and one reached 69. He starts when he starts.

**Fixed** with the estimator that assumes nothing about the clock: match past days on posts-so-far
at this hour (normalised by that weekday's level), resample what those days did with their
remaining time. It wins at every hour, and silence by noon predicts about 6 because that is what
silence by noon actually produced. `truthforecast/partial.py`; today now enters `observed` like any
other day and only its remainder is simulated.

Live effect when this was written, at 21:20 on a quiet Friday: median **147 → 134**, and the site
now states how much of the day is already spent next to the number.

## 8. "Variance is 16× the mean" was aimed at a model nobody proposed

**Was:** the dispersion test compares the daily counts against a single fixed Poisson rate. That
model is refuted by the weekly cycle alone — a busy Monday against a quiet Thursday inflates the
ratio all by itself — and refuting it says nothing about whether a Poisson *GLM*, which is what
the model zoo actually contains, is adequate. The strongest claim on the site was resting on the
weakest comparison available.

**Fixed, and the claim got stronger.** Granting the Poisson model everything it normally knows
barely helps:

| conditioning | Pearson χ²/df |
|---|---|
| none (raw variance ÷ mean) | 15.71 |
| rates fitted per weekday | 14.56 |
| weekday × causal 28-day level | 15.70 |

against 1.0 for a well-specified Poisson model. The overdispersion is not the calendar in
disguise and it is not a drifting level in disguise; it is a property of the process. That the
level term makes it slightly *worse* is itself consistent with the near-zero autocorrelation
reported elsewhere — there is no local level to condition on.

---

## Left alone, deliberately — but you should know

**The backtest runs on revised data.** The series is "what trumpstruth.org says today", not "what
was knowable then". Deleted posts vanish from history, late arrivals appear in it, and a post
reclassified as a ReTruth leaves the modelled series retroactively. Every backtest score is
therefore measured against a history no live forecaster ever had. `first_seen_utc` is already
stored on every row and currently unused; a vintage backtest that filters to
`first_seen_utc < origin` is maybe thirty lines and would put a number on how optimistic the
current scores are. Until then, treat the leaderboard as an upper bound on live performance.

**~~Today's posts are thrown away.~~ Fixed — see below.**

**The headline and the threshold probabilities use different ensembles.** `headline` averages
quantiles across the top three models (Vincentization); `threshold_probabilities` averages
probabilities across the same three (a linear pool). These are different distributions, so
"P(≥ 140)" is not the probability implied by the interval printed above it. They will usually
agree closely and occasionally will not.

**Midnight is a modelling choice, not a fact.** Eastern calendar days split a late-night burst
across two days — the largest day in the record peaks at hour 23. A 4am-to-4am day would change
the daily maximum, the variance, and therefore the headline dispersion ratio. A one-line
robustness check nobody has run.

**Deleted posts are counted.** `load_frame(include_deleted=True)`. Defensible — he did post them —
but it makes the series depend on when the mirror happened to look, and a post deleted before the
mirror's first pass is invisible to everyone. Worth one sentence on the About page.

**Zero-inflated models solve a problem this data does not have.** Two zero days in 557. Keeping
them is right — they are pre-registered, and their failure is informative — but the models page
could say that they are negative controls rather than candidates.

**The embargo makes the backtest slightly pessimistic.** Training stops a day before the target
week, so backtest models never see the immediately preceding Sunday; the live path does. The
direction is safe (backtest understates live skill), which is the right way round.

---

## Is the idea sound?

Mostly yes, and it is more honest than the genre it belongs to. The thing being predicted is
precisely defined, the uncertainty is reported as a distribution rather than a point, the scoring
rules are proper, the model list is fixed in advance, and the front page tells you the models
barely beat climatology. That last one is rare enough to be worth saying out loud: most projects
in this shape are built to make a number look confident.

Three things about the premise are worth naming, though, because the site does not name them.

**The forecastable quantity and the interesting quantity are not the same thing.** A weekly total
is seven draws from a heavy-tailed distribution with almost no day-to-day memory. Given that,
climatology is close to the best any model can do, and the 3% CRPS gap between best and worst
non-broken model is mostly the noise of 53 weeks. Twenty models are being ranked on differences
that largely cannot exist. The project half-says this in prose; it could say it with a number by
computing an irreducibility bound — the CRPS a forecaster would achieve *knowing the true daily
rates* — which would turn "we beat climatology by 2.9%" from a disappointment into a measurement
of how much signal there was to get.

**The construct and the measure differ by more than the page implies.** The site says it tracks
how often Trump posts. It measures posts visible on a third-party mirror, excluding ReTruths,
including deleted posts, bucketed by Eastern midnight. Every one of those choices is defended
somewhere in the code, and none of them is wrong. But the gap between "how often he posts" and
"what the mirror had when we asked" is exactly where a reader would be misled, and it costs two
sentences to close.

**The most valuable output is not the forecast.** It is the calibration finding: across 53 weeks,
nearly every model's 80% interval covered the truth 65–75% of the time, and the two Poisson
variants managed 29%. That result transfers — it is a statement about what happens when you force
variance to equal the mean on a heavy-tailed count process — in a way that "this week will land
near 150" never will. It deserves to be the headline of the models page rather than a column in
the table.

---

## Worth building next

In rough order of what I would do first.

1. **Log live forecasts and score them prospectively.** This is the one thing hosting unlocks, and
   it changes what the project *is*. Right now every number is retrospective: the models are
   graded on history, by code written after that history happened. A scheduled job that appends
   each live forecast — timestamped, immutable, committed — and scores it once the week closes
   turns a backtest into a genuine forecasting record. After a year it is evidence nobody can
   accuse of hindsight, and it is the only version of this project whose calibration claims cannot
   be doubted. Everything needed already exists; it is an append-only JSONL file and a scoring
   pass.

2. **A vintage backtest.** Use `first_seen_utc` to reconstruct what was knowable at each origin
   and re-run. The gap between that and the current leaderboard is the size of the optimism
   nobody has measured.

3. **Condition on the partial day.** The hourly profile is computed and unused; "he has posted 40
   times and it is 9pm on Friday" is real information currently discarded at every cut.

4. **Report skill by cut, not just overall.** `crps_by_cut` already shows CRPS falling from ~23 on
   Monday to ~9 on Saturday. Most of a model's average score is therefore a statement about how
   much of the week is already known — a fact about the calendar, not the model. Skill against
   climatology *at each cut* is the comparison that isolates the model.

5. **Say what would change your mind.** The project pre-registers its model list; it could also
   pre-register the result that would make it abandon the exercise — e.g. "if no model beats
   climatology by more than 5% CRPS after 100 scored weeks, the honest conclusion is that weekly
   volume is not forecastable and this site should say so on the front page." Written down in
   advance, that is a much stronger claim than any leaderboard.
