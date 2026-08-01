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

## 9. The headline measured a week nobody else was measuring

**Was:** Monday–Sunday, original posts only, deleted posts included. Every one of those was
hardcoded, and together they decided what the number *meant*.

The Kalshi weekly market — the thing people actually quote for this — resolves on Sunday 00:00
through Saturday 23:59 ET, counting Truths, ReTruths and Quote Truths, excluding deleted posts.
Three differences, all invisible from the page.

The gap is not cosmetic. On the week of 26 July the site read **97** while the market's week stood
at **164**, and both were correct about different things. The single day between the two windows —
Sunday 26 July — was a 57-post day, the loudest in three weeks: the market's week opens with it,
the site's excluded it. A reader comparing the site's number to a Kalshi bracket would conclude the
market was mispriced by a factor of two, when in fact 140–159 had already been passed.

**Fixed.** `config.CountConvention` now defaults to the market's definition, and every consumer
follows it — daily counts, the weekly resample anchor, the week-progress arithmetic, the backtest's
week boundaries, and the event-level feeds for Hawkes and the partial-day sampler. Setting
`TTF_WEEK_ANCHOR=MON`, `TTF_INCLUDE_RETRUTHS=0`, `TTF_INCLUDE_DELETED=1` restores the original,
which is a cleaner measure of original authorship — it just answers a different question.

Counting ReTruths was previously impossible for a good reason: the mirror files them under the
*original* author's timestamp, so a reshare of a month-old post lands a month back, and counting
them as filed would credit activity to days on which nothing happened. That is now recoverable.
`trumpstruth_id` is the mirror's arrival order, so the ids either side of a ReTruth bracket the
moment it appeared; interpolating the neighbouring originals places the reshare within hours. 72%
of stored ReTruth dates agree to within a day — he mostly reshares fresh material — and the other
28% are exactly the ones the stored date gets wrong, one by 420 days.

One landmine this exposed: models index day-of-week pools by `remaining_dows`, which assumed
position-in-week equalled pandas weekday. Under a Sunday-anchored week they diverge, and a model
forecasting the week's opening day would have read Monday's history to predict a Sunday.
`ForecastTask.remaining_dows` now does the mapping.

**What changed, and what did not.** Daily-level conclusions barely moved — mean 18.80 → 18.87,
median 15 → 15, variance/mean 15.7 → 15.4 — so every finding above survives intact. Weekly figures
moved, because the boundary moved: the leaderboard is 52 weeks rather than 53, best CRPS 18.24 →
17.84, and calibration *improved*, with the top model's 80% coverage going 0.728 → **0.799**
against a nominal 0.80.

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

---
---

# Second review

A second pass, over the code as the first review left it. Same method: check whether the code does
what it says, then whether what it says is worth saying. Same ordering: by how much a finding moves
a number someone might act on.

The first review fixed three defects and left a list. This pass found **twelve**, and they share a
shape worth naming before the list starts. Not one of them raised an exception, produced a NaN, or
failed a test. Every one of them was a plausible number that was wrong — a week cut on the wrong
day, a model told to forecast Tuesday and asked about Saturday, a distribution that had been given
a name it did not have. Three of them were contradicted by a docstring directly above the code. In
a project whose entire subject is the difference between a number and a *justified* number, that is
the failure mode to design against, and the tests added here (`tests/test_regressions.py`) exist
because none of the existing ones could have caught any of it.

One of them (§8) came from the user watching the live page rather than from reading the
code, which is worth recording: it was invisible in every test and every export, and obvious
within a minute of looking at the number move.

The largest single finding is not in the list. It is this: **the cost of not knowing in advance
which model will win is larger than everything the models win in the first place.** See §7.

---

## 1. The holdout was still choosing — just less of it

**Was:** the first review's §2 moved live selection off the holdout board and onto "the development
leaderboard". `pipeline.load_ranking()` reads `backtest.json → leaderboard`. But `build_report`
computed that board with `leaderboard(rows)` over **every** row — the eight holdout weeks included —
while the notes shipped in the same file said "the holdout is reported, never used to select".

So the fix went four fifths of the way. The holdout stopped being the *only* selector and became
one seventh of the selection sample instead, which is worse than either honest option: the holdout
is contaminated, and the contamination is now invisible because the leak is small.

It is not cosmetic here, because the two boards genuinely disagree. Development weeks only, against
all weeks:

    development (44 weeks)          all weeks (52)
    1 hier-dow-eb                   1 hier-dow-eb
    2 negbin-glm-trend              2 log-sarima
    3 log-sarima                    3 negbin-glm-trend
    4 negbin-hmm                    4 zinb-glm
    5 poisson-hmm                   5 ensemble-pool

`negbin-hmm` is 4th on the board that is allowed to choose and 9th on the board that is not.

**Fixed:** `leaderboard` is the development board and nothing else. The full-sample board is still
computed and exported as `leaderboard_all_weeks`, labelled for reading rather than selecting.

## 2. Half the zoo cut the week on the wrong day

**Was:** the first review moved the counting convention to the Kalshi week — Sunday through
Saturday — and reported that it had chased the change through "every consumer". Four models were
still slicing history into **Monday-anchored** weeks while being scored against Sunday-anchored
ones:

- `LastWeekModel` and `TrailingMedianModel`: `history.resample("W-SUN")`, which is Monday-to-Sunday.
  "Last week's total" was a week offset one day from the week being forecast.
- `BlockBootstrapModel`: grouped days by `index - dayofweek`, i.e. by Monday. The model's entire
  argument is that it resamples an *intact* stretch of one historical week, preserving whatever
  clustering lives inside it. Anchored a day off, every block straddled two weeks.
- `QuantileGBM`: `_build_training` anchored training rows on Monday weeks, so the `cut`,
  `observed_sum` and `days_left` features meant one thing in training and another at prediction.
  Cut 0 taught it "the opening day is banked, six to go" using a Monday and then asked it about a
  Sunday.

This is the same class of defect the first review found in `remaining_dows` and fixed there — the
convention change was chased through the *task* but not through the *models*.

**Fixed:** one `_week_starts` helper per module, driven by `CONVENTION`. Set `TTF_WEEK_ANCHOR=MON`
and everything moves together.

## 3. log-SARIMA was forecasting days that had already happened

**Was:** `LogSarimaModel.sample_days` called `self.res.get_forecast(steps=len(remaining_days))`.

A state-space model forecasts forward from the end of its training data. The remaining days of the
target week do not start there — the backtest embargoes a day before the week opens, and at a
mid-week cut the observed days sit in between. At a Friday cut the single remaining day is the
**eighth** step past the end of history; the model was being asked for the first. It was answering
a different question at every cut, and for a model with a period-7 seasonal term that is not a
shift but a **phase error**: a Saturday forecast carrying Sunday's seasonal component.

**Fixed:** `ForecastTask` now exposes `remaining_dates` and `gap_days`, and the model steps to the
day it was actually asked about. This also fixes a second-order case nobody would have found by
inspection: `log-sarima` refits only every second week, so on the off weeks the offsets are seven
days larger again, and the fix handles that for free because it works in calendar days rather than
in steps.

## 4. Both models with memory forgot the week they were forecasting

**Was:** the two models that exist *because* they carry state were throwing it away at the moment
it pays.

`INGARCHModel` seeded `prev_y` from the target week's most recent observed day but left
`prev_lambda` at the end of history — a state from up to nine days earlier paired with an
observation from yesterday, in a recursion whose persistence term `b·log(λ)` is most of what the
model is. Two situations it could not tell apart: one loud day at the end of a quiet week, and five
loud days in a row. Both ended on a 90-post day, so both got the same forecast.

`PoissonHMM` and `NegBinHMM` ignored `task.observed` entirely. The state belief came from
`predict_proba` at the end of history and was propagated blind. At a Friday cut that belief is more
than a week stale, and the six days of the target week the forecaster genuinely knows about never
entered it. "He has had two quiet days — does that mean anything for the weekend?" is the question
this model exists to answer, and it was answering it from last week's mood.

**Fixed:** both now run their filter forward from the end of history, stepping blind through the
embargoed days and taking a Bayes update on each observed one. The HMM update is done in logs — a
168-post day against a rate of 8 underflows a Poisson pmf to exactly zero, and the resulting 0/0
would have silently reset the belief to uniform.

**And here the fix made the score slightly worse, which is the finding.** `negbin-hmm` goes 17.89 →
18.18 CRPS. That is not the filter misbehaving; it is the data. The fitted busy state has a
stickiness of 0.29 and an expected run length of **1.4 days**, so knowing today's regime says almost
nothing about tomorrow's — and conditioning on it just sharpens the forecast in a direction the
process does not support. It corroborates the project's own headline finding, arrived at from the
opposite end: once the weekly cycle is removed there is essentially no day-to-day memory here. The
incoherent version scored better by accident. It is still going.

*(One real bug did surface inside the fix, and it is worth recording because it is subtle: the
first version filtered with a Poisson likelihood while `negbin-hmm` **samples** from a Negative
Binomial. A 60-post day against rates of 5/15/45 is decisive under Poisson and merely suggestive
under an overdispersed emission, so the belief was far more confident than the model was entitled
to be. `_emission_logpmf` is now overridden alongside `_emit`, so the two cannot drift apart.)*

## 5. `zip-glm` and `zinb-glm` were not zero-inflated

**Was:** two models named after a mechanism they did not implement.

`ZeroInflatedPoissonModel.sample_days` asked statsmodels for `which="mean"`, which returns the
**marginal** mean (1−π)·μ, and drew a single Poisson at it. That is an ordinary Poisson at a
deflated rate: no structural zeros, and *narrower* than the plain Poisson this family exists to
widen. The leaderboard had been saying so for as long as it existed — `zip-glm` and `poisson-glm`
scored 20.67 and 20.67, with interval widths of 20.1 and 20.1 — and the matching digits read as a
finding about zero-inflation rather than as the two models being the same code.

`ZeroInflatedNegBinModel` was worse: it fitted the zero-inflated model, read `alpha` off it, threw
the entire result away and refitted a plain Negative Binomial GLM. `zinb-glm` was `negbin-glm` under
another name, at 18.12 against 18.10.

**Fixed:** both sample the mechanism they claim — a Bernoulli switch on the count component, drawn
from `which="mean-main"` and `which="prob-main"`. On *this* data the correction is small (two zero
days in 558, so π is near zero and the fitted models are nearly Poisson and nearly NB anyway), which
is exactly why it survived: the bug was invisible precisely where the models were pre-registered as
negative controls. On a series with real off-days it would have been the whole model.

## 6. Hawkes was fitted on one population and scored against another — and had no weekly cycle

**Was:** two independent problems in the model that sat 16th of 20 with the worst calibration on the
board.

`refresh_backtest` passed `events=df[df["is_retruth"] == 0]` — originals only — while the target it
was scored against counts ReTruths. The branching ratio and the background rate were estimates of a
process that was not the one being forecast. The live path already used `convention_events`, which
is the worse way round: the model was *graded* under one definition and *deployed* under another.

And the simulation drew its immigrants at a flat rate across the week. Every other model in the zoo
knows Saturday is not Tuesday; on a series whose weekday means run from 14 to 24 posts, this one did
not.

**Fixed:** the backtest feeds it `convention_events`, and the immigrant rate is modulated by the
fitted weekday factor, normalised to preserve the process's overall level. `hawkes` goes **19.27 →
18.71** CRPS and its 80% coverage **0.533 → 0.599** — the largest single-model improvement in this
pass, and it moves the model from "broken" to merely "beaten by climatology".

## 7. The number on the front page was the one number nobody scored

**Was:** the tracker does not show a model. It shows a *rule* — take the top three of the
leaderboard and average them quantile-wise — and that composite appeared on no row of the board it
was built from. Its members' scores are not its score: a combination can be worse than its best
member, and nothing here could have told you.

**Fixed:** the rule is replayed week by week and graded like everything else, under the name
`headline-top3`, with its ranking recomputed at each week from **weeks that had already closed** —
so it never selects using the week it is about to be scored on. Three alternatives it had never been
measured against are scored alongside it, named in advance in `HEADLINE_RULES` for the same reason
the model list is pre-registered.

    rule                  CRPS    skill    cov80
    headline-top5        20.53   +0.9%     0.696
    headline-top3-pool   20.62   +0.4%     0.696
    headline-top3        20.63   +0.4%     0.696   <- deployed
    headline-top1        20.64   +0.4%     0.739
    ------------------------------------------------
    best single model    20.13             (hier-dow-eb, chosen with hindsight)

Two things fall out, and the second is the most important number this review produced.

**The choice of rule does not matter.** Four rules spanning "trust one model" to "blend five" sit
inside 0.11 CRPS of each other over 40 weeks. There was never anything to tune here.

**The cost of choosing does.** All four lose about **0.5 CRPS — 2.5%** — to the model that turns
out best on those weeks. That gap is the price of not knowing in advance which model will win, and
it is paid by any procedure that has to pick. Set it beside what the picking is *for*: the leading
model's skill over climatology is **2.3%**. The selection premium is larger than the entire prize.

Stated plainly: on this problem, choosing a model costs more than having a good one is worth.
Everything on the models page — twenty entries, four scoring rules, a calibration test — is
arbitrating differences smaller than the cost of the arbitration. That is not an argument for
deleting the leaderboard; it is an argument for reading it as the project's own prose already says,
as groupings rather than an ordering, and for the front page to lead with a number it can defend
rather than a winner it cannot.

## 8. The projection jumped at midnight, and stepped on every hour

**Reported from watching the live page**, which is how this class of bug gets
found — the projection moved when the date rolled over, and nothing had happened.

Time is continuous and a calendar day is a modelling convenience. The forecast
was treating the boundary as real, in three separate ways.

**It changed hands at midnight.** Up to 23:59, tomorrow was drawn by a *model*, as
one of the week's remaining days. At 00:00 that same calendar day stopped being
the models' problem and became the partial-day estimator's. Two different
objects, swapped at the stroke of a clock, disagreeing by about two posts on
the day and considerably more on the week. Measured on 13 June: the model's own
view of the day was a mean of **16.59** posts; the estimator, one minute into
that same day with nothing yet posted, said **14.66**.

**It was a step function within the day.** Conditioning ran on `hour <= now.hour`,
so 00:01 and 00:59 returned *identical* answers and 01:01 jumped by 1.5 posts.
And today's partial hour was being compared against history's complete one — at
14:01 today has had one minute of hour 14 and was matched against days that had
had sixty, so the estimator read "quieter than usual" at every :01 and recovered
by :59. A sawtooth, twenty-four times a day.

**Early in the day the conditioning was degenerate.** With nothing yet posted,
408 of 509 historical days tied at distance zero and `np.argsort` kept an
arbitrary 60 of them. The answer depended on numpy's tie-breaking.

**Fixed, and the fix is structural rather than a smoothing.** The conditioning
now runs on elapsed *seconds* into the local day, on both sides of the
comparison, and weights fall off smoothly with distance instead of taking a hard
60-nearest cut. More importantly, `partial.py` no longer produces a day
distribution of its own. It produces the **map** — the monotone transform from a
day's unconditional distribution to its distribution given how far today has got
— and each model's own day is pushed through it. Two properties fall straight
out of the arithmetic rather than being tuned in:

- At elapsed time zero, nothing has happened on any day in the pool, so every
  weight is exactly one and the map is the **identity**. The day a model was
  drawing at 23:59 is the day it draws at 00:00.
- At the end of the day every matched day has nothing left, so the map collapses
  to zero regardless of how loosely the matching is done.

The second property is why the map runs from day *totals* to *remainders*
rather than totals to totals. The first version went through totals and
subtracted the posts already banked, which left the estimator predicting **3.1
more posts in the final minute of a finished day** — any looseness in the
matching came back as phantom posts. The rewritten map has no such term.

Measured end to end, on the headline number the page actually shows:

| as of | before | after |
|---|---|---|
| 23:59:59 | 110.7 | 110.7 |
| 00:00:00 | **122.7** | **111.0** |
| 00:00:01 | 122.7 | 110.7 |

A **+12.0** step becomes a ±0.3 wobble, which is Monte-Carlo noise on 8,000
samples rather than a discontinuity. Through the rest of the day the projection
now drifts with the evidence — down while the day stays quiet, up by 1.8 when
seven posts land at 14:00 — and closes at exactly the realised total.

**One of those twelve posts was not the estimator's fault, and finding out why
fixed a second bug.** `week_progress` counts whatever is *stored* under today's
date, while the conditioning counts posts whose own timestamp is at or before
`now`. Live those agree, because nothing later exists yet. They disagree the
moment a post carries a timestamp slightly ahead of the clock — source skew,
which this pipeline reads from a third-party mirror and cannot rule out — and
then the day is banked once and forecast again. Both now use one definition,
which additionally makes the forecast a reproducible function of `now`. That is
what allowed the table above to be measured at all.

**And reconciling them introduced a third bug, caught by running it.** Writing
"today's count so far" onto the end of the observed array assumes the last row
of that array is today. It is not, on a day that has yet to post: the daily
series is built from the days that *have* posts, so a silent Saturday has no row
at all and the last element is Friday. The first version of this fix therefore
overwrote a completed day with today's zero — Friday's seven posts vanished, and
the published median came out at **155.4 against 158 already banked**. A week
total below the posts already in it is the one kind of number this project must
never print, and it printed one. The observed array is now reindexed across the
whole elapsed part of the week so every position has a row, and
`test_the_projection_can_never_fall_below_what_has_already_happened` asserts the
invariant directly rather than trusting the indexing.

Worth stating plainly, since this review is otherwise a list of other people's
mistakes: that one was mine, it was live for one pipeline run, and the only
reason it did not ship is that the export was read afterwards instead of assumed
correct. Every fix in this file went out with a test for the same reason.

## 8b. A week was "complete" the moment its last day existed, not when it ended

**Found by reading the deployed site**, one run after the merge — the prospective
record announced that it had scored a week, and no logged week had closed.

`complete_weeks` kept any week whose final day appeared in the series, and its
own docstring said it existed to stop "a partial final week reading as a sudden
collapse in volume". Those two things come apart on exactly one day in seven:
the week's own last day, while it is still running. The daily series is
zero-filled up to the last day carrying a post, so on a Saturday with any
activity at all the seventh row exists — holding a *partial* count — and the
week looked finished. `walkforward._week_starts` had the identical rule, phrased
as `len(week) == 7`.

Two consequences, and the second is worse than the first:

- The **backtest** scored its most recent week against a total that was still
  accumulating. One week in 52, and always the newest one — which is the week
  the live headline sits closest to.
- The **prospective record** graded a forecast against an answer that had not
  finished arriving. That record's only value is that it cannot be doubted, and
  a scored-too-early week is precisely the kind of thing that would make it
  doubtable. It reported one scored week; the correct answer was zero.

Locally this was invisible: the development archive ended the day before, so the
in-flight week had no seventh row and the bug had nothing to bite on. It needed
a deployment polling a live source on a Saturday afternoon to appear at all.

**Fixed:** a week is complete when it has seven rows **and** has ended. Both call
sites take an injectable `now` so the rule is testable rather than dependent on
when the suite happens to run.

## 9. Every poll resurrected the deleted posts

**Was:** `upsert_posts` refreshed `is_deleted = excluded.is_deleted` from whichever source wrote
last. The listing renders a deleted badge and reports it; the RSS feed carries no deletion
information at all, so `parse_rss` reports every post as not-deleted — and the feed is re-read on
**every poll**.

So any post in the 100 newest that the listing had marked deleted was un-deleted within minutes, and
stayed that way until a listing pass happened to see it again. The default convention excludes
deleted posts, so this quietly re-inflated the most recent days: the days the live forecast is most
sensitive to, and the ones no reader can check against a stable record.

**Fixed:** only a listing-sourced row may set the flag, in either direction. `time_precision` is
what tells the two sources apart ('minute' is the listing, 'second' is the feed).

## 10. Today's forecast was drawn from four years of a different process

**Was:** `partial.sample_rest_of_today` matches today against past days that looked like it at this
hour and resamples what those days did with their remaining time. It was handed the whole archive —
February 2022 onward — while every model beside it is fitted from the inauguration. A quiet Friday
morning in the second term was being matched against days from the campaign, and their remainders
were what the site printed as "more likely today".

It also built its day pool with `groupby("local_date")`, which silently drops every day that
produced nothing — the exact days that tell the estimator a quiet day is possible.

**Fixed:** the same `WINDOW.start` as everything else, and a zero-filled calendar.

## 11. The site printed certainties

**Was:** `threshold_probabilities` inverted each model's quantile function over the nine published
levels with `np.interp(..., left=0.0, right=1.0)`. Any threshold past the 97.5th percentile came
back as an exact **0%**, and anything under the 2.5th as an exact **100%** — and the shipped
`forecast.json` had two thresholds at `"probability": 1.0`. A weekly post count is unbounded above.
There is no threshold it clears with probability one.

It was also a different distribution from the one printed above it: the headline averaged
*quantiles* (Vincentization) and the threshold table averaged *probabilities* (a linear pool). The
first review noted this and left it. So "P(≥ 140)" was not the probability implied by the interval
directly above it.

**Fixed:** one combination, in `combine.py`, used by both. It runs on a 999-point grid rather than
on the nine published levels, and past the ends of that grid the site now prints `<0.1%` or `>99.9%`
— an inequality, which is what is actually known.

## 12. The leaderboard reshuffled itself between identical runs

**Was:** every model drew from the global numpy RNG, unseeded. Two backtests over the same data
produced different boards.

That is not a tidiness complaint. The top twelve models span about 0.5 CRPS over 52 weeks; the
Monte-Carlo spread of 4,000 samples is of the same order. So an unseeded board reorders its own top
half between runs, and the ordering — which the site displays, and which the live headline selects
from — is partly a record of which random numbers came up.

**Fixed:** each forecast draws from a stream seeded on `(model, week, cut)`. Keyed on what the
forecast *is* rather than on when it was made, so a model failing to fit no longer shifts every
later draw, and adding or removing a model leaves every other model's numbers untouched — which is
what makes a before/after comparison of a fix mean anything. It is also what let this review measure
its own changes at all: **every delta quoted above is a comparison between two reproducible runs.**

Re-running under a different seed is now the way to measure the noise rather than inherit it.

---

## Smaller, and fixed

- **The ReTruth share double-counted its own denominator.** `retruths / (modelled_series +
  retruths)`, where the modelled series already contains the ReTruths under the current convention.
  The note attached to it still said they were excluded from the series, which stopped being true
  one review ago; so did a line in the site footer and the `Post` docstring in `parse.py`.
- **The recent-posts feed sorted on the wrong timestamp.** `created_utc` for a ReTruth is the
  *original author's* time, so a reshare of a month-old post appeared a month down the "Latest
  posts" list, which is a feed of what just happened. Now ordered by `effective_utc`, with the
  mirror's arrival id as the tie-break.
- **The models page mislabelled every column of its by-cut table.** It hardcoded `Mon..Sun` for cuts
  0–6: wrong day names under a Sunday-anchored week, and no column at all for cut −1 — so the
  hardest forecast on the board, the one on display for a seventh of every week, was invisible. The
  labels now come from the export.
- **Notifications dated a ReTruth by its original.** An alert firing now could carry a timestamp
  from two months ago with nothing to say which it was.
- **The GBM built its features from a stale copy of history.** It refits every fourth week, and
  `_anchor_row` used the series captured at fit time — so lag-1 through lag-14 could be three weeks
  old. Refitting is what the interval saves; the input row costs nothing to rebuild. Together with
  the week-anchor fix and a gapless calendar reindex, `lgbm-quantile` goes **21.82 → 21.07**.

## Added

- **A prospective record** (`truthforecast/live.py`). Every forecast the pipeline publishes is
  appended to `data/live_forecasts.jsonl` the moment it is built, and scored once the week closes.
  One line per (week, cut), and it is the **first** forecast made at that cut that stands — a
  register you may keep amending until the answer arrives is not a register. This is the first
  number here that is not retrospective: however careful the embargoes, every choice in this
  project was made with the whole series visible, so the backtest is an upper bound and cannot be
  anything else. The record starts at zero weeks and the site says so rather than implying
  otherwise.
- **Skill against climatology at each cut** (`skill_by_cut`). A model's average CRPS is mostly a
  statement about the calendar — 28.0 at the week's opening against 7.6 on its last full day — and
  dividing by the reference model at the *same* cut is what isolates the model. The leading model's
  skill turns out to be remarkably flat across the week: 2.2% with nothing observed, 2.9% mid-week,
  1.3% on Friday.
- **Coverage by cut**, which the first review added, now has something to say. The leading model's
  headline 0.80 coverage is an *average of 0.71 and 0.89*: overconfident at the start of the week,
  too timid by Thursday. It is not calibrated at any cut; it is miscalibrated in two directions that
  happen to cancel.

## Still true, still unfixed

The first review's "left alone deliberately" list stands, with one item now more urgent:

**The backtest still runs on revised data.** Every score is measured against a history no live
forecaster had. `first_seen_utc` is stored and still unused. The prospective record added here is
the honest long-run answer, but it needs a year; a vintage backtest is the answer available today,
and the reason it was not built in this pass is worth stating plainly rather than filing as
laziness: the archive was re-walked from scratch while doing this work, so every row's
`first_seen_utc` is now the same afternoon. The information required to run it no longer exists in
this database and will only accumulate from here. That is itself a finding about `first_seen_utc` —
it is destroyed by exactly the operation the project performs whenever the parser changes.
