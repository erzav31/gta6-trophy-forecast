"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at the case's target, from the house's
reads of the trophy page up to the case's checkpoint. Python 3.13, standard
library only, no network, deterministic.

Model
-----
A trophy's earn rate is a ratio, so it is forecast as one: rate = earners /
players tracked. Both sides are fitted separately and recombined, which is
what lets the forecast fall when the tracked-player base grows faster than
the trophy's earners (dilution at a refresh) and rise when earners grow
faster (players progressing through the game).

- Time is hours since the game's release, x = ln(1 + t). Over the reads of
  the last 24 hours before the checkpoint (all reads when fewer), least
  squares fits ln(players tracked) on x, and, per trophy, ln(earners + 0.5)
  on x, where earners = rate * players / 100. A fit uses only the reads
  where the trophy really had a rate (a null is never fitted as 0.01). A power law in (1 + t) is the
  middle ground between exponential growth (which explodes over a 4.5-day
  horizon) and a flat line: launch adoption grows fast, then decelerates.
- Per-trophy slopes are shrunk toward the cross-trophy median slope with an
  empirical-Bayes weight from each fit's standard error, so a trophy with
  only a couple of distinct counts does not get a wild slope, while a trophy
  whose earners are genuinely flat keeps a slope near zero and is forecast
  to dilute. When the last 24 hours show no movement at all, the fit falls
  back to the whole history, so a page that froze recently still uses the
  trend it had before freezing.
- Earners are cumulative, so the forecast is clamped to at least the last
  read's earners, and at most the forecast players tracked; the median rate
  is 100 * earners / players, clamped to [0.01, 99.99].
- Spread, on the logit scale, combines a random-walk term
  (0.5 * sqrt(horizon in days), the house baseline's), the fit's residual
  spread widened by its leverage at the target, and a staleness term that
  grows with the hours since the page last changed. A trophy near 100% is
  damped: its earners cannot exceed the players tracked, so its logit
  spread shrinks as the median approaches 100.

Refresh mixture
---------------
Exophase refreshes a game's counts on its own schedule, hours apart, so the
page a case's history shows can be stale: every read identical since launch
night means no refresh has ever been observed, and a page that has not
changed in the last 24 hours is stale too. In that regime the smooth fit
above says "nothing moves", which is exactly wrong when a refresh lands
inside the horizon: the tracked base jumps and most rates fall. So when the
page has been unchanged for the last 24 hours, the forecast is a
two-component mixture over the logit scale:

- no refresh before the target, probability 1 - P: the smooth forecast;
- a refresh lands before the target, probability P: every rate is
  multiplied by the refresh dilution factor delta.

P comes from a renewal view of the refresh gap: gaps are taken as
Normal(mean, sd), the mean and sd estimated from the gaps between the
change points the history shows (prior mean 4.5h, sd 1.25h when none has
been observed), conditioned on having already waited out the staleness
since the last change. Delta is the median rate ratio across the refreshes
the history shows (last five changes; the trophy at 100% is left out,
every tracked player holds it, so its factor is 1). With no refresh ever
observed, delta is the launch-night prior 0.20: the factor that jointly
reproduces the house's published practice figures for the persistence
baseline, loss about 1.51 and typical miss about 0.68 points; the rules
publish the refresh's aggregates, 638 -> 3,868 players tracked and
943 -> 4,276 trophies earned.

The mixture's quantiles are the honest predictive quantiles of that
two-outcome view, found by bisecting the mixture CDF. Once the history
shows a refresh inside the last 24 hours, the smooth fit carries the
growth and the mixture is not used.

Output: /output/<id>.out per case, {"trophies": [{"id": n, "q": [...]}]},
five quantiles at levels 0.05, 0.25, 0.5, 0.75, 0.95, rounded to 4 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
SIGMA_RW = 0.5  # logit spread per sqrt(day), the house baseline's random walk
FIT_WINDOW_HOURS = 24.0
# Refresh prior, used only while the page has been unchanged for the last
# 24 hours (see docstring). REFRESH_DELTA is the prior dilution factor when
# no refresh has ever been observed; otherwise it is estimated from the
# history's own refreshes.
REFRESH_GAP_MEAN_HOURS = 4.5
REFRESH_GAP_SD_HOURS = 1.25
REFRESH_DELTA = 0.20  # rate multiplier at a refresh, except the 100% trophy
REFRESH_JUMP_SIGMA = 0.35  # logit spread of the refresh component beyond the smooth one
SLOPE_PRIOR_SD = 0.5  # prior sd of a trophy's log-log slope, for shrinkage
STALE_SIGMA = 0.25  # logit spread added when the page has been unchanged for 48h


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def refresh_prob(stale_h, horizon_h, gap_mean, gap_sd):
    """P(the next refresh lands within the horizon | none for stale_h).

    Refresh gaps are taken as Normal(gap_mean, gap_sd); the probability is
    the gap's conditional CDF over the horizon given it already exceeds the
    staleness. A page long overdue for a refresh gets a probability near 1.
    """
    z0 = (stale_h - gap_mean) / gap_sd
    z1 = (stale_h + horizon_h - gap_mean) / gap_sd
    surv = 1.0 - norm_cdf(z0)
    if surv <= 1e-9:
        return 1.0
    return min(1.0, max(0.0, (norm_cdf(z1) - norm_cdf(z0)) / surv))


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, resid_sd, se_b, xbar, sxx, n)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return ybar, 0.0, 0.0, 0.0, xbar, sxx, n
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = n - 2
    if dof <= 0:
        return a, b, 0.5, 0.5, xbar, sxx, n
    var = sum(r * r for r in resid) / dof
    resid_sd = math.sqrt(var)
    se_b = resid_sd / math.sqrt(sxx) if sxx > 0 else 0.0
    return a, b, resid_sd, se_b, xbar, sxx, n


def mixture_quantile(mu0, s0, mu1, s1, p, tau):
    """tau-quantile of (1-p) * N(mu0, s0) + p * N(mu1, s1), by bisection."""
    lo = min(mu0 - 8 * s0, mu1 - 8 * s1)
    hi = max(mu0 + 8 * s0, mu1 + 8 * s1)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = (1.0 - p) * norm_cdf((mid - mu0) / s0) + p * norm_cdf((mid - mu1) / s1)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    horizon_h = case["horizon_minutes"] / 60.0

    if not history:
        return {"trophies": [{"id": t["id"], "q": [0.2, 0.5, 1.0, 2.0, 5.0]} for t in trophies]}

    try:
        release = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    except (KeyError, TypeError, ValueError):
        release = parse_time(history[0]["at"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    t_target = (target - release).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target, 0.0))

    # Series: hours since release, players tracked, per-trophy rates (nulls filled
    # forward from the last read that had the trophy, else 0.01). `raw` keeps
    # the rates as read, so a fit never learns from a filled-in value: a
    # trophy that only just appeared (nulls before) gets a flat fit at its
    # last earners instead of an explosive slope off the 0.01 fill.
    times, players, rates, raw = [], [], [], []
    last_rate = [0.01] * len(trophies)
    for read in history:
        times.append((parse_time(read["at"]) - release).total_seconds() / 3600.0)
        players.append(read["players_tracked"])
        raw.append(read["rates"])
        row = []
        for i, r in enumerate(read["rates"]):
            if r is not None:
                last_rate[i] = r
            row.append(last_rate[i])
        rates.append(row)

    # Change points: a read that differs from the one before it, in players or
    # in any rate. The last change sets the page's staleness.
    changed = [False] * len(history)
    for k in range(1, len(history)):
        if players[k] != players[k - 1] or rates[k] != rates[k - 1]:
            changed[k] = True
    n_changes = sum(changed)
    last_change_at = None
    for k in range(len(history) - 1, -1, -1):
        if changed[k]:
            last_change_at = parse_time(history[k]["at"])
            break
    if last_change_at is None:
        last_change_at = parse_time(history[0]["at"])
    stale_h = max(0.0, (checkpoint - last_change_at).total_seconds() / 3600.0)

    # Fit window: last 24h of reads; fall back to all reads when that window
    # has fewer than 3 reads or shows no movement at all.
    window_start = (checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0)
    idx = [k for k in range(len(history)) if parse_time(history[k]["at"]).timestamp() >= window_start]
    if len(idx) < 3:
        idx = list(range(len(history)))
    else:
        win_players = {players[k] for k in idx}
        win_rates = {tuple(rates[k]) for k in idx}
        if len(win_players) < 2 and len(win_rates) < 2:
            idx = list(range(len(history)))

    xs = [math.log1p(max(times[k], 0.0)) for k in idx]
    a_n, b_n, _, _, xbar, sxx, _ = ols(xs, [math.log(max(players[k], 1)) for k in idx])

    n_last = players[-1]
    n_hat = math.exp(a_n + b_n * x_target)
    n_hat = max(n_hat, float(n_last))

    # Per-trophy log-earners fits, slopes shrunk to the cross-trophy median.
    # Each fit uses only the reads where the trophy really had a rate; a
    # trophy with fewer than two such reads in the window is held flat at
    # its last earners (its slope error stays wide, so shrinkage still
    # pulls it toward the median rather than letting it run wild).
    slopes, ses, fits = [], [], []
    for i in range(len(trophies)):
        pts = [
            (xs[j], math.log(max(rates[idx[j]][i] * players[idx[j]] / 100.0, 0.0) + 0.5))
            for j in range(len(idx))
            if raw[idx[j]][i] is not None
        ]
        if len(pts) >= 2:
            a, b, sd, se, xb, sx, n = ols([p[0] for p in pts], [p[1] for p in pts])
        else:
            e0 = rates[-1][i] * n_last / 100.0
            a, b, sd, se, xb, sx, n = math.log(e0 + 0.5), 0.0, 0.5, 0.5, xs[-1], 0.0, len(pts)
        slopes.append(b)
        ses.append(se)
        fits.append((a, b, sd, xb, sx, n))
    ordered = sorted(slopes)
    b_med = ordered[len(ordered) // 2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])

    sigma_rw = SIGMA_RW * math.sqrt(max(horizon_h, 0.0) / 24.0)
    sigma_stale = STALE_SIGMA * math.sqrt(min(stale_h, 48.0) / 48.0)

    # Refresh mixture inputs: gap statistics and per-refresh dilution from
    # the change points the history shows, or the launch-night prior.
    change_times = [parse_time(history[k]["at"]) for k in range(len(history)) if changed[k]]
    gaps = [
        (change_times[j + 1] - change_times[j]).total_seconds() / 3600.0
        for j in range(len(change_times) - 1)
    ]
    if len(gaps) >= 2:
        gap_mean = sum(gaps) / len(gaps)
        gap_sd = max(0.5, math.sqrt(sum((g - gap_mean) ** 2 for g in gaps) / (len(gaps) - 1)))
    elif len(gaps) == 1:
        gap_mean, gap_sd = gaps[0], REFRESH_GAP_SD_HOURS
    else:
        gap_mean, gap_sd = REFRESH_GAP_MEAN_HOURS, REFRESH_GAP_SD_HOURS
    per_change_delta = []
    for k in range(1, len(history)):
        if not changed[k]:
            continue
        ratios = [
            rates[k][i] / rates[k - 1][i]
            for i in range(len(trophies))
            if rates[k - 1][i] > 0 and rates[k - 1][i] < 99.99 and rates[k][i] < 99.99
        ]
        if ratios:
            per_change_delta.append(median(ratios))
    delta = median(per_change_delta[-5:]) if per_change_delta else REFRESH_DELTA
    delta = min(max(delta, 0.05), 1.0)
    # The page is stale when nothing changed in the last 24 hours; only then
    # does the mixture apply (a fresh page's trend already carries refreshes).
    window_start_dt = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0
    tail_flat = not any(
        changed[k] for k in range(1, len(history))
        if parse_time(history[k]["at"]).timestamp() >= window_start_dt
    )
    p_refresh = refresh_prob(stale_h, horizon_h, gap_mean, gap_sd) if tail_flat else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        a, b, sd, xb, sx, n = fits[i]
        se = ses[i]
        if se > 1e-9:
            w = (1.0 / (se * se)) / (1.0 / (se * se) + 1.0 / (SLOPE_PRIOR_SD ** 2))
            b_star = w * b + (1.0 - w) * b_med
        else:
            b_star = b
        e_last = rates[-1][i] * n_last / 100.0
        e_hat = math.exp(a + b_star * x_target)
        e_hat = min(max(e_hat, e_last), n_hat)
        med = clamp(100.0 * e_hat / n_hat)

        lev = 1.0 + 1.0 / n + ((x_target - xb) ** 2) / sx if sx > 0 else 1.0
        sigma_fit = sd * math.sqrt(max(lev, 1.0))
        sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_stale ** 2)
        sigma = min(max(sigma, 0.12), 2.5)
        # Near 100% a rate cannot move up, only slightly down: damp the spread.
        sigma *= (1.0 - med / 100.0) ** 0.25 if med > 50.0 else 1.0
        sigma = max(sigma, 0.05)

        mu = logit(med)
        if p_refresh > 0.0 and rates[-1][i] < 99.99:
            mu1 = logit(med * delta)
            s1 = math.sqrt(sigma ** 2 + REFRESH_JUMP_SIGMA ** 2)
            qs = [inv_logit(mixture_quantile(mu, sigma, mu1, s1, p_refresh, tau)) for tau in LEVELS]
        elif p_refresh > 0.0:
            # The 100% trophy: every tracked player holds it, so a refresh
            # leaves it at 100%; the mixture collapses to one component.
            qs = [inv_logit(mu + z * sigma) for z in Z]
        else:
            qs = [inv_logit(mu + z * sigma) for z in Z]
        qs = [round4(min(max(q, 0.01), 100.0)) for q in qs]
        # Rounding must not break the non-decreasing order.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
