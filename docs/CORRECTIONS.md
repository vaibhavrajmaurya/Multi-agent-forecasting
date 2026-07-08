# Corrections & Debugging Log — Multi-Agent Financial Forecasting

> This file documents every correction made to the project **after the original
> 7-day build**, in the order they were discovered and fixed. The original build
> (`PROGRESS.md`) was written but never executed — Python wasn't installed. The
> moment the code actually ran, a chain of bugs surfaced: first environment/compat
> errors, then deep methodological flaws in the centerpiece algorithm itself.
>
> **Audience:** mentors and students who want to understand not just *what* the
> final code does, but *why every line is the way it is* — most of these were real,
> instructive mistakes.

**Repo:** https://github.com/tezzuk/Mutli-Agent-Forecasting
**Build state at start of this log:** all files written (Days 1–7 complete), zero lines ever executed.

---

## Summary table

| # | Correction | Type | Commit |
|---|---|---|---|
| 1 | `CalendarFourier(freq="Y")` → `"YE"` | Library compat | `1a57140` |
| 2 | `out_of_sample(index=...)` → `forecast_index=...` | Library compat | `1a57140`, `376cf37` |
| 3 | `plotly` missing from `requirements.txt` | Deploy / deps | `1caa9dd` |
| 4 | Dashboard unreadable (dark UI + white charts) | UI / contrast | `febf2d4` |
| 5 | **Hedge η 10,000× too small → weights never move** | **Algorithm** | `c803415` |
| 6 | **LSTM zero-pads ~48% of test days** | **Methodology** | `c803415` |
| 7 | LSTM trained only 5 epochs in backtest | Training | `c803415` |
| 8 | Streamlit serves stale CSVs after regeneration | Caching | `4eea3ae` |
| 9 | **Hedge collapses onto one agent (no adaptivity)** | **Algorithm** | `d78bc07` |
| 10 | **Hedge loss (MSE) misaligned with trading P&L** | **Methodology** | `31ebdf0` |
| 10b | Directional loss tested → didn't hold → reverted to MSE | Methodology (negative result) | revert |
| 11 | No fixed seed → results not reproducible (incl. GPU/cuDNN) | Reproducibility | seed |
| 12 | Conviction threshold tested → hurt performance | Methodology (negative result) | `85e7aca` |

Bold = substantive findings worth talking about in an interview. The rest are
ordinary "make it run / make it deploy" fixes.

---

## Phase A — Getting it to run at all (environment & compatibility)

### Correction 1 — `CalendarFourier` frequency string deprecated
**File:** `pipeline/agents.py` (TrendAgent) · **Commit:** `1a57140`

**Symptom (first backtest run, in Colab):**
```
FutureWarning: 'Y' is deprecated and will be removed in a future version, please use 'YE' instead.
```
then a hard crash downstream.

**Root cause:** The code was written against an older `statsmodels`. Pandas/statsmodels
renamed the year-end offset alias from `"Y"` to `"YE"`. Colab ships statsmodels 0.14+,
where `"Y"` is deprecated.

**Fix:**
```python
# before
fourier = CalendarFourier(freq="Y", order=self.fourier_order)
# after
fourier = CalendarFourier(freq="YE", order=self.fourier_order)
```

---

### Correction 2 — `DeterministicProcess.out_of_sample()` keyword changed
**File:** `pipeline/agents.py` (TrendAgent.predict) · **Commits:** `1a57140`, `376cf37`

**Symptom (two successive crashes):**
```
TypeError: DeterministicProcess.out_of_sample() got an unexpected keyword argument 'index'
```
then, after the first attempt to fix it:
```
TypeError: CalendarFourier terms can only be computed from DatetimeIndex and PeriodIndex
```

**Root cause:** In statsmodels 0.14+, `out_of_sample()` no longer accepts `index=`.
The first patch removed the kwarg entirely, but then statsmodels couldn't infer the
future dates for the Fourier seasonal terms (it needs the actual calendar dates of the
test period to compute annual Fourier features), producing the second error.

**Fix (final form):** pass the test index via the correct kwarg, `forecast_index`:
```python
# original (broke):   X = self._dp.out_of_sample(steps=len(test_df), index=test_df.index)
# attempt 1 (broke):  X = self._dp.out_of_sample(steps=len(test_df))
# final (works):
X = self._dp.out_of_sample(steps=len(test_df), forecast_index=test_df.index)
```

**Lesson:** the TrendAgent's seasonality is calendar-based, so the model genuinely needs
to know *which* future dates it's forecasting — dropping the index wasn't an option.

---

### Correction 3 — `plotly` missing from `requirements.txt`
**File:** `requirements.txt` · **Commit:** `1caa9dd`

**Symptom (on Streamlit Community Cloud):**
```
ModuleNotFoundError ... File "app.py", line 9, in <module>
    import plotly.express as px
```

**Root cause:** `app.py` imports `plotly.express` and `plotly.graph_objects`, but
`plotly` was never listed in `requirements.txt`. It worked in Colab only because Colab
pre-installs plotly; Streamlit Cloud builds a clean environment from the requirements
file and had no plotly.

**Fix:** added `plotly>=5.20.0` to `requirements.txt`.

**Lesson:** "works on my machine / in Colab" ≠ "works on a clean deploy." The
requirements file must list *every* third-party import, not just the ones you remember.

---

## Phase B — Making the dashboard usable

### Correction 4 — Dashboard contrast / theme
**File:** `.streamlit/config.toml` (new) · **Commit:** `febf2d4`

**Symptom:** Streamlit Cloud rendered the app in dark mode, but every Plotly chart had a
hard-coded `plot_bgcolor="white"`. The result: white chart boxes floating on a near-black
page, axis labels barely legible.

**Root cause:** mismatch between the Streamlit page theme (auto/dark) and the charts'
fixed white backgrounds.

**Fix:** pin a light theme that matches the project palette so page and charts blend:
```toml
[theme]
base = "light"
primaryColor = "#4361ee"
backgroundColor = "#f8f9ff"
secondaryBackgroundColor = "#eef0ff"
textColor = "#1a1a2e"
font = "sans serif"
```

---

### Correction 8 — Streamlit served stale results after every regeneration
**File:** `app.py` (`load_results`, `load_metrics`) · **Commit:** `4eea3ae`
*(out of numeric order — discovered during Phase C testing)*

**Symptom:** After pushing freshly regenerated `results/*.csv`, the deployed dashboard
kept showing the **old** numbers, even though the new sidebar text proved the new code
had deployed.

**Root cause:** `@st.cache_data` memoizes a function by its **arguments**, not by the
contents of files it reads. `load_results()` took no arguments, so its cache key never
changed — Streamlit kept returning the first CSV it ever loaded.

**Fix:** make the file's modification time part of the cache key, so a regenerated CSV
(new mtime) automatically busts the cache:
```python
def _mtime(path): return path.stat().st_mtime if path.exists() else 0.0

@st.cache_data
def load_results(_mtime_key: float):   # key changes when the file changes
    ...
results = load_results(_mtime(RESULTS_PATH))
```

**Lesson:** caching a pure function of *arguments* is wrong when the function secretly
reads external state (a file). Either include that state in the key or disable caching.

---

## Phase C — The deep algorithmic corrections (the important ones)

These are the corrections that actually matter. The original pipeline *ran* after Phase A,
but the results were quietly meaningless: the Hedge algorithm — the entire point of the
project — was doing nothing, and the LSTM's metrics were artifacts.

### Correction 5 — Hedge η was ~10,000× too small; weights never adapted
**File:** `pipeline/aggregator.py` (`update`) · **Commit:** `c803415`

**Symptom:** The "Agent Weights Over Time" chart was four perfectly flat horizontal
bands for 12 years. Final weights were 25.0% / 24.9% / 25.0% / 25.0% — essentially the
uniform initialization. **`HedgeAggregator` and `EqualWeightAggregator` produced
identical metrics to three decimals** — the dead giveaway that the adaptive algorithm
was a no-op.

**Root cause (the math):** the update is `w_i ← w_i · exp(−η · loss_i)` with
`loss_i = (pred_i − actual)²`. On daily log returns:
- typical prediction ≈ 0.001, typical actual ≈ 0.009 → loss ≈ (0.008)² ≈ **6.4e-5**
- update multiplier = `exp(−0.1 · 6.4e-5)` ≈ **0.9999936**

The weights moved by less than one part in 100,000 per step. η = 0.1 is calibrated for
losses of order 1; squared log-return losses are of order 1e-4, so η needed to be ~10,000×
larger — or, better, the losses need to be normalized.

**Fix:** normalize each step's losses by their **mean across agents**, making the update
scale-invariant and depend only on *relative* agent performance:
```python
losses = (preds - actual) ** 2
scale  = losses.mean() + 1e-12
norm_losses = losses / scale          # ~O(1), unit-independent
self.weights *= np.exp(-self.eta * norm_losses)
self.weights /= self.weights.sum()
```

**Lesson:** the Hedge regret bound assumes losses in [0, 1]. If your real losses live on
a wildly different scale, the algorithm silently degenerates into equal-weighting. Always
normalize losses to a known range before a multiplicative-weights update.

---

### Correction 6 — LSTM (SequenceAgent) zero-padded ~48% of every test window
**File:** `pipeline/agents.py` (SequenceAgent `fit`/`predict`) · **Commit:** `c803415`

**Symptom:** SequenceAgent reported **28.5% directional accuracy** (far *worse* than a
coin flip) yet a **positive 0.620 Sharpe** — a contradiction that signals an artifact.

**Root cause:** the LSTM needs a 30-day lookback before it can predict, so `predict()`
filled the first `window_size = 30` rows of **every fold** with zeros:
```python
full_preds = np.zeros(len(test_df))
full_preds[self.window_size:] = preds   # first 30 rows are fake zeros
```
With 48 folds, that's ~48 × 30 ≈ **1,440 zero predictions out of 2,992 test days (~48%)**.
Two consequences:
1. **Tanked directional accuracy:** `sign(0)` never matches `sign(actual)`, so all ~1,440
   padded days counted as *wrong direction*, capping accuracy far below 50%.
2. **Fake Sharpe:** on padded days the strategy return is `sign(0)·actual = 0`. Halving
   the number of non-zero return days deflates the standard deviation, and
   `Sharpe = mean/std·√252` is inflated by the small denominator. The 0.620 was not alpha.

**Fix:** carry the **trailing 30 training-set returns** into prediction so the very first
test day already has a full history — no padding, and **no leakage** (that window is
strictly *before* the test period):
```python
# in fit():
self._train_tail = train_df["log_return"].values[-self.window_size:]

# in predict():
full = np.concatenate([self._train_tail, test_returns])   # prepend past context
scaled = self._scaler.transform(full.reshape(-1, 1)).flatten()
X, _ = self._make_sequences(scaled, self.window_size)     # yields exactly len(test) preds
```

**Result:** SequenceAgent's directional accuracy jumped to a realistic ~54% and its Sharpe
became an honest number instead of a variance-deflation artifact.

---

### Correction 7 — LSTM trained only 5 epochs inside the backtest
**File:** `pipeline/backtest.py` (`__main__`) · **Commit:** `c803415`

**Symptom:** contributed to the LSTM's reversed/garbage predictions.

**Root cause:** the backtest instantiated `SequenceAgent(epochs=5)` — a deliberate
speed shortcut from the build phase that was never raised. Five epochs is not enough for
an LSTM to learn anything stable, especially retraining from scratch each fold.

**Fix:** `SequenceAgent(epochs=25)` (early stopping with patience=5 still cuts it short
when it converges, so the cost is bounded).

---

### Correction 9 — Hedge collapsed onto a single agent (lost all adaptivity)
**File:** `pipeline/aggregator.py` (`update`, Fixed-Share) · **Commit:** `d78bc07`

**Symptom:** After Correction 5 the weights finally moved — but *too* hard. The chart
showed all weight collapsing onto **SequenceAgent within the first few weeks** and staying
pinned at ~100% for the rest of the 12 years. The "ensemble" had silently become a single
agent (Hedge Sharpe ≈ SequenceAgent Sharpe), which defeats the purpose of a *multi-agent,
regime-adaptive* system.

**Root cause:** vanilla Hedge is designed for *stationary* problems — it provably
converges onto the single best expert and never reconsiders. Once an agent's weight decays
to ~1e-10, the multiplicative update can never revive it, even if it becomes the best agent
in a later regime. Markets are non-stationary, so this is the wrong behavior here.

**Fix:** **Fixed-Share** (Herbster & Warmuth, 1998) — after each Hedge update, mix a small
uniform component back in so every weight stays ≥ α/N and any agent can recover:
```python
self.weights = (1 - self.alpha) * self.weights + self.alpha / self.n_agents
self.weights /= self.weights.sum()
```
with `alpha = 0.05`. This is the standard upgrade to Hedge for tracking a *shifting* best
expert.

**Result:** the weight chart became genuinely adaptive — all four agents stay alive and
"breathe," widening when their regime favors them (e.g., VolatilityAgent during COVID/2022)
and shrinking otherwise. Final weights became a real blend (~27/30/11/32) instead of
100/0/0/0.

**Trade-off surfaced:** keeping the bad agents alive means the ensemble always carries some
weight on coin-flippers, which dragged its Sharpe *down* (0.67 → 0.37) relative to the
degenerate single-agent collapse. That tension led directly to Correction 10.

---

### Correction 10 — Hedge weighted agents by MSE, which is misaligned with trading P&L
**File:** `pipeline/aggregator.py` (`update`, `loss_mode`) · **Commit:** `31ebdf0`

**Symptom:** Even with a working, adaptive Hedge, the ensemble couldn't match its own best
agent. The smoking gun in the final weights: **MomentumAgent received ~30% weight despite a
Sharpe of only 0.06** (a coin-flipper). The algorithm was rewarding a useless agent.

**Root cause (the deepest bug in the project):** the Hedge loss was squared prediction
error (MSE). But:
- MSE rewards predicting ≈0, which **every** agent does (daily-return magnitude is
  unpredictable), so MSE barely separates the agents; and
- MSE **completely ignores direction**, which is the only thing that drives P&L
  (`strategy = sign(pred) · actual`).

So an agent that predicts tiny safe values has low MSE and earns weight, even if its
*directional* calls are no better than a coin flip. **The online-learning objective was
misaligned with the financial objective.**

**Fix:** switch the loss to the agent's realized **negative P&L** for the day, normalized
per-step to [0, 1] (added as `loss_mode="directional"`, now the default; `"mse"` retained
for teaching contrast):
```python
if self.loss_mode == "directional":
    pnl = np.sign(preds) * actual         # profit if you follow agent i
    losses = -pnl                          # lower loss = more profit
    lo, hi = losses.min(), losses.max()
    norm_losses = (losses - lo) / (hi - lo + 1e-12)
else:  # "mse"
    losses = (preds - actual) ** 2
    norm_losses = losses / (losses.mean() + 1e-12)
```
Now agents are up-weighted for calling **direction** right (making money) and starved when
they don't — concentrating the ensemble on the agents that actually trade well.

**Honesty note for interviews:** choosing the directional loss *after* observing MSE
underperform on this data is a mild form of tuning on the test set. It's defensible because
P&L-aligned loss is the correct objective for a trading system *on first principles* (MSE
was simply the wrong default), but the principled framing — not "it scored better" — is the
honest one to give.

---

### Correction 10b — Directional loss was a tested hypothesis that did NOT hold; reverted to MSE
**File:** `pipeline/backtest.py` (default `loss_mode`) · **Commit:** `<reverted to mse>`

**What we expected:** weighting agents by realized P&L (Correction 10) should concentrate
the ensemble on agents that trade profitably and lift its Sharpe toward the best agent.

**What actually happened:** the directional run came in *lower* (Hedge Sharpe 0.192) than
the MSE run (0.368). Two reasons:
1. **Daily direction is near-random (~50–54%).** Weighting agents by *who called yesterday
   right* tracks recent **luck**, not skill — the skill gap between agents is tiny and the
   day-to-day outcome is mostly noise. MSE, though misaligned with P&L, at least produces a
   *stable* allocation.
2. **The comparison was confounded** (see Correction 11) — without a fixed seed the LSTM
   differed between runs (SequenceAgent 0.884 vs 0.698), so part of the gap was training
   noise, not the loss function.

**Decision:** revert the default to `loss_mode="mse"` (η=0.2). `"directional"` is kept in
the code as a documented alternative for teaching. This is a genuine negative result: the
"obvious" objective-alignment fix ran into a subtler problem — *the signal you'd align to is
itself too noisy at daily frequency.*

**Lesson:** aligning your learning objective with your real objective is correct in
principle, but only helps if the real objective carries a learnable signal. At daily
frequency it largely doesn't.

---

### Correction 11 — No fixed random seed → results not reproducible
**File:** `pipeline/agents.py` (SequenceAgent.fit) · **Commit:** `<seed>`

**Symptom:** Every backtest produced different numbers; SequenceAgent's Sharpe wandered
between ~0.70 and ~0.88 across identical runs. This made it impossible to compare design
changes (e.g. MSE vs directional loss) — any difference could be training noise.

**Root cause:** the LSTM (PyTorch weight init + training) used no fixed seed, so each
`fit()` produced a different network.

**Fix (first attempt — incomplete):**
```python
torch.manual_seed(0)
np.random.seed(0)
```
This made the LSTM reproducible on **CPU**, but on Colab's **GPU** the Sharpe still wandered
(SequenceAgent 0.884 one run, 0.728 the next). Reason: `torch.manual_seed` does not control
cuDNN, which selects non-deterministic RNN/LSTM algorithms by default.

**Fix (complete):**
```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Lesson:** reproducibility is a prerequisite for *any* honest A/B comparison — and on a GPU
that takes more than `manual_seed`. You must also seed CUDA and force cuDNN deterministic,
or the deep-learning agent silently re-randomizes every run.

---

### Correction 12 — Conviction threshold tested; it *hurt* (another negative result)
**File:** `pipeline/metrics.py`, `pipeline/backtest.py` · **Commit:** `85e7aca`

**Hypothesis:** trade only on the most-confident days (largest |prediction|) and sit in cash
otherwise — high-conviction days should carry more edge, lifting Sharpe.

**Result:** the opposite. `Hedge (Conviction)` (trade only the top-50% by |pred|) came in at
**Sharpe −0.133**, well below the always-trading `Hedge Ensemble` (0.382).

**Why:** for these models, prediction *magnitude* is uninformative about prediction
*accuracy*. Large predictions cluster in volatile regimes (COVID, 2022) where direction is
hardest, and the discarded low-|pred| days were the quiet uptrend days where being long
quietly worked. Concentrating on "high conviction" concentrated on the hardest days.

**Decision:** keep the `conviction` parameter in the code (default 0 = trade always) and
keep the `Hedge (Conviction)` row as a *documented negative result*. The model's confidence
does not track its accuracy — worth showing, not hiding.

**Lesson:** "only trade when confident" is sound intuition *only if* your confidence signal
is calibrated. Test that assumption before relying on it.

---

## The metrics, run by run (how the numbers evolved)

| Run | Key change | Hedge Sharpe | Hedge vs EqualWeight | SequenceAgent | Weight chart |
|---|---|---|---|---|---|
| 1 | original (after Phase A only) | 0.215 | **identical** (0.215) | 0.620 Sharpe / 28.5% dir (artifact) | flat 25/25/25/25 |
| 2 | + η normalize, LSTM padding, epochs | 0.871 | 0.871 vs 0.229 ✓ | 0.822 / 54.0% (honest) | collapses to 100% Sequence |
| 3 | + Fixed-Share | 0.368 | 0.368 vs 0.200 ✓ | 0.884 / 54.3% | adaptive, all 4 breathing ✓ |
| 4 | + directional (P&L) loss | 0.192 | 0.192 vs 0.145 ✓ | 0.698 / 53.9% | adaptive ✓ (but Hedge regressed) |
| 5 | revert to MSE + partial seed | 0.382 | 0.382 vs 0.241 ✓ | 0.728 / 54.6% | conviction tested → −0.133 (hurt) |
| 6 | + full GPU determinism | *re-run pending* | — | — | truly reproducible from here on |

(Runs 2–5 differed between executions because the LSTM seed didn't cover the GPU/cuDNN path —
absolute Sharpe wandered ~0.05–0.15 run to run, which is exactly why Correction 11 was
completed in Run 6. From Run 6 onward the numbers are reproducible. The *structural* findings
were stable throughout: Hedge > Equal Weight always; Buy & Hold is hard to beat; the
technical-indicator agents are coin flips; conviction thresholding doesn't help.)

---

## What was genuinely learned (the honest takeaways)

1. **The centerpiece algorithm was silently broken in three independent ways** (scale,
   collapse, objective). Each fix is a real, teachable lesson in online learning.
2. **Daily SPY return magnitude is essentially unpredictable** — every model correctly
   predicts ≈0, which is why the "Predicted vs Actual" line looks flat. Only *direction*
   is (weakly) predictable, so only directional metrics and the equity curve are
   meaningful.
3. **Beating Buy & Hold on daily index data is near-impossible**, and not doing so is the
   *scientifically correct* result (weak-form market efficiency + the equity premium +
   shorting against the secular uptrend). The legitimate selling point is **adaptive,
   regime-aware model selection with competitive risk-adjusted return and lower drawdown**
   — not "we beat the market."
4. **Metric artifacts are sneaky.** The LSTM's "positive Sharpe with 28% directional
   accuracy" looked like a result; it was a zero-padding artifact. Always reconcile metrics
   that seem to contradict each other.

---

*Stamatics IIT Kanpur · Mentor: Aayushman Tripathi · Corrections log maintained from the
first execution of the original 7-day build onward.*
