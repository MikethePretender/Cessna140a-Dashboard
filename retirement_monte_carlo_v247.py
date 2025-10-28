
import numpy as np
import pandas as pd
import streamlit as st

# ====================
# Helpers (from v247)
# ====================
def pct_to_decimal(p): 
    return p/100.0

def years_to_months(years): 
    import numpy as _np
    return int(_np.round(years*12))

def inflation_path(months, annual_inflation):
    dt = 1.0/12.0
    return (1+annual_inflation)**(np.arange(months)*dt)

def gbm_path_monthly(n_months, s0, annual_mu, annual_sigma, seed=None):
    rng = np.random.default_rng(seed)
    dt = 1.0/12.0
    mu = np.full(n_months, annual_mu, dtype=float) if np.isscalar(annual_mu) else np.asarray(annual_mu, dtype=float)
    sigma = np.full(n_months, annual_sigma, dtype=float) if np.isscalar(annual_sigma) else np.asarray(annual_sigma, dtype=float)
    prices = np.empty(n_months+1, dtype=float)
    prices[0] = float(s0)
    for t in range(n_months):
        z = rng.standard_normal()
        mu_t = float(mu[t])
        sig_t = float(sigma[t])
        prices[t+1] = prices[t] * np.exp((mu_t - 0.5 * sig_t**2) * dt + sig_t * np.sqrt(dt) * z)
    return prices

def btc_mu_sigma_schedules(n_months, start_age, retire_age, end_age,
                           mu_start, mu_taper_start_age, mu_end_age, mu_end,
                           sig_start, sig_taper_start_age, sig_end_age, sig_end):
    ages = start_age + (np.arange(n_months)+1)/12.0
    def schedule(v0, taper_start, v1, age1):
        arr = np.empty(n_months)
        for i, a in enumerate(ages):
            if a < taper_start:
                arr[i] = v0
            elif a <= age1:
                frac = (a - taper_start) / max(1e-9, (age1 - taper_start))
                frac = np.clip(frac, 0.0, 1.0)
                arr[i] = v0 + frac * (v1 - v0)
            else:
                arr[i] = v1
        return arr
    mu_arr  = schedule(mu_start,  mu_taper_start_age,  mu_end,  mu_end_age)
    sig_arr = schedule(sig_start, sig_taper_start_age, sig_end, sig_end_age)
    return mu_arr, sig_arr

# One simulation path
def simulate_once(params, seed=None):
    start_age = params['current_age']; retire_age = params['retirement_age']; final_age = params['final_age']
    months = years_to_months(final_age - start_age)

    # A (USD)
    A0 = params['A_start_balance']; A_mu = params['A_cagr']; A_sig = params['A_vol']
    A_w_amt = params['A_withdraw_amount']

    # B (BTC units + price)
    B0 = params['B_start_btc']; btc0 = params['btc_spot']
    B_mu_taper_start_age = params.get("B_mu_taper_start_age", 65.0)
    B_sig_taper_start_age = params.get("B_sig_taper_start_age", 65.0)
    B_mu0 = params['B_mu_start']; B_mu_age = params['B_mu_end_age']; B_mu1 = params['B_mu_end']
    B_sig0 = params['B_sig_start']; B_sig_age = params['B_sig_end_age']; B_sig1 = params['B_sig_end']

    infl = params['inflation']
    include_infl = params.get('include_inflation', True)
    B_w = params['B_withdraw_base']
    B_w_start_age = params['B_withdraw_start_age']
    B_start_earliest = params['B_start_earliest']

    ages_me = start_age + (np.arange(months)+1)/12.0
    infl_idx_full = inflation_path(months, infl)
    m_ret = years_to_months(retire_age - start_age)
    infl_idx_A = np.ones(months)
    if include_infl and m_ret < months:
        infl_idx_A[m_ret:] = inflation_path(months - m_ret, infl)

    # A withdrawals after retirement
    A_base = np.zeros(months)
    if m_ret < months:
        A_base[m_ret:] = A_w_amt
    A_w_sched = A_base * (infl_idx_A if include_infl else 1.0)

    mu_arr, sig_arr = btc_mu_sigma_schedules(months, start_age, retire_age, final_age, 
                                             B_mu0, B_mu_taper_start_age, B_mu_age, B_mu1, 
                                             B_sig0, B_sig_taper_start_age, B_sig_age, B_sig1)

    A_idx = gbm_path_monthly(months, 1.0, A_mu, A_sig, seed=seed)
    btc = gbm_path_monthly(months, btc0, mu_arr, sig_arr, seed=None if seed is None else seed+1)

    A = A0; B = B0
    A_bal = np.empty(months+1); B_units = np.empty(months+1)
    A_bal[0] = A; B_units[0] = B
    A_dep = None; B_dep = None
    B_start_m = years_to_months(B_w_start_age - start_age); B_started = False

    A_withdraw_taken = np.zeros(months)
    b_infl_months = 0
    for t in range(months):
        A *= A_idx[t+1]/A_idx[t]

        # Determine if BTC withdrawals active
        if not B_started:
            if B_start_earliest:
                if (t >= B_start_m) or (A <= 1e-9):
                    B_started = True
            else:
                if t >= B_start_m:
                    B_started = True

        A_need = A_w_sched[t]
        B_need = 0.0
        if B_started:
            b_infl_months += 1 if include_infl else 0
            b_factor = (1+infl)**(b_infl_months/12.0) if include_infl else 1.0
            B_need = B_w * b_factor

        takeA = min(A, A_need); A -= takeA; short = A_need - takeA
        A_withdraw_taken[t] = takeA

        if short > 1e-9 and B > 0.0:
            sell = min(B, short / btc[t+1]); B -= sell

        if B_need > 1e-9 and B > 0.0:
            sell = min(B, B_need / btc[t+1]); B -= sell

        if A_dep is None and A <= 1e-9: A_dep = ages_me[t]
        if B_dep is None and B <= 1e-12: B_dep = ages_me[t]

        A_bal[t+1] = A; B_units[t+1] = B

    out = pd.DataFrame({
        'month': np.arange(months+1),
        'age': start_age + np.arange(months+1)/12.0,
        'A_balance_usd': A_bal,
        'B_units': B_units,
        'btc_price': btc,
        'A_value_index': A_idx
    })
    out['A_withdraw_taken'] = np.append(A_withdraw_taken, 0.0)

    meta = {
        'A_depleted_age': A_dep,
        'B_depleted_age': B_dep,
        'A_end_balance': A_bal[-1],
        'B_end_units': B_units[-1],
        'btc_price_end': btc[-1],
    }
    return out, meta

def run_monte_carlo(params, n_sims=500, seed=42):
    paths = []; metas = []
    rng = np.random.default_rng(seed); seeds = rng.integers(0, 10_000_000, size=n_sims)
    for i in range(n_sims):
        df, m = simulate_once(params, seed=int(seeds[i])); paths.append(df); metas.append(m)

    ages = paths[0]['age'].values
    A_mat = np.vstack([p['A_balance_usd'].values for p in paths])
    B_mat = np.vstack([p['B_units'].values for p in paths])
    P_mat = np.vstack([p['btc_price'].values for p in paths])
    A_w_mat = np.vstack([p['A_withdraw_taken'].values for p in paths])

    A_q10 = np.nanpercentile(A_mat, 10, axis=0); A_q50 = np.nanpercentile(A_mat, 50, axis=0); A_q90 = np.nanpercentile(A_mat, 90, axis=0)
    B_q10 = np.nanpercentile(B_mat, 10, axis=0); B_q50 = np.nanpercentile(B_mat, 50, axis=0); B_q90 = np.nanpercentile(B_mat, 90, axis=0)
    BV_q10 = np.nanpercentile(B_mat*P_mat, 10, axis=0); BV_q50 = np.nanpercentile(B_mat*P_mat, 50, axis=0); BV_q90 = np.nanpercentile(B_mat*P_mat, 90, axis=0)

    A_deps = np.array([m['A_depleted_age'] if m['A_depleted_age'] is not None else np.nan for m in metas])
    B_deps = np.array([m['B_depleted_age'] if m['B_depleted_age'] is not None else np.nan for m in metas])

    A_end = np.array([m['A_end_balance'] for m in metas])
    B_end_units = np.array([m['B_end_units'] for m in metas])
    P_end = np.array([m['btc_price_end'] for m in metas])
    B_end_usd = B_end_units * P_end

    # Build year-by-year summary for A
    age_years = np.floor(ages).astype(int)
    unique_years = np.unique(age_years)
    year_idx_map = {yr: np.where(age_years == yr)[0] for yr in unique_years}

    n_sims = len(paths); n_years = len(unique_years)
    A_withdraw_annual = np.zeros((n_sims, n_years))
    A_balance_yearend = np.zeros((n_sims, n_years))

    for j, yr in enumerate(unique_years):
        idxs = year_idx_map[yr]
        idxs = idxs[(idxs >= 0) & (idxs < A_w_mat.shape[1])]
        if len(idxs) == 0: continue
        A_withdraw_annual[:, j] = np.sum(A_w_mat[:, idxs], axis=1)
        last_idx = idxs[-1]
        A_balance_yearend[:, j] = A_mat[:, last_idx]

    A_withdraw_annual_q10 = np.nanpercentile(A_withdraw_annual, 10, axis=0)
    A_withdraw_annual_q50 = np.nanmedian(A_withdraw_annual, axis=0)
    A_withdraw_annual_q90 = np.nanpercentile(A_withdraw_annual, 90, axis=0)

    A_balance_yearend_q10 = np.nanpercentile(A_balance_yearend, 10, axis=0)
    A_balance_yearend_q50 = np.nanmedian(A_balance_yearend, axis=0)
    A_balance_yearend_q90 = np.nanpercentile(A_balance_yearend, 90, axis=0)

    year_summary = pd.DataFrame({
        'Age (year)': unique_years,
        'A Withdrawal (median, annual USD)': A_withdraw_annual_q50,
        'A Withdrawal p10': A_withdraw_annual_q10,
        'A Withdrawal p90': A_withdraw_annual_q90,
        'A Year-end Balance (median USD)': A_balance_yearend_q50,
        'A Balance p10': A_balance_yearend_q10,
        'A Balance p90': A_balance_yearend_q90,
    })

    return {
        'ages': ages,
        'A_q10': A_q10, 'A_q50': A_q50, 'A_q90': A_q90,
        'B_q10': B_q10, 'B_q50': B_q50, 'B_q90': B_q90,
        'BV_q10': BV_q10, 'BV_q50': BV_q50, 'BV_q90': BV_q90,
        'A_dep_ages': A_deps, 'B_dep_ages': B_deps,
        'A_end_balances': A_end, 'B_end_units': B_end_units, 'B_end_usd': B_end_usd,
        'year_summary_A': year_summary
    }

# ====================
# UI (Dashboard)
# ====================
st.set_page_config(page_title="Retirement Monte Carlo — Dashboard", layout="wide")

# === Modern Light Theme (inline) ===
st.markdown("""
<style>
/* App background + typography */
html, body { background: #fafbfc; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
:root { --ec-primary: #4285f4; --ec-green: #34a853; --ec-text: #202124; --ec-border: rgba(0,0,0,0.08); }

/* Cards */
.ec-card, .stTable, .stDataFrame, div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--ec-border);
  border-radius: 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}

/* Hover lift for cards */
.ec-card:hover { box-shadow: 0 10px 24px rgba(0,0,0,0.08); transition: box-shadow 0.2s ease; }

/* Metrics */
div[data-testid="stMetricValue"] { color: var(--ec-text); font-weight: 800; }
div[data-testid="stMetricDelta"] span { font-weight: 700; }

/* Tabs styling (emphasize active tab) */
div[data-testid="stTabs"] > div[role="tablist"] { gap: 10px; border-bottom: 1px solid var(--ec-border); margin-bottom: 8px; }
div[data-testid="stTabs"] button[role="tab"] {
  background: #f7f9fc; border: 1px solid var(--ec-border); border-bottom: none;
  padding: 10px 16px; border-top-left-radius: 10px; border-top-right-radius: 10px;
  font-weight: 600; color: var(--ec-text);
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: #ffffff; box-shadow: 0 -2px 10px rgba(0,0,0,0.06);
  border-bottom: 2px solid var(--ec-primary);
}

/* Tables */
thead tr th { padding-top: 8px !important; padding-bottom: 8px !important; }
tbody tr td { padding-top: 6px !important; padding-bottom: 6px !important; }

/* Links */
a { color: var(--ec-primary); }
</style>
""", unsafe_allow_html=True)


# ---- Dashboard CSS (tabs emphasis) ----
st.markdown("""
<style>
/* Emphasize tabs */
div[data-testid="stTabs"] > div[role="tablist"] {
  gap: 8px;
  border-bottom: 1px solid rgba(0,0,0,0.08);
  margin-bottom: 8px;
}
div[data-testid="stTabs"] button[role="tab"] {
  background: #f7f9fc;
  border: 1px solid rgba(0,0,0,0.08);
  border-bottom: none;
  padding: 8px 14px;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
  font-weight: 600;
  color: #222;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: #ffffff;
  box-shadow: 0 -2px 8px rgba(0,0,0,0.06);
}
/* Slightly larger tab labels */
div[data-testid="stTabs"] button p {
  font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Simulation Inputs")
    current_age     = st.slider("Current age", 18, 100, 53, 1, key="cur_age")
    retirement_age  = st.slider("Retirement age", current_age, 100, max(55, current_age), 1, key="ret_age")
    final_age       = st.slider("Final age (model horizon)", retirement_age, 120, 120, 1, key="final_age")

    inflation_pct = st.slider("Annual inflation rate (%)", -5.0, 20.0, 2.0, 0.1, key="infl_pct")
    include_inflation = st.checkbox("Include inflation adjustment after withdrawals begin", True, key="infl_incl")

    st.markdown("---")
    st.subheader("Portfolio A (USD)")
    A_start_balance = st.slider("Starting balance (USD)", 0, 10_000_000, 1_650_000, 10_000, key="a_start")
    A_cagr_pct      = st.slider("CAGR (%)", -50.0, 50.0, 6.0, 0.1, key="a_cagr")
    A_vol_pct       = st.slider("Volatility (%)", 0.0, 100.0, 10.0, 0.1, key="a_vol")
    A_withdraw_amount = st.slider("A withdrawal (USD/month)", 0, 50_000, 10_000, 100, key="a_wdrw")

    st.markdown("---")
    st.subheader("Portfolio B (Bitcoin)")
    B_start_btc = st.slider("Starting BTC (units)", 0.0, 100.0, 11.1, 0.1, key="b_start")
    btc_spot        = st.slider("BTC spot price (USD)", 0, 1_000_000, 110_000, 1000, key="b_spot")
    B_mu_start_pct  = st.slider("BTC CAGR start (%)", -50.0, 200.0, 25.0, 0.1, key="b_mu_start")
    B_mu_end_pct    = st.slider("BTC CAGR taper to (%)", -50.0, 200.0, 10.0, 0.1, key="b_mu_end")
    B_mu_end_age    = st.slider("BTC CAGR taper stops at age", retirement_age, 120, max(80, retirement_age), 1, key="b_mu_end_age")

    st.markdown("**Volatility taper**")
    B_sig_start_pct = st.slider("BTC Vol start (%)", 0.0, 200.0, 25.0, 0.1, key="b_sig_start")
    B_sig_end_pct   = st.slider("BTC Vol at taper stop age (%)", 0.0, 200.0, 10.0, 0.1, key="b_sig_end")
    B_sig_end_age   = st.slider("BTC Vol taper stops at age", retirement_age, 120, max(80, retirement_age), 1, key="b_sig_end_age")

    B_withdraw_base = st.slider("B withdrawal (USD/month)", 0, 50_000, 10_000, 50, key="b_w_base")
    B_withdraw_start_age = st.slider("B withdrawal START age", retirement_age, 120, 120, 1, key="b_w_start_age")
    B_start_earliest = st.checkbox("Start BTC withdrawals at A depletion OR start age (whichever comes first)", True, key="b_start_earliest")

    st.markdown("---")
    n_sims = st.slider("Number of Monte Carlo simulations", 100, 10000, 2000, 100, key="n_sims")
    seed   = st.slider("Random seed", 0, 10_000_000, 42, 1, key="seed")

params = dict(
    current_age=float(current_age),
    retirement_age=float(retirement_age),
    final_age=float(final_age),
    A_start_balance=float(A_start_balance),
    A_cagr=pct_to_decimal(A_cagr_pct),
    A_vol=pct_to_decimal(A_vol_pct),
    A_withdraw_amount=float(A_withdraw_amount),
    B_start_btc=float(B_start_btc),
    btc_spot=int(btc_spot),
    B_mu_start=pct_to_decimal(B_mu_start_pct),
    B_mu_end=pct_to_decimal(B_mu_end_pct),
    B_mu_end_age=float(B_mu_end_age),
    B_mu_taper_start_age=float(max(65.0, float(current_age))),
    B_sig_start=pct_to_decimal(B_sig_start_pct),
    B_sig_end=pct_to_decimal(B_sig_end_pct),
    B_sig_end_age=float(B_sig_end_age),
    B_sig_taper_start_age=float(max(65.0, float(current_age))),
    inflation=pct_to_decimal(inflation_pct),
    include_inflation=bool(include_inflation),
    B_withdraw_base=float(B_withdraw_base),
    B_withdraw_start_age=float(B_withdraw_start_age),
    B_start_earliest=bool(B_start_earliest),
)

with st.spinner("Running Monte Carlo simulations…"):
    results = run_monte_carlo(params, n_sims=int(n_sims), seed=int(seed))

ages = results['ages']
A10, A50, A90 = results['A_q10'], results['A_q50'], results['A_q90']
B10, B50, B90 = results['B_q10'], results['B_q50'], results['B_q90']
BV10, BV50, BV90 = results['BV_q10'], results['BV_q50'], results['BV_q90']

A_deps = np.array(results['A_dep_ages'], dtype=float)
B_deps = np.array(results['B_dep_ages'], dtype=float)
success_mask = ~((A_deps <= float(final_age)) & (B_deps <= float(final_age)))
success_rate = float(np.nanmean(success_mask)) if success_mask.size else float('nan')

A_end_med = float(np.median(results['A_end_balances']))
B_end_med_units = float(np.median(results['B_end_units']))
B_end_med_usd = float(np.median(results['B_end_usd']))

# Header metrics

st.markdown("## Retirement Monte Carlo — Executive Dashboard")
# Metrics row with a compact donut chart
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,0.8])

# Compute median depletion age for Portfolio A (A50 crossing zero)
A_med_depl_age = None
try:
    import numpy as _np
    _hits = (A50 <= 1e-9)
    if _hits.any():
        _idx = int(_np.argmax(_hits))
        A_med_depl_age = float(ages[_idx])
except Exception:
    A_med_depl_age = None

c1.markdown("**Portfolio A**")
c1.metric("Portfolio A Depletion Age", f"{A_med_depl_age:.1f}" if A_med_depl_age is not None else "—")
c2.markdown("**Portfolio A**")
c2.metric("Median A End Balance", f"${A_end_med:,.0f}")
c3.markdown("**Portfolio B**")
c3.metric("Median B End (Units)", f"{B_end_med_units:,.4f} BTC")
c4.markdown("**Portfolio B**")
c4.metric("Median B End (USD)", f"${B_end_med_usd:,.0f}")


# Gauge next to metrics
import plotly.graph_objects as go
_gauge_val = round(success_rate * 100, 1)
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=_gauge_val,
    number={'suffix': "%", 'font': {'size': 28}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': "#4285f4"},
        'bgcolor': "white",
        'borderwidth': 1,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 50], 'color': "#f9dede"},
            {'range': [50, 80], 'color': "#fff3cd"},
            {'range': [80, 100], 'color': "#d4edda"}
        ],
    },
    title={'text': "Funding Confidence", 'font': {'size': 12}}
))
st.markdown("**Funding Confidence**")
fig_gauge.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=130)
c5.plotly_chart(fig_gauge, use_container_width=True)


tab_sum, tab_charts, tab_tables = st.tabs(["Summary", "Charts", "Tables"])

with tab_sum:
    st.markdown("### Key Takeaways")
    st.markdown(f"- Probability funded to age **{int(final_age)}** (not both portfolios depleted): **{success_rate*100:.1f}%**.")
    st.markdown(f"- Portfolio A median end balance: **${A_end_med:,.0f}**.")
    st.markdown(f"- Portfolio B median end value: **${B_end_med_usd:,.0f}** (**{B_end_med_units:.4f} BTC**).")
    st.caption("Shaded bands in charts show the 10th–90th percentile range across simulations.")
    # --- "About This Simulation" (Layperson + Inputs) ---
    with st.expander("About This Simulation", expanded=False):
        st.markdown(
            """
            <div class="ec-card">
              <p style="margin:0 0 10px 0;line-height:1.5;">
                This Monte Carlo simulation explores how two portfolios might behave through retirement under uncertainty.
                Each simulation path varies annual returns around expected growth (CAGR) and volatility to reflect real‑world market swings.
                The model accounts for monthly withdrawals, inflation adjustments (if enabled), and a tapering plan that gradually lowers
                Bitcoin’s expected growth and volatility as you age—similar to a glidepath that de‑risks over time.
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Build a live inputs table from current sidebar values
        _inputs_rows = [
            ("Current Age", "Starting age for the simulation", f"{current_age:.0f}"),
            ("Retirement Age", "Age when Portfolio A withdrawals begin", f"{retirement_age:.0f}"),
            ("Final Age (Horizon)", "Model end age", f"{final_age:.0f}"),
            ("Inflation", "Annual inflation used for withdrawal indexing", f"{inflation_pct:.1f}%"),
            ("Include Inflation", "Apply inflation to withdrawals after they begin", "Yes" if include_inflation else "No"),
            ("A — Starting Balance", "Initial USD balance", f"${A_start_balance:,.0f}"),
            ("A — CAGR", "Expected annual growth", f"{A_cagr_pct:.1f}%"),
            ("A — Volatility", "Annual volatility", f"{A_vol_pct:.1f}%"),
            ("A — Withdrawal (monthly)", "USD per month after retirement", f"${A_withdraw_amount:,.0f}"),
            ("B — Starting BTC", "Initial Bitcoin units", f"{B_start_btc:.4f} BTC"),
            ("B — BTC Spot", "USD price used for value", f"${btc_spot:,.0f}"),
            ("B — CAGR start", "Expected growth at start (before taper)", f"{B_mu_start_pct:.1f}%"),
            ("B — CAGR taper stops at age", "Age when BTC growth taper completes", f"{B_mu_end_age:.0f}"),
            ("B — CAGR taper to", "Expected growth at taper stop", f"{B_mu_end_pct:.1f}%"),
            ("B — Vol start", "BTC annual volatility at start", f"{B_sig_start_pct:.1f}%"),
            ("B — Vol taper stops at age", "Age when BTC vol taper completes", f"{B_sig_end_age:.0f}"),
            ("B — Vol at taper stop", "BTC annual volatility at taper stop age", f"{B_sig_end_pct:.1f}%"),
            ("B — Withdrawal (monthly)", "Target USD/month from BTC when active", f"${B_withdraw_base:,.0f}"),
            ("B — Withdrawal starts age", "Age when BTC withdrawals begin", f"{B_withdraw_start_age:.0f}"),
            ("BTC withdrawals start earliest", "Start at A depletion or start age (whichever first)", "Yes" if B_start_earliest else "No"),
            ("Simulations", "Number of Monte Carlo paths", f"{n_sims:,}"),
            ("Random Seed", "Seed for reproducibility", f"{seed}"),
        ]

        _df_inputs = pd.DataFrame(_inputs_rows, columns=["Parameter", "Description", "Value"])
        st.dataframe(_df_inputs, use_container_width=True)


    # --- Original Framework card ---
    st.markdown(
        """
        <div style="max-width:864px;margin:12px 0 0 0;">
          <div style="background:#ffffff;color:#111111;border-radius:12px;padding:20px 28px;border:1px solid rgba(0,0,0,0.08);box-shadow:0 8px 24px rgba(0,0,0,0.08);">
            <h3 style="text-align:center;margin:0 0 6px 0;">The Framework of this Monte Carlo Simulation</h3>
            <h4 style="margin:18px 0 6px 0;">Purpose</h4>
            <p style="margin:0 0 6px 0;line-height:1.45;">Quantifies portfolio longevity and the probability of depletion under uncertainty. It evaluates strategy resilience—not prediction—by simulating thousands of market paths.</p>
            <h4 style="margin:18px 0 6px 0;">Structure</h4>
            <p style="margin:0 0 6px 0;line-height:1.45;">Two investment sleeves are modeled: Portfolio A (traditional assets) and Portfolio B (Bitcoin). Each simulation accounts for CAGR, volatility, withdrawals, and inflation.</p>
            <h4 style="margin:18px 0 6px 0;">Process</h4>
            <p style="margin:0 0 6px 0;line-height:1.45;">Thousands of randomized paths represent potential market conditions across retirement years, capturing variability in returns and inflation effects.</p>
            <h4 style="margin:18px 0 6px 0;">Tapering Strategy</h4>
            <p style="margin:0 0 6px 0;line-height:1.45;">Implements a dynamic risk glidepath for Bitcoin, reducing both return and volatility assumptions from a chosen start age to a stop age—mimicking a progressive de-risking framework.</p>
            <h4 style="margin:18px 0 6px 0;">Outputs</h4>
            <p style="margin:0 0 6px 0;line-height:1.45;">Displays median and percentile wealth paths, probabilities of depletion by age, and median depletion ages (or '—' when depletion doesn’t occur).</p>
            <h4 style="margin:18px 0 6px 0;">Use Case</h4>
            <p style="margin:0 0 6px 0;line-height:1.45;">Provides data-driven insights for executive and advisory decision-making. Enables clear visualization of trade-offs between growth, risk, and spending policies.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with tab_charts:
    import plotly.graph_objects as go
    def usd_prefix(fig):
        fig.update_yaxes(tickprefix="$", separatethousands=True)
        return fig

    figA = go.Figure()
    figA.add_trace(go.Scatter(x=ages, y=A50, name="Median", mode="lines"))
    figA.add_trace(go.Scatter(x=ages, y=A90, name="90th pct", mode="lines", line=dict(dash="dot")))
    figA.add_trace(go.Scatter(x=ages, y=A10, name="10th pct", mode="lines", line=dict(dash="dot"), fill="tonexty"))
    figA.update_layout(title="Portfolio A — USD Balance", template="plotly_white", colorway=["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#6c757d"], legend_title="", margin=dict(l=20,r=20,t=50,b=40))
    st.plotly_chart(usd_prefix(figA), use_container_width=True)

    figB_units = go.Figure()
    figB_units.add_trace(go.Scatter(x=ages, y=B50, name="Median", mode="lines"))
    figB_units.add_trace(go.Scatter(x=ages, y=B90, name="90th pct", mode="lines", line=dict(dash="dot")))
    figB_units.add_trace(go.Scatter(x=ages, y=B10, name="10th pct", mode="lines", line=dict(dash="dot"), fill="tonexty"))
    figB_units.update_layout(title="Portfolio B — BTC Units", template="plotly_white", colorway=["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#6c757d"], legend_title="", margin=dict(l=20,r=20,t=50,b=40))
    st.plotly_chart(figB_units, use_container_width=True)

    figB_val = go.Figure()
    figB_val.add_trace(go.Scatter(x=ages, y=BV50, name="Median", mode="lines"))
    figB_val.add_trace(go.Scatter(x=ages, y=BV90, name="90th pct", mode="lines", line=dict(dash="dot")))
    figB_val.add_trace(go.Scatter(x=ages, y=BV10, name="10th pct", mode="lines", line=dict(dash="dot"), fill="tonexty"))
    figB_val.update_layout(title="Portfolio B — USD Value (units × price)", template="plotly_white", colorway=["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#6c757d"], legend_title="", margin=dict(l=20,r=20,t=50,b=40))
    st.plotly_chart(usd_prefix(figB_val), use_container_width=True)

with tab_tables:
    step = 5 * 12
    idxs = [0] + list(range(step, len(ages), step))
    if idxs[-1] != len(ages) - 1: idxs.append(len(ages) - 1)

    rows = []
    for i in idxs:
        age = ages[i]
        rows.append({
            "Age": f"{age:.0f}",
            "A (USD, median)": f"${A50[i]:,.0f}",
            "B (BTC units, median)": f"{B50[i]:,.4f} BTC",
            "B (USD value, median)": f"${BV50[i]:,.0f}",
        })
    five_table = pd.DataFrame(rows)
    st.markdown("#### 5-Year Snapshot (Median)")
    st.dataframe(five_table, use_container_width=True)

    year_df = results['year_summary_A'].copy()
    depl_mask = year_df["A Year-end Balance (median USD)"] <= 1e-9
    if depl_mask.any():
        cut_idx = int(np.argmax(depl_mask))
        year_df = year_df.iloc[:cut_idx + 1]

    def fmt_usd_series(s): return s.apply(lambda v: f"${v:,.0f}")
    display_df = pd.DataFrame({
        "Age": year_df["Age (year)"],
        "A Withdrawal (median)": fmt_usd_series(year_df["A Withdrawal (median, annual USD)"]),
        "A Year-end Balance (median)": fmt_usd_series(year_df["A Year-end Balance (median USD)"]),
    })
    st.markdown("#### Portfolio A — Year-by-Year (Median)")
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    cdl1, cdl2 = st.columns(2)
    cdl1.download_button(
        "Download 5-year snapshot (CSV)",
        data=five_table.to_csv(index=False).encode("utf-8"),
        file_name="five_year_snapshot.csv",
        mime="text/csv",
        use_container_width=True
    )
    cdl2.download_button(
        "Download Portfolio A yearly summary (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="portfolioA_yearly_summary.csv",
        mime="text/csv",
        use_container_width=True
    )
