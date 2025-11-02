
import numpy as np
import pandas as pd
import streamlit as st

# === Neon / Futuristic Plotly styling (auto-applied) ===
import plotly.graph_objects as go
import plotly.io as pio
import json

_NEON = {
    "accent":   "#00E5FF",
    "fill":     "rgba(0,229,255,0.14)",
    "glow":     "rgba(0,229,255,0.22)",
    "font":     "Inter, Segoe UI, Roboto, system-ui, -apple-system, sans-serif",
    "grid":     "rgba(255,255,255,0.06)",
}

def _neon_apply(fig, *, title=None):
    try:
        pio.templates.default = "plotly_dark"
        fig.update_layout(
            title=dict(
                text=title or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else None),
                x=0.02, xanchor="left",
                font=dict(size=20, color=_NEON["accent"]),
            ),
            font=dict(family=_NEON["font"], size=14, color="white"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            margin=dict(t=50, l=60, r=40, b=50),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=0
            ),
            modebar=dict(bgcolor="rgba(0,0,0,0)", color="#9aa0a6", activecolor=_NEON["accent"]),
        )
        fig.update_xaxes(showgrid=True, gridcolor=_NEON["grid"], zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=_NEON["grid"], zeroline=False)
        fig.update_traces(line=dict(width=2.6, color=_NEON["accent"]), selector=dict(mode="lines"))
    except Exception:
        pass
    return fig

def _neon_glow(fig, trace_index=0, width=9.0, alpha=0.22):
    try:
        if len(fig.data) == 0:
            return fig
        tr = fig.data[trace_index]
        fig.add_trace(go.Scatter(
            x=tr["x"], y=tr["y"], mode="lines",
            line=dict(width=width, color=_NEON["glow"]),
            hoverinfo="skip", showlegend=False
        ))
        fig.data = fig.data[:-1] + (fig.data[-1],)
    except Exception:
        pass
    return fig

# Monkey-patch Streamlit to auto-apply neon styling to all Plotly charts
try:
    _st_plotly_chart = st.plotly_chart
    def plotly_chart(fig, *args, **kwargs):
        fig = _neon_apply(fig)
        fig = _neon_glow(fig, 0)
        return _st_plotly_chart(fig, *args, **kwargs)
    st.plotly_chart = plotly_chart
except Exception:
    # If Streamlit is not yet imported or patching fails, ignore.
    pass


# Optional performance: Numba acceleration
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    _NUMBA_AVAILABLE = False
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
@st.cache_data(show_spinner=False)
def simulate_once(params, seed=None):
    start_age = params['current_age']; retire_age = params['retirement_age']; final_age = params['final_age']
    months = years_to_months(final_age - start_age)

    # A (USD)
    A0 = params['A_start_balance']; A_mu = params['A_cagr']; A_sig = params['A_vol']
    A_w_amt = params['A_withdraw_amount']
    A_c_amt = params.get('A_contrib_amount', 0.0)
    A_c_infl_on = params.get('A_contrib_infl', True)

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

    # A contributions before retirement
    A_c_base = np.zeros(months)
    if m_ret > 0:
        A_c_base[:m_ret] = A_c_amt
    A_c_infl = np.ones(months)
    if A_c_infl_on and m_ret > 0:
        A_c_infl[:m_ret] = inflation_path(m_ret, infl)
    A_c_sched = A_c_base * (A_c_infl if A_c_infl_on else 1.0)

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
        # Add contribution before withdrawals
        A += A_c_sched[t]


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

        # Only tap BTC to cover A shortfall once BTC withdrawals have officially started,
        # unless the user explicitly enabled "start at earliest (A depletion or start age)".
        if short > 1e-9 and B > 0.0:
            if B_started or B_start_earliest:
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

@st.cache_data(show_spinner=False)
def run_monte_carlo(params, n_sims=500, seed=42):
    paths = []; metas = []
    rng = np.random.default_rng(seed); seeds = rng.integers(0, 10_000_000, size=n_sims)

    # Experimental: small speedup by batching simulate_once; if Numba available, use prange for seed loop
    for i in (prange(n_sims) if _NUMBA_AVAILABLE else range(n_sims)):
        df, m = simulate_once(params, seed=int(seeds[i]))
        paths.append(df); metas.append(m)

    ages = paths[0]['age'].values
    A_mat = np.vstack([p['A_balance_usd'].values for p in paths])
    B_mat = np.vstack([p['B_units'].values for p in paths])
    P_mat = np.vstack([p['btc_price'].values for p in paths])
    A_w_mat = np.vstack([p['A_withdraw_taken'].values for p in paths])

    # Build a small sample of per-path trajectories for audit/export
    try:
        _sample_n = int(min(n_sims, 100))
        paths_sample = []
        for _sid, _df in enumerate(paths[:_sample_n]):
            dfc = _df.copy()
            dfc["sim_id"] = _sid
            paths_sample.append(dfc)
        paths_sample_df = pd.concat(paths_sample, ignore_index=True)
    except Exception:
        paths_sample_df = pd.DataFrame()

    A_q10 = np.nanpercentile(A_mat, 10, axis=0); A_q50 = np.nanpercentile(A_mat, 50, axis=0); A_q90 = np.nanpercentile(A_mat, 90, axis=0)
    B_q10 = np.nanpercentile(B_mat, 10, axis=0); B_q50 = np.nanpercentile(B_mat, 50, axis=0); B_q90 = np.nanpercentile(B_mat, 90, axis=0)
    BV_q10 = np.nanpercentile(B_mat*P_mat, 10, axis=0); BV_q50 = np.nanpercentile(B_mat*P_mat, 50, axis=0); BV_q90 = np.nanpercentile(B_mat*P_mat, 90, axis=0)

    # Additional bands for tables
    A_q25 = np.nanpercentile(A_mat, 25, axis=0); A_q75 = np.nanpercentile(A_mat, 75, axis=0)
    B_q25 = np.nanpercentile(B_mat, 25, axis=0); B_q75 = np.nanpercentile(B_mat, 75, axis=0)
    BV_q25 = np.nanpercentile(B_mat*P_mat, 25, axis=0); BV_q75 = np.nanpercentile(B_mat*P_mat, 75, axis=0)

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
        'A Withdrawal p25': np.nanpercentile(A_withdraw_annual, 25, axis=0),
        'A Withdrawal p75': np.nanpercentile(A_withdraw_annual, 75, axis=0),
        'A Year-end Balance (median USD)': A_balance_yearend_q50,
        'A Balance p10': A_balance_yearend_q10,
        'A Balance p90': A_balance_yearend_q90,
        'A Balance p25': np.nanpercentile(A_balance_yearend, 25, axis=0),
        'A Balance p75': np.nanpercentile(A_balance_yearend, 75, axis=0),
    })

    return {
            'ages': ages,
            'A_q25': A_q25, 'A_q75': A_q75,
            'B_q25': B_q25, 'B_q75': B_q75,
            'BV_q25': BV_q25, 'BV_q75': BV_q75,
            'A_q10': A_q10, 'A_q50': A_q50, 'A_q90': A_q90,
            'B_q10': B_q10, 'B_q50': B_q50, 'B_q90': B_q90,
            'BV_q10': BV_q10, 'BV_q50': BV_q50, 'BV_q90': BV_q90,
            'A_dep_ages': A_deps, 'B_dep_ages': B_deps,
            'A_end_balances': A_end, 'B_end_units': B_end_units, 'B_end_usd': B_end_usd,
            'year_summary_A': year_summary,
            'paths_sample': paths_sample_df
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

    # --- Apply any pending preset BEFORE widgets are created ---
    if "_pending_preset" in st.session_state:
        _pp = st.session_state.pop("_pending_preset", {})
        for _k, _v in _pp.items():
            st.session_state[_k] = _v
        st.session_state["_active_preset_name"] = st.session_state.get("_active_preset_name", "(loaded)")
        st.toast(f"Preset applied: {st.session_state['_active_preset_name']}", icon="✅")
    # --- Performance helper ---
    if st.button("⚡ Force recompute (clear cache)"):
        try:
            st.cache_data.clear()
            st.toast("Cache cleared. Re-run to recompute.", icon="⚡")
        except Exception:
            st.warning("Cache could not be cleared in this environment.")

    st.header("Simulation Inputs")
    current_age     = st.slider("Current age", 18, 100, 53, 1, key="cur_age")
    retirement_age  = st.slider("Retirement age", current_age, 100, max(55, current_age), 1, key="ret_age")
    final_age       = st.slider("Final age (model horizon)", retirement_age, 120, 120, 1, key="final_age")

    inflation_pct = st.slider("Annual inflation rate (%)", -5.0, 20.0, 2.0, 0.1, key="infl_pct")

    future_dollars = st.checkbox("Future dollars", value=False, key="future_dollars")

    if future_dollars:

        fiat_debasement_pct = st.slider("Annual fiat debasement (Future Dollars) (%)", 0.0, 10.0, 7.0, 0.1, key="fiat_deb")

    else:

        fiat_debasement_pct = 0.0

    include_inflation = st.checkbox("Include inflation adjustment after withdrawals begin", True, key="infl_incl")

    st.markdown("---")
    st.subheader("Portfolio A (USD)")
    A_start_balance = st.slider("Starting balance (USD)", 0, 10_000_000, 1_000_000, 10_000, key="a_start")
    A_cagr_pct      = st.slider("CAGR (%)", -50.0, 50.0, 6.0, 0.1, key="a_cagr")
    A_vol_pct       = st.slider("Volatility (%)", 0.0, 100.0, 10.0, 0.1, key="a_vol")
    A_withdraw_amount = st.slider("A withdrawal (USD/month)", 0, 50_000, 10_000, 100, key="a_wdrw")

    A_contrib_amount = st.slider("A monthly contribution until retirement (USD/month)", 0, 50_000, 0, 100, key="a_contrib")

    A_contrib_infl = st.checkbox("Increase contribution with inflation", True, key="a_contrib_infl")

    st.markdown("---")

    st.subheader("Portfolio B (Bitcoin)")
    B_start_btc = st.slider("Starting BTC (units)", 0.0, 100.0, 2.0, 0.1, key="b_start")
    btc_spot        = st.slider("BTC spot price (USD)", 0, 1_000_000, 110_000, 1000, key="b_spot")
    B_mu_start_pct  = st.slider("BTC CAGR start (%)", -50.0, 200.0, 25.0, 0.1, key="b_mu_start")
    # Moved outside advanced: B withdrawal monthly
    B_withdraw_base = st.slider("B withdrawal (USD/month)", 0, 50_000, 10_000, 50, key="b_w_base")

    with st.expander("Advanced BTC options", expanded=True):
        B_mu_end_pct    = st.slider("BTC CAGR taper to (%)", -50.0, 200.0, 10.0, 0.1, key="b_mu_end")
        B_mu_start_age  = st.slider("BTC CAGR taper starts at age", retirement_age, 120, max(65, retirement_age), 1, key="b_mu_start_age")
        B_mu_end_age    = st.slider("BTC CAGR taper stops at age", retirement_age, 120, max(80, retirement_age), 1, key="b_mu_end_age")

        st.markdown("**Volatility taper**")
        B_sig_start_pct = st.slider("BTC Vol start (%)", 0.0, 200.0, 25.0, 0.1, key="b_sig_start")
        B_sig_end_pct   = st.slider("BTC Vol at taper stop age (%)", 0.0, 200.0, 10.0, 0.1, key="b_sig_end")
        B_sig_start_age = st.slider("BTC Vol taper starts at age", retirement_age, 120, max(65, retirement_age), 1, key="b_sig_start_age")
        B_sig_end_age   = st.slider("BTC Vol taper stops at age", retirement_age, 120, max(80, retirement_age), 1, key="b_sig_end_age")

        B_withdraw_start_age = st.slider("B withdrawal START age", retirement_age, 120, 120, 1, key="b_w_start_age")
        B_start_earliest = st.checkbox("Start BTC withdrawals at Portfolio A depletion or start age (whichever comes first)", True, key="b_start_earliest")
    st.markdown("---")
    n_sims = st.slider("Number of Monte Carlo simulations", 100, 10000, 2000, 100, key="n_sims")
    seed   = st.slider("Random seed", 0, 10_000_000, 42, 1, key="seed")

    # --- Presets ---
    st.markdown("### Presets")
    if "presets" not in st.session_state:
        st.session_state["presets"] = {}
    preset_name = st.text_input("Preset name", value="Preset 1", key="preset_name")
    c1, c2, c3 = st.columns([1,1,1])

    core_keys = ["cur_age","ret_age","final_age","infl_pct","future_dollars","fiat_deb","infl_incl","a_start","a_cagr","a_vol","a_wdrw","a_contrib","a_contrib_infl","b_start","b_spot","b_mu_start","b_w_base","b_mu_end","b_mu_start_age","b_mu_end_age","b_sig_start","b_sig_end","b_sig_start_age","b_sig_end_age","b_w_start_age","b_start_earliest","n_sims","seed"]
    def _collect_core_state():
        d = {}
        for k in core_keys:
            if k in st.session_state:
                d[k] = st.session_state[k]
        return d

    with c1:
        if st.button("💾 Save preset"):
            st.session_state["presets"][preset_name] = _collect_core_state()
            st.session_state["_active_preset_name"] = preset_name
            st.toast(f"Saved preset “{preset_name}”.")

    with c2:
        _ap = st.session_state.get("_active_preset_name")
        if _ap: st.caption(f"Active preset: **{_ap}**")
        preset_choices = list(st.session_state["presets"].keys()) or []
        chosen = st.selectbox("Choose", options=preset_choices, key="preset_select")

    with c3:
        if st.button("📥 Load preset") and len(preset_choices) > 0:
            if chosen in st.session_state["presets"]:
                # Stage preset for next rerun so it applies before widgets
                st.session_state["_pending_preset"] = st.session_state["presets"][chosen]
                st.toast(f"Loaded preset “{chosen}”. Applying...")
                st.rerun()
            else:
                st.warning("No preset saved yet.")

    # JSON import/export
    st.download_button("⬇️ Download presets JSON", data=json.dumps(st.session_state["presets"], indent=2), file_name="presets.json", mime="application/json")
    up_json = st.file_uploader("⬆️ Upload presets JSON", type=["json"], key="preset_upload")
    if up_json is not None:
        try:
            new_presets = json.load(up_json)
            if isinstance(new_presets, dict):
                st.session_state["presets"].update(new_presets)
                st.success(f"Imported {len(new_presets)} presets.")
            else:
                st.error("Invalid JSON format (expected an object).")
        except Exception as e:
            st.error(f"Could not import presets: {e}")


params = dict(
    current_age=float(current_age),
    A_contrib_amount=float(A_contrib_amount),
    A_contrib_infl=bool(A_contrib_infl),
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
    B_mu_taper_start_age=float(B_mu_start_age),
    B_sig_start=pct_to_decimal(B_sig_start_pct),
    B_sig_end=pct_to_decimal(B_sig_end_pct),
    B_sig_end_age=float(B_sig_end_age),
    B_sig_taper_start_age=float(B_sig_start_age),
    inflation=pct_to_decimal(inflation_pct),
    include_inflation=bool(include_inflation),
    B_withdraw_base=float(B_withdraw_base),
    B_withdraw_start_age=float(B_withdraw_start_age),
    B_start_earliest=bool(B_start_earliest),
    fiat_debasement=pct_to_decimal(fiat_debasement_pct),
)
with st.spinner("Running Monte Carlo simulations…"):
    results = run_monte_carlo(params, n_sims=int(n_sims), seed=int(seed))

ages = results['ages']
A10, A50, A90 = results['A_q10'], results['A_q50'], results['A_q90']
B10, B50, B90 = results['B_q10'], results['B_q50'], results['B_q90']
BV10, BV50, BV90 = results['BV_q10'], results['BV_q50'], results['BV_q90']
A_q25, A_q75 = results['A_q25'], results['A_q75']
B_q25, B_q75 = results['B_q25'], results['B_q75']
BV_q25, BV_q75 = results['BV_q25'], results['BV_q75']

# (Future dollars presentation intentionally not applied to charts)

A_deps = np.array(results['A_dep_ages'], dtype=float)
B_deps = np.array(results['B_dep_ages'], dtype=float)
success_mask = ~((A_deps <= float(final_age)) & (B_deps <= float(final_age)))
success_rate = float(np.nanmean(success_mask)) if success_mask.size else float('nan')

# --- Export per-path sample (CSV) ---
with st.expander("Export per-path sample (first up to 100 simulations)"):
    sample_df = results.get('paths_sample', None)
    if sample_df is not None and not getattr(sample_df, "empty", True):
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"path_sample_{int(n_sims)}sims_seed{int(seed)}.csv", mime="text/csv")
    else:
        st.info("No sample available.")


A_end_med = float(np.median(results['A_end_balances']))
B_end_med_units = float(np.median(results['B_end_units']))
B_end_med_usd = float(np.median(results['B_end_usd']))
# Adjust headline metrics for future dollars if enabled
try:
    _fiat_dec = pct_to_decimal(fiat_debasement_pct)
    if future_dollars and _fiat_dec > 0:
        _h = float(final_age) - float(current_age)
        _factor_h = (1.0 + _fiat_dec) ** _h
        A_end_med /= _factor_h
        B_end_med_usd /= _factor_h
except Exception:
    pass

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
label_a = "Median A End Balance" + (" (Future $)" if future_dollars else "")
c2.metric(label_a, f"${A_end_med:,.0f}")
if future_dollars:
    c2.markdown(
        "<div style='color:#1a73e8; font-size:0.9em; text-align:center; font-weight:600;'>💠 Future $</div>",
        unsafe_allow_html=True,
    )
c3.markdown("**Portfolio B**")
c3.metric("Median B End (Units)", f"{B_end_med_units:,.4f} BTC")
c4.markdown("**Portfolio B**")
label_b = "Median B End (USD)" + (" (Future $)" if future_dollars else "")
c4.metric(label_b, f"${B_end_med_usd:,.0f}")
if future_dollars:
    c4.markdown(
        "<div style='color:#1a73e8; font-size:0.9em; text-align:center; font-weight:600;'>💠 Future $</div>",
        unsafe_allow_html=True,
    )
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

tab_charts, tab_sum, tab_tables = st.tabs(["Charts", "Summary", "Tables"])

with tab_sum:



    # --- Animated "Summary Portal" (Neon + Multiverse Lottie) ---
    # Dependencies: pip install streamlit-lottie
    from typing import Optional
    try:
        from streamlit_lottie import st_lottie
    except Exception:
        st.write("Tip: Install animation support with: pip install streamlit-lottie")

    def load_lottie_url(url: str) -> Optional[dict]:
        try:
            import requests
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    # Local Summary theme toggle (persists between reruns)
    if "summary_neon" not in st.session_state:
        st.session_state["summary_neon"] = True
    st.session_state["summary_neon"] = st.toggle("🟢 Neon Mode (Summary)", value=st.session_state["summary_neon"], help="Switches the Summary look only")

    # Styles
    if st.session_state["summary_neon"]:
        st.markdown("""
        <style>
        .portal-wrap { padding: 10px 14px 0; border-radius: 14px;
          background: linear-gradient(180deg, rgba(10,15,30,0.9), rgba(11,16,34,0.92));
          border: 1px solid rgba(0,229,255,.18);
          box-shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 0 32px rgba(0,229,255,.08);
          transition: all 1s ease;
        }
        .portal-title {
          font-size: 1.7rem; font-weight: 800; margin: .2rem 0 .25rem;
          background: linear-gradient(90deg,#00e5ff 0%,#7b61ff 50%,#ff7b9c 100%);
          -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .bar { height:2px; margin:6px 0 12px;
          background: linear-gradient(90deg,rgba(0,229,255,.0),rgba(0,229,255,.9),rgba(123,97,255,.9),rgba(0,229,255,.0));
          box-shadow: 0 0 12px rgba(0,229,255,.35), 0 0 20px rgba(123,97,255,.25);
        }
        .glass { padding: 10px 12px; border-radius: 12px;
          background: linear-gradient(180deg, rgba(0,229,255,.10), rgba(123,97,255,.08));
          border: 1px solid rgba(0,229,255,.25);
          color:#e5faff; font-size:.95rem;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .portal-wrap { padding: 10px 14px 0; border-radius: 14px;
          background: #ffffff; border: 1px solid rgba(0,0,0,.06);
          box-shadow: 0 10px 24px rgba(0,0,0,.06); transition: all 1s ease;
        }
        .portal-title { font-size: 1.7rem; font-weight: 800; margin:.2rem 0 .25rem; color:#0f172a; }
        .bar { height:2px; margin:6px 0 12px; background: rgba(0,0,0,.08); }
        .glass { padding: 10px 12px; border-radius: 12px; background:#f8fafc; color:#0f172a; }
        </style>
        """, unsafe_allow_html=True)

    # Header + animation
    st.markdown('<div class="portal-wrap">', unsafe_allow_html=True)
    try:
        ani = load_lottie_url("https://lottie.host/171c08dc-7ef1-4985-bb37-2d3d8c25f78e/5C5j9kq2uj.json")
        if ani and 'st_lottie' in globals():
            st_lottie(ani, height=180, speed=0.7, loop=True, quality="high")
    except Exception:
        pass

    st.markdown('<div class="portal-title">Monte Carlo: What this simulation actually does</div>', unsafe_allow_html=True)
    st.markdown('<div class="bar"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">This app runs thousands of randomized futures to quantify uncertainty—so you can compare strategies by <b>probabilities</b> and <b>ranges</b>, not a single forecast.</div>', unsafe_allow_html=True)

    # Expanders content
    with st.expander("How it works (high level) ⚙️", expanded=True):
        st.markdown("""
    - Start with today’s balances for **Portfolio A (USD assets)** and **Portfolio B (BTC)**.
    - Each month in each path, we apply random returns from your **CAGR/vol** assumptions, handle **inflation**, add **contributions**, and process **withdrawals**.
    - We repeat to age 120 and run the whole path **thousands of times**.
    - Finally, we plot **median and percentile bands** and compute **funding probability**.
    """)

    with st.expander("The strategy under the microscope 🎯", expanded=True):
        st.markdown("""
    **Primary rule:** draw spending from **Portfolio A first** while allowing **Portfolio B** to compound untouched; switch to B **only when A depletes** (or at your forced start age, if enabled).

    **Why this can be a creative, asymmetric approach**
    - **Preserve upside:** BTC’s return distribution is wide. Preserving units lets the right tail do more work in strong paths.
    - **Sequence risk mitigation:** Spending first from lower-volatility USD can reduce forced BTC sales after drawdowns.
    - **Operational clarity:** One source until a clear trigger, instead of constant rebalancing.

    **Trade-offs to understand**
    - If BTC underperforms for years, delaying BTC withdrawals may **postpone diversification** benefits.
    - If A’s real return is weak vs inflation, A-first can **deplete sooner** than blended rules.
    - It’s a **barbell** (safety-now / growth-later); risk tolerance matters.
    """)

    with st.expander("Reading the outputs 📊", expanded=False):
        st.markdown("""
    - **Funding probability** = share of trials that avoid full depletion before terminal age, under your inputs.
    - **Median vs band:** You’ll live one path, not the median. The shaded (p10–p90) area shows where most paths landed.
    - **BTC units vs USD value:** We display both — **units** (what you own) and **value** (units × price).
    """)

    with st.expander("Assumptions you control 🔧", expanded=False):
        st.markdown("""
    - **CAGR & volatility** (with optional BTC return/vol tapers)
    - **Inflation** for spending and contributions
    - **Monthly contribution to A** until retirement (**inflation on/off**)
    - **Withdrawal policy:** A-first, then B; optional “start at A depletion **or** forced start age”
    """)

    with st.expander("Limitations & good hygiene 🧭", expanded=False):
        st.markdown("""
    - Simulations **model randomness, not predictions**; changing inputs changes outcomes.
    - Returns are simplified (e.g., normal log-returns); taxes/fees may be approximated unless explicitly modeled.
    - Use this as a **decision aid**, not advice.
    """)

    st.markdown('</div>', unsafe_allow_html=True)  # close portal-wrap
    # --- End Animated Summary Portal ---
with tab_charts:

    import plotly.graph_objects as go

    # Chart dollar mode
    chart_mode = st.selectbox("Charts dollar mode", ["Match tables", "Nominal USD", "Future $"], index=0, key="chart_mode")
    _deb_rate = pct_to_decimal(fiat_debasement_pct) if "fiat_debasement_pct" in globals() else 0.0

    def _chart_divisors(ages_arr):
        # Decide whether charts use divisors
        _use_future = future_dollars if chart_mode == "Match tables" else (chart_mode == "Future $")
        if not _use_future or _deb_rate <= 0:
            return np.ones_like(ages_arr, dtype=float)
        return (1.0 + _deb_rate) ** (ages_arr - float(current_age))

    _chart_divs = _chart_divisors(ages)

    # Apply divisors to A-USD and B-USD series (units unchanged)
    A10_adj, A50_adj, A90_adj = A10/_chart_divs, A50/_chart_divs, A90/_chart_divs
    BV10_adj, BV50_adj, BV90_adj = BV10/_chart_divs, BV50/_chart_divs, BV90/_chart_divs

    def usd_prefix(fig):
        fig.update_yaxes(tickprefix="$", separatethousands=True)
        return fig

    figA = go.Figure()
    figA.add_trace(go.Scatter(x=ages, y=A50_adj, name="Median", mode="lines", line=dict(color="#4285f4", width=2)))
    figA.add_trace(go.Scatter(x=ages, y=A90_adj, name="90th pct", mode="lines", line=dict(color="rgba(66,133,244,0.7)", dash="dot")))
    figA.add_trace(go.Scatter(x=ages, y=A10_adj, name="10th pct", mode="lines", line=dict(color="rgba(66,133,244,0.2)", dash="dot"), fill="tonexty", fillcolor="rgba(66,133,244,0.5)"))
    figA.update_layout(title="Portfolio A — USD Balance", template="plotly_white", colorway=["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#6c757d"], legend_title="", margin=dict(l=20,r=20,t=50,b=40))
    st.plotly_chart(usd_prefix(figA), use_container_width=True)

    figB_units = go.Figure()
    figB_units.add_trace(go.Scatter(x=ages, y=B50, name="Median", mode="lines", line=dict(color="#4285f4", width=2)))
    figB_units.add_trace(go.Scatter(x=ages, y=B90, name="90th pct", mode="lines", line=dict(color="rgba(66,133,244,0.7)", dash="dot")))
    figB_units.add_trace(go.Scatter(x=ages, y=B10, name="10th pct", mode="lines", line=dict(color="rgba(66,133,244,0.2)", dash="dot"), fill="tonexty", fillcolor="rgba(66,133,244,0.5)"))
    figB_units.update_layout(title="Portfolio B — BTC Units", template="plotly_white", colorway=["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#6c757d"], legend_title="", margin=dict(l=20,r=20,t=50,b=40))
    st.plotly_chart(figB_units, use_container_width=True)

    figB_val = go.Figure()
    figB_val.add_trace(go.Scatter(x=ages, y=BV50_adj, name="Median", mode="lines", line=dict(color="#4285f4", width=2)))
    figB_val.add_trace(go.Scatter(x=ages, y=BV90_adj, name="90th pct", mode="lines", line=dict(color="rgba(66,133,244,0.7)", dash="dot")))
    figB_val.add_trace(go.Scatter(x=ages, y=BV10_adj, name="10th pct", mode="lines", line=dict(color="rgba(66,133,244,0.2)", dash="dot"), fill="tonexty", fillcolor="rgba(66,133,244,0.5)"))
    figB_val.update_layout(title="Portfolio B — USD Value (units × price)", template="plotly_white", colorway=["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#6c757d"], legend_title="", margin=dict(l=20,r=20,t=50,b=40))
    st.plotly_chart(usd_prefix(figB_val), use_container_width=True)

with tab_tables:

    # Helper: per-age debasement divisor when "Future dollars" is enabled
    def _future_divisors_for_ages(age_array, current_age, deb_rate):
        if not future_dollars or deb_rate <= 0:
            return np.ones_like(age_array, dtype=float)
        return (1.0 + deb_rate) ** (age_array - float(current_age))
    _deb_rate = pct_to_decimal(fiat_debasement_pct) if "fiat_debasement_pct" in globals() else 0.0

    # Toggle defaults (unchecked)
    _show_pcts_five = False
    _show_pcts_yby = False

    # ---- 5-Year Snapshot (Future $ respected) ----
    step = 5 * 12
    idxs = [0] + list(range(step, len(ages), step))
    if idxs[-1] != len(ages) - 1:
        idxs.append(len(ages) - 1)

    st.markdown("#### 5-Year Snapshot")
    _show_pcts_five = st.checkbox("Show percentile bands for 5-year snapshot", value=False, key="pct_five")

    rows = []
    for i in idxs:
        age = ages[i]
        div = 1.0
        if future_dollars and _deb_rate > 0:
            div = (1.0 + _deb_rate) ** (float(age) - float(current_age))
        a_val = A50[i] / div
        bv_val = BV50[i] / div

        row = {
            "Age": f"{age:.0f}",
            f"A ({'Future $' if future_dollars else 'USD'}, median)": f"${a_val:,.0f}",
            "B (BTC units, median)": f"{B50[i]:,.4f} BTC",
            f"B ({'Future $' if future_dollars else 'USD value'}, median)": f"${bv_val:,.0f}",
        }
        if _show_pcts_five:
            row.update({
                f"A ({'Future $' if future_dollars else 'USD'}, p25)": f"${(A_q25[i]/div):,.0f}",
                f"A ({'Future $' if future_dollars else 'USD'}, p75)": f"${(A_q75[i]/div):,.0f}",
                "B (BTC units, p25)": f"{B_q25[i]:,.4f} BTC",
                "B (BTC units, p75)": f"{B_q75[i]:,.4f} BTC",
                f"B ({'Future $' if future_dollars else 'USD value'}, p25)": f"${(BV_q25[i]/div):,.0f}",
                f"B ({'Future $' if future_dollars else 'USD value'}, p75)": f"${(BV_q75[i]/div):,.0f}",
            })



        rows.append(row)

    five_table = pd.DataFrame(rows)
    st.dataframe(five_table, use_container_width=True)

    # ---- Portfolio A — Year-by-Year (NOMINAL ONLY) ----
    year_df = results['year_summary_A'].copy()

    # Cut at depletion if needed
    depl_mask = year_df["A Year-end Balance (median USD)"] <= 1e-9
    if depl_mask.any():
        cut_idx = int(np.argmax(depl_mask))
        year_df = year_df.iloc[:cut_idx + 1]

    def fmt_usd_series(s): return s.apply(lambda v: f"${v:,.0f}")

    _wd_vals = year_df["A Withdrawal (median, annual USD)"].astype(float).values
    _bal_vals = year_df["A Year-end Balance (median USD)"].astype(float).values

    display_df = pd.DataFrame({
        "Age": year_df["Age (year)"],
        "A Withdrawal (median)": fmt_usd_series(pd.Series(_wd_vals)),
        "A Year-end Balance (median)": fmt_usd_series(pd.Series(_bal_vals)),
    })

    _show_pcts_yby = st.checkbox("Show percentile bands for year-by-year", value=False, key="pct_yby")

    if _show_pcts_yby and "A Withdrawal p25" in year_df.columns and "A Withdrawal p75" in year_df.columns:
        display_df["A Withdrawal p25"] = fmt_usd_series(year_df["A Withdrawal p25"])
        display_df["A Withdrawal p75"] = fmt_usd_series(year_df["A Withdrawal p75"])
    if _show_pcts_yby and "A Balance p25" in year_df.columns and "A Balance p75" in year_df.columns:
        display_df["A YE Balance p25"] = fmt_usd_series(year_df["A Balance p25"])
        display_df["A YE Balance p75"] = fmt_usd_series(year_df["A Balance p75"])

    st.markdown("#### Portfolio A — Year-by-Year")
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


# === Context-aware upgrades for neon styling ===
def _neon_smart(fig):
    # Decide formatting by chart title
    title = (fig.layout.title.text or "").lower() if fig.layout and fig.layout.title else ""
    if "btc unit" in title:
        yfmt = "btc"
    elif "usd" in title or "$" in title or "value" in title:
        yfmt = "usd"
    else:
        yfmt = "raw"

    # Apply hover templates only to line traces (avoid overriding filled bands)
    if yfmt == "usd":
        hover = "<b>Age %{x}</b><br>Value: %{y:$,.0f}<extra></extra>"
        fig.update_yaxes(tickprefix="$", tickformat="~s")
    elif yfmt == "btc":
        hover = "<b>Age %{x}</b><br>Units: %{y:,.4f} BTC<extra></extra>"
        fig.update_yaxes(tickformat=".4f")
    else:
        hover = "<b>Age %{x}</b><br>%{y:,.2f}<extra></extra>"

    try:
        for tr in fig.data:
            if getattr(tr, "mode", None) and "lines" in tr.mode and getattr(tr, "fill", None) in (None, "none"):
                tr.hovertemplate = hover
    except Exception:
        pass

    # Add vertical target-age marker if not present (assume age 120)
    try:
        _shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        has_v120 = any(getattr(s, "type", "") == "line" and str(getattr(s, "x0", "")) == "120" and str(getattr(s, "x1", "")) == "120" for s in _shapes)
        if not has_v120:
            fig.add_vline(x=120, line_width=1, line_dash="dot", line_color="#9AA0A6", opacity=0.7)
    except Exception:
        pass

    return fig

# Replace the earlier monkey patch with a smarter one
try:
    _st_plotly_chart  # if exists
except NameError:
    try:
        _st_plotly_chart = st.plotly_chart
    except Exception:
        _st_plotly_chart = None

if _st_plotly_chart is not None:
    def plotly_chart(fig, *args, **kwargs):
        fig = _neon_apply(fig)
        fig = _neon_smart(fig)
        fig = _neon_glow(fig, 0)
        return _st_plotly_chart(fig, *args, **kwargs)
    st.plotly_chart = plotly_chart