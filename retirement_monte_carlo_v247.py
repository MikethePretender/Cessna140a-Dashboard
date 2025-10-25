
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
st.set_page_config(page_title="Retirement Monte Carlo", layout="wide")


# Hide Streamlit slider value labels (floating numbers)
st.markdown(
    """
    <style>
    div[data-testid="stThumbValue"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==== Center matplotlib charts globally to align with framework card ====
if "_center_matplotlib_css" not in st.session_state:
    st.session_state["_center_matplotlib_css"] = True
    st.markdown(
        """
        <style>
        /* Streamlit renders Matplotlib figures via stImage; centre and size them here */
        div[data-testid="stImage"] {
            max-width: 864px;
            margin: 12px auto 16px auto;  /* align with framework box */
            text-align: center;
        }
        div[data-testid="stImage"] img, div[data-testid="stImage"] canvas {
            width: 518px;     /* identical width for every chart */
            height: auto;     /* preserve aspect ratio (518 x ~292) */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# ======================================================================

# ==== Center expanders to align with framework card ====
if "_center_expanders_css" not in st.session_state:
    st.session_state["_center_expanders_css"] = True
    st.markdown(
        """
        <style>
        /* Center any Streamlit expander and cap its width to the framework's width */
        div[data-testid="stExpander"] {
            max-width: 864px;
            margin: 12px auto 16px auto; /* tight gap under the framework */
        }
        /* Optional: soften expander corners to match the framework card */
        div[data-testid="stExpander"] > details {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# =======================================================

def show_centered_fig(fig, target_width_px: int = 518, target_height_px: int = 292):
    """
    Render charts uniformly (518x292 px) centered directly beneath the framework card.
    Vertical spacing reduced for a unified visual alignment.
    """
    try:
        dpi = fig.dpi or 100
        new_w_in = max(1.0, float(target_width_px) / float(dpi))
        new_h_in = max(1.0, float(target_height_px) / float(dpi))
        fig.set_size_inches(new_w_in, new_h_in, forward=True)
    except Exception:
        pass

    # Tighter vertical spacing (12px top margin) and centered alignment under framework
    st.markdown("<div style='max-width:864px;margin:12px auto 16px auto;text-align:center;'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
# (Removed duplicate helper block)

def _safe_clear_and_rerun():
    # Clear new-style caches
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    # Legacy caches (older Streamlit)
    for _legacy in ("experimental_memo", "experimental_singleton"):
        try:
            getattr(st, _legacy).clear()  # type: ignore[attr-defined]
        except Exception:
            pass
    # Clear all session state
    st.session_state.clear()
    # Rerun across versions
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            st.warning("Cleared. Please rerun the app (Ctrl+R / ⌘R).")

# ===== Sidebar Controls (Top) =====
with st.sidebar:
    run_clicked = st.button("▶️ Run simulation", help="Generate charts, tables, and metrics using the current inputs.", use_container_width=True)
    clear_clicked = st.button("🧹 Clear cache & contents", help="Reset all inputs and clear Streamlit caches.", use_container_width=True)

    if run_clicked:
        st.session_state['_trigger_run'] = True
        st.session_state['_run_sim'] = True

    if clear_clicked:
        _safe_clear_and_rerun()

# Gate the rest of the app until user clicks Run
if not st.session_state.get('_run_sim', False):
    st.info("Adjust inputs in the sidebar, then click **Run simulation** to generate results.")
    st.stop()
# ==================================
def _pdf_exec_summary(c, x=40, y=720):
    from reportlab.lib.utils import simpleSplit
    c.setFont('Helvetica-Bold', 12)
    c.drawString(x, y, 'The Framework of this Monte Carlo Simulation')
    y -= 18
    def section(header, body):
        nonlocal y
        c.setFont('Helvetica-Bold', 11)
        c.drawString(x, y, header)
        y -= 14
        c.setFont('Helvetica', 10)
        for ln in simpleSplit(body, 'Helvetica', 10, 520):
            if y < 60:
                c.showPage(); y = 760; c.setFont('Helvetica', 10)
            c.drawString(x, y, ln)
            y -= 14
        y -= 6
    section('Purpose', 'Quantifies portfolio longevity and the probability of depletion under uncertainty. It evaluates strategy resilience, not prediction, by simulating thousands of market paths.')
    section('Structure', 'Two investment sleeves are modeled: Portfolio A (traditional assets) and Portfolio B (Bitcoin). Each simulation accounts for CAGR, volatility, withdrawals, and inflation.')
    section('Process', 'Thousands of randomized paths represent potential market conditions across retirement years, capturing variability in returns and inflation effects.')
    section('Tapering Strategy', 'Implements a dynamic risk glidepath for Bitcoin, reducing both return and volatility assumptions from a chosen start age to a stop age—mimicking a progressive de-risking framework.')
    section('Outputs', 'Displays median and percentile wealth paths, probabilities of depletion by age, and median depletion ages (or "-" when depletion does not occur).')
    section('Use Case', 'Provides data-driven insights for executive and advisory decision-making. Enables clear visualization of trade-offs between growth, risk, and spending policies.')
    return y

# ===== About (Layperson Summary) =====
# -------- Helpers --------
def pct_to_decimal(p): return p/100.0
def years_to_months(years): return int(np.round(years*12))
def inflation_path(months, annual_inflation):
    dt = 1.0/12.0
    return (1+annual_inflation)**(np.arange(months)*dt)
def gbm_path_monthly(n_months, s0, annual_mu, annual_sigma, seed=None):
    rng = np.random.default_rng(seed)
    dt = 1.0/12.0
    # Convert mu/sigma to 1-D arrays of length n_months
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
def rolling_recent_high(prices, window_months=1):
    s = pd.Series(prices)
    win = max(1, int(window_months))
    return s.rolling(window=win, min_periods=1).max().values
def usd_formatter(): return FuncFormatter(lambda x, pos: f"${x:,.0f}")

# -------- One simulation path (with actual A withdrawals tracked) --------
def simulate_once(params, seed=None):
    start_age = params['current_age']; retire_age = params['retirement_age']; final_age = params['final_age']
    months = years_to_months(final_age - start_age)

    # A (USD) simplified
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
    pause = params.get('pause_enabled', False); thr = params.get('pause_threshold', 0.8); look = int(params.get('pause_window_m', 24))

    ages_me = start_age + (np.arange(months)+1)/12.0
    infl_idx_full = inflation_path(months, infl)
    # For A: start inflation at retirement month so index=1 at first withdrawal
    m_ret = years_to_months(retire_age - start_age)
    infl_idx_A = np.ones(months)
    if include_infl and m_ret < months:
        infl_idx_A[m_ret:] = inflation_path(months - m_ret, infl)

    # Portfolio A: flat withdrawal post-retirement
    m_ret = years_to_months(retire_age - start_age)
    A_base = np.zeros(months)
    if m_ret < months:
        A_base[m_ret:] = A_w_amt
    A_w_sched = A_base * (infl_idx_A if include_infl else 1.0)  # scheduled withdrawal, not constrained by depletion

    mu_arr, sig_arr = btc_mu_sigma_schedules(months, start_age, retire_age, final_age, B_mu0, B_mu_taper_start_age, B_mu_age, B_mu1, B_sig0, B_sig_taper_start_age, B_sig_age, B_sig1)

    A_idx = gbm_path_monthly(months, 1.0, A_mu, A_sig, seed=seed)
    btc = gbm_path_monthly(months, btc0, mu_arr, sig_arr, seed=None if seed is None else seed+1)
    recent = rolling_recent_high(btc, window_months=max(1, look))

    A = A0; B = B0
    A_bal = np.empty(months+1); B_units = np.empty(months+1)
    A_bal[0] = A; B_units[0] = B
    A_dep = None; B_dep = None
    B_start_m = years_to_months(B_w_start_age - start_age); B_started = False

    # track actual withdrawals taken from A
    A_withdraw_taken = np.zeros(months)
    b_infl_months = 0  # months since B withdrawals began (for inflation)
    for t in range(months):
        # Grow A
        A *= A_idx[t+1]/A_idx[t]

        # BTC pause rule
        btc_pause = False
        if pause and btc[t+1] < thr*recent[t+1]: btc_pause = True
        if A_dep is not None: btc_pause = False  # ignore pause after A depleted

        # Determine if BTC withdrawals active
        if not B_started:
            if B_start_earliest:
                if (t >= B_start_m) or (A <= 1e-9):
                    B_started = True
            else:
                if t >= B_start_m:
                    B_started = True

        # Needs this month
        A_need = A_w_sched[t]
        B_need = 0.0
        if B_started:
            b_infl_months += 1 if include_infl else 0
            b_factor = (1+infl)**(b_infl_months/12.0) if include_infl else 1.0
            B_need = B_w * b_factor
        if btc_pause:
            A_need += B_need; B_need = 0.0

        # Fulfill from A, record actual
        takeA = min(A, A_need); A -= takeA; short = A_need - takeA
        A_withdraw_taken[t] = takeA

        # Recoup any shortfall from B
        if short > 1e-9 and B > 0.0:
            sell = min(B, short / btc[t+1]); B -= sell

        # Normal B withdrawal (if active and not paused)
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
    # align withdrawal array to the same length for convenience (append 0 at end)
    out['A_withdraw_taken'] = np.append(A_withdraw_taken, 0.0)

    meta = {
        'A_depleted_age': A_dep,
        'B_depleted_age': B_dep,
        'A_end_balance': A_bal[-1],
        'B_end_units': B_units[-1],
        'btc_price_end': btc[-1],
    }
    return out, meta

# -------- Monte Carlo --------
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

# -------- Streamlit UI --------
with st.sidebar:
    st.header("Simulation Inputs")
    current_age     = st.slider("Current age", 18, 100, 53, 1)
    retirement_age  = st.slider("Retirement age", current_age, 100, max(55, current_age), 1)
    final_age       = st.slider("Final age (model horizon)", retirement_age, 120, 120, 1)

    inflation_pct = st.slider("Annual inflation rate (%)", -5.0, 20.0, 2.0, 0.1)
    dollar_debasement_pct = st.slider("Dollar debasement (%)", 0.0, 10.0, 0.0, 0.1)
    include_inflation = st.checkbox("Include inflation adjustment after withdrawals begin", True, key="includeInflation")

    st.markdown("---")
    st.subheader("Portfolio A (USD)")
    A_start_balance = st.slider("Starting balance (USD)", 0, 10_000_000, 1_650_000, 10_000)
    A_cagr_pct      = st.slider("CAGR (%)", -50.0, 50.0, 6.0, 0.1)
    A_vol_pct       = st.slider("Volatility (%)", 0.0, 100.0, 10.0, 0.1)
    A_withdraw_amount = st.slider("A withdrawal (USD/month)", 0, 50_000, 10_000, 100)

    st.markdown("---")
    st.subheader("Portfolio B (Bitcoin)")
    show_b_advanced = True
    B_start_btc = st.slider("Starting BTC (units)", 0.0, 100.0, 11.1, 0.1)
    btc_spot        = st.slider("BTC spot price (USD)", 0, 1_000_000, 110_000, 1000)
    B_mu_start_pct  = st.slider("BTC CAGR start (%)", -50.0, 200.0, 25.0, 0.1)
    B_withdraw_base = st.slider("B withdrawal (USD/month)", 0, 50_000, 10_000, 50)
    # Advanced (hidden unless toggled on)
    with st.expander('Advanced Bitcoin options', expanded=True):
        if show_b_advanced:
            st.markdown('---')
            with st.expander('BTC Pause Rule', expanded=False):
                pause_enabled = st.checkbox('Enable BTC pause rule', False, key='btcPause')
                thr_pct = st.slider('Pause if BTC < % of recent high', 50, 100, 80, 1)
                pause_window_m = st.slider('Recent-high lookback (months)', 1, 60, 24, 1)
            pause_threshold = thr_pct / 100.0
            st.session_state['pause_enabled'] = pause_enabled
            st.session_state['pause_threshold'] = pause_threshold
            st.session_state['pause_window_m'] = pause_window_m
            st.markdown("**CAGR taper controls**")
            B_mu_taper_start_age = st.slider("BTC CAGR taper starts at age", float(current_age), float(final_age), 65.0, 0.5)
            B_mu_end_age = st.slider("BTC CAGR taper stops at age", retirement_age, 120, max(80, retirement_age), 1)
            B_mu_end_pct = st.slider("BTC CAGR taper to (%)", -50.0, 200.0, 10.0, 0.1)
            
            st.markdown("**Volatility taper controls**")
            B_sig_taper_start_age = st.slider("BTC Vol taper starts at age", float(current_age), float(final_age), 65.0, 0.5)
            B_sig_end_age = st.slider("BTC Vol taper stops at age", retirement_age, 120, max(80, retirement_age), 1)
            B_sig_start_pct = st.slider("BTC Vol start (%)", 0.0, 200.0, 25.0, 0.1)
            B_sig_end_pct = st.slider("BTC Vol at taper stop age (%)", 0.0, 200.0, 10.0, 0.1)

            # Additional advanced BTC controls
            B_withdraw_start_age = st.slider("B withdrawal START age", retirement_age, 120, 120, 1)
            B_start_earliest = st.checkbox("Start BTC withdrawals at A depletion OR start age (whichever comes first)", True)
        else:
            # Defaults used when advanced settings are hidden
            pause_enabled = st.session_state.get('pause_enabled', False)
            pause_threshold = st.session_state.get('pause_threshold', 0.8)
            pause_window_m = st.session_state.get('pause_window_m', 24)
            B_mu_end_pct = 10.0
            B_mu_end_age = float(max(80, retirement_age))
            B_sig_start_pct = 25.0
            B_sig_end_pct = 10.0
            B_sig_end_age = float(max(80, retirement_age))
            B_withdraw_start_age = 120.0
            B_start_earliest = True

    st.markdown("---")

    with st.expander("⚙️ Advanced Display Options", expanded=False):
        n_sims = st.slider("Number of Monte Carlo simulations", 100, 5000, 1000, 100, key="sidebar_n_sims_slider")
        seed   = st.slider("Random seed", 0, 10_000_000, 42, 1, key="sidebar_seed_slider")

# ===== Parameter Freeze & Run Control =====
# --- Safety defaults for taper-start ages (sliders may be hidden) ---
try:
    B_mu_taper_start_age
except NameError:
    B_mu_taper_start_age = 65.0
try:
    B_sig_taper_start_age
except NameError:
    B_sig_taper_start_age = 65.0
# --------------------------------------------------------------------

# Snapshot of current widget inputs (to freeze on Run)
_current_params = dict(
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
    B_mu_taper_start_age=float(B_mu_taper_start_age),
    B_sig_start=pct_to_decimal(B_sig_start_pct),
    B_sig_end=pct_to_decimal(B_sig_end_pct),
    B_sig_end_age=float(B_sig_end_age),
    B_sig_taper_start_age=float(B_sig_taper_start_age),
    inflation=pct_to_decimal(inflation_pct),
    include_inflation=bool(include_inflation),
    B_withdraw_base=float(B_withdraw_base),
    B_withdraw_start_age=float(B_withdraw_start_age),
    B_start_earliest=bool(B_start_earliest),
    pause_enabled=bool(st.session_state.get('pause_enabled', False)),
    pause_threshold=float(st.session_state.get('pause_threshold', 0.8)),
    pause_window_m=int(st.session_state.get('pause_window_m', 24)),
)

# If user clicked Run this rerun, freeze params & invalidate previous results
if st.session_state.pop('_trigger_run', False):
    st.session_state['_frozen_params'] = _current_params
    st.session_state['_run_version'] = st.session_state.get('_run_version', 0) + 1
    st.session_state.pop('_results_cache', None)

# Gate: don't run the heavy simulation until we have frozen params
if not st.session_state.get('_run_sim', False) or '_frozen_params' not in st.session_state:
    st.info("Adjust inputs in the sidebar, then click **Run simulation** to generate results.")
    st.stop()

# Use frozen params and compute results only if not cached
params = st.session_state['_frozen_params']
if '_results_cache' not in st.session_state:
    st.session_state['_results_cache'] = run_monte_carlo(params, n_sims=int(n_sims), seed=int(seed))
results = st.session_state['_results_cache']

# Overall success rate (not both depleted by final age)
final_age_value = float(final_age)
A_deps = np.array(results['A_dep_ages'], dtype=float)
B_deps = np.array(results['B_dep_ages'], dtype=float)
success_mask = ~((A_deps <= final_age_value) & (B_deps <= final_age_value))
success_rate = float(np.nanmean(success_mask)) if success_mask.size else float('nan')

# ===== Framework (Always Visible) =====
st.markdown(
    """
    <div style="max-width:864px;margin:0 auto 24px auto;">
      <div style="background:#ffffff;color:#111111;border-radius:12px;padding:20px 28px;border:1px solid rgba(0,0,0,0.08);box-shadow:0 8px 24px rgba(0,0,0,0.15);">
        <h2 style="text-align:center;margin:0 0 6px 0;">The Framework of this Monte Carlo Simulation</h2>
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
st.markdown("### Probability of Success")

col1, col2 = st.columns([2, 3])
with col1:
    target_pct = st.slider("Success target (%)", 50, 100, 90, 1,
                           help="Green if success ≥ target; red otherwise.",
                           key="success_target")
with col2:
    delta_val = (success_rate * 100.0) - target_pct
    delta_str = f"{delta_val:+.1f}% vs target"
    delta_color = "normal" if delta_val >= 0 else "inverse"
    st.metric(label="Retirement plan success",
              value=f"{success_rate*100:.1f}%",
              delta=delta_str,
              delta_color=delta_color)
st.caption("Success = at least one portfolio (USD or Bitcoin) remains funded through the final age.")
# ==========================================================

ages = results['ages']
A10, A50, A90 = results['A_q10'], results['A_q50'], results['A_q90']
B10, B50, B90 = results['B_q10'], results['B_q50'], results['B_q90']
BV10, BV50, BV90 = results['BV_q10'], results['BV_q50'], results['BV_q90']

def prob_depleted_by_age(ages_grid, dep_ages): return np.array([np.nanmean(dep_ages <= a) for a in ages_grid])
probA = prob_depleted_by_age(ages, results['A_dep_ages']); probB = prob_depleted_by_age(ages, results['B_dep_ages'])
def crossing_age(prob_series, ages_grid, target=0.5):
    idx = np.where(prob_series >= target)[0]; return float(ages_grid[idx[0]]) if len(idx) else None
A_cross50 = crossing_age(probA, ages, 0.5); B_cross50 = crossing_age(probB, ages, 0.5)

CHART_W_IN, CHART_H_IN, DPI = 6.4, 3.6, 96
def draw_chart(x, y_mid, y_lo=None, y_hi=None, title="", y_label="", dep_age=None, labels=()):
    # +20% scale from prior 50% version
    fig, ax = plt.subplots(figsize=(5.76, 3.6), dpi=DPI)
    fig.set_facecolor("white")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.85, bottom=0.26)

    # Font sizes +20%
    TITLE_SIZE  = 14.4
    LABEL_SIZE  = 10.08
    TICK_SIZE   = 8.64
    LEGEND_SIZE = 8.64

    ax.plot(x, y_mid, label=labels[0] if labels else "median")
    if y_lo is not None and y_hi is not None:
        ax.fill_between(x, y_lo, y_hi, alpha=0.2, label=labels[1] if len(labels)>1 else "10–90%")
    if dep_age is not None:
        ax.axvline(dep_age, linestyle='--', label=labels[2] if len(labels)>2 else "50% depleted")

    ax.set_xlabel("Age", fontsize=LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    if y_label.lower().startswith("usd"):
        ax.yaxis.set_major_formatter(usd_formatter())
    ax.set_title(title, fontsize=TITLE_SIZE)

    # Subtle dashed gridlines
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    # Compact legend (inside plot, upper-left)
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h, l, loc="lower left", frameon=False, ncol=1, fontsize=9)

    return fig

# --- Charts stacked vertically in expanders for consistent sizing ---

# ===== Optional detailed portfolio strategy description =====
show_strategy_details = st.checkbox("Show detailed portfolio strategy description", value=False)
if show_strategy_details:
    st.markdown(f"""
### 📘 Portfolio Strategy Overview
This simulation models two complementary investment approaches:

- **Portfolio A – Cathay Pacific Provident Fund**  
  Represents a traditional, diversified retirement account.  
  **Key Inputs**
  - *Starting balance (USD):* Defines initial capital.  
  - *Expected annual CAGR (%):* Average return estimate.  
  - *Volatility (%):* Year‑to‑year variability of returns.  
  - *Monthly withdrawal (USD):* Amount drawn starting at retirement.  
  - *Inflation:* Applied **only after withdrawals begin** (if enabled) to maintain purchasing power.

- **Portfolio B – Bitcoin**  
  A Bitcoin allocation whose value is units × price along simulated paths.  
  **Key Inputs**
  - *Starting BTC units:* Total BTC held at start.  
  - *BTC spot price (USD):* Current price used to compute USD value.  
  - *Expected annual growth (%):* Assumed average appreciation.  
  - *Volatility (%):* Reflects BTC price swings.  
  - *Withdrawal start age:* When BTC begins supplementing income.

### 🔧 Current Variable Inputs
- Current age: **{current_age}**
- Retirement age: **{retirement_age}**
- Final age: **{final_age}**
- Inflation rate: **{inflation_pct:.1f}%**
- Inflation applied after withdrawals: **{include_inflation}**
- A start balance: **${A_start_balance:,.0f}**
- A CAGR / Vol: **{A_cagr_pct:.1f}% / {A_vol_pct:.1f}%**
- A withdrawal (per month): **${A_withdraw_amount:,.0f}**
- B starting BTC units: **{B_start_btc:.4f}**
- BTC spot price: **${btc_spot:,.0f}**
- B growth parameters: **μ/σ path** (configure in Advanced)
- B withdrawal start age: **{B_withdraw_start_age}**
""")
    st.info("Tip: adjust the inputs in the sidebar and re-open this section to see how assumptions flow into the strategy.")

left_col, mid_col, right_col = st.columns([1,2,1])
with mid_col:

    # Always visible, centered chart for Portfolio A
    showA = st.session_state.get("bandsA", True)
    fig = draw_chart(ages, A50, A10 if showA else None, A90 if showA else None, "",
        "USD balance",
        A_cross50,
        ("A (median)", "A 10–90%", "50% of sims depleted")
    )
    # Force Portfolio A legend to upper-left, smaller font with white box
    axA = fig.axes[0]
    axA.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.75, edgecolor="none", ncol=1, fontsize=8, handletextpad=0.5, labelspacing=0.2)
    show_centered_fig(fig, 518, 292)
left_col, mid_col, right_col = st.columns([1,2,1])
with mid_col:

    # Always visible, centered chart for Portfolio B
    showB = st.session_state.get("bandsB", True)
    fig = draw_chart(
        ages,
        B50,
        B10 if showB else None,
        B90 if showB else None,
        "Portfolio B — Median units",
        "BTC units",
        B_cross50,
        ("B (median units)", "B 10–90%", "50% of sims depleted")
    )
    show_centered_fig(fig, 518, 292)
    
    
left_col, mid_col, right_col = st.columns([1,2,1])
with mid_col:

    figBV, axBV = plt.subplots(figsize=(5.18, 2.92), dpi=DPI)
    figBV.subplots_adjust(bottom=0.25)
    axBV.plot(ages, BV50, label="B value (median)")
    axBV.fill_between(ages, BV10, BV90, alpha=0.2, label="B value 10–90%")
    axBV.set_xlabel("Age")
    axBV.set_ylabel("USD value")
    axBV.yaxis.set_major_formatter(usd_formatter())
    axBV.set_title("Portfolio B — USD value over time (units × price)")
    h, l = axBV.get_legend_handles_labels()
    axBV.legend(h, l, loc="lower left", frameon=False, ncol=1, fontsize=9)
    show_centered_fig(figBV, 518, 292)
    
st.subheader("Median End Balances at Final Age")

A_end_med = float(np.median(results['A_end_balances'])); B_end_med_units = float(np.median(results['B_end_units'])); B_end_med_usd = float(np.median(results['B_end_usd']))
show_btc_end_percentiles = st.checkbox("Show 10th/90th percentiles for BTC end balances", False)
if show_btc_end_percentiles:
    B_end_units_10 = float(np.percentile(results['B_end_units'], 10)); B_end_units_90 = float(np.percentile(results['B_end_units'], 90))
    B_end_usd_10 = float(np.percentile(results['B_end_usd'], 10)); B_end_usd_90 = float(np.percentile(results['B_end_usd'], 90))
    tbl = pd.DataFrame({"Portfolio":["A (USD)","B (BTC units)","B (BTC units)","B (BTC units)","B (USD equiv)","B (USD equiv)","B (USD equiv)"],
                        "Statistic":["Median","Median","10th","90th","Median","10th","90th"],
                        "Value":[f"${A_end_med:,.0f}",f"{B_end_med_units:,.4f} BTC",f"{B_end_units_10:,.4f} BTC",f"{B_end_units_90:,.4f} BTC",
                                 f"${B_end_med_usd:,.0f}",f"${B_end_usd_10:,.0f}",f"${B_end_usd_90:,.0f}"]})
else:
    tbl = pd.DataFrame({"Portfolio":["A (USD)","B (BTC units)","B (USD equiv)"],
                        "Statistic":["Median","Median","Median"],
                        "Value":[f"${A_end_med:,.0f}",f"{B_end_med_units:,.4f} BTC",f"${B_end_med_usd:,.0f}"]})
st.table(tbl)

# -------- Optional 5-year snapshot --------
# Purchasing power adjustment for 5-year snapshot
_annual_debase = dollar_debasement_pct / 100.0
try:
    _df = snapshot_df
    if 'Year' in _df.columns and any(col in _df.columns for col in ['Balance','Portfolio','End Balance','Ending Balance','Net Worth','FV']):
        _valcol = next(col for col in ['Balance','Portfolio','End Balance','Ending Balance','Net Worth','FV'] if col in _df.columns)
        _yrs = _df['Year'] - _df['Year'].iloc[0]
        snapshot_df['Adj. purchasing power'] = _df[_valcol] / (1.0 + _annual_debase) ** _yrs
except Exception as _e:
    pass
st.subheader("5-Year Snapshot Table (optional)")
show_5y = st.checkbox("Show 5-year portfolio balances table", True, key="fiveYrShow")
include_5y_pct = st.checkbox("Include 10th/90th percentiles (5-year table)", False, key="fiveYrPct")
if show_5y:
    step = 5 * 12; idxs = [0] + list(range(step, len(ages), step))
    if idxs[-1] != len(ages) - 1: idxs.append(len(ages) - 1)
    rows = []
    for i in idxs:
        age = ages[i]
        row = {"Age": f"{age:.0f}", "A (USD, median)": f"${A50[i]:,.0f}", "B (BTC units, median)": f"{B50[i]:,.4f} BTC", "B (USD value, median)": f"${BV50[i]:,.0f}",  "B (USD value, adj. PP)": f"${(BV50[i] / (1.0 + dollar_debasement_pct/100.0) ** (age - ages[0])):,.0f}"}
        if include_5y_pct:
            row.update({"A p10": f"${A10[i]:,.0f}", "A p90": f"${A90[i]:,.0f}",
                        "B units p10": f"{B10[i]:,.4f} BTC", "B units p90": f"{B90[i]:,.4f} BTC",
                        "B USD p10": f"${BV10[i]:,.0f}", "B USD p90": f"${BV90[i]:,.0f}"})
        rows.append(row)
    five_table = pd.DataFrame(rows); st.table(five_table)

# -------- Year-by-Year table for Portfolio A (cut after depletion) --------

show_year_table = st.checkbox("Show Portfolio A year-by-year table", True, key="yearAshow")
include_year_pct = st.checkbox("Include 10th/90th percentiles (year-by-year table)", False, key="yearApct")

if show_year_table:
    year_df = results['year_summary_A'].copy()

    # Stop showing rows after the first year where the *median* year-end balance hits zero
    depl_mask = year_df["A Year-end Balance (median USD)"] <= 1e-9
    if depl_mask.any():
        cut_idx = int(np.argmax(depl_mask))  # first True index
        year_df = year_df.iloc[:cut_idx + 1]  # include depletion year, exclude later years

    # Format currency columns
    def fmt_usd_series(s): return s.apply(lambda v: f"${v:,.0f}")
    if include_year_pct:
        display_df = pd.DataFrame({
            "Age": year_df["Age (year)"],
            "A Withdrawal (median)": fmt_usd_series(year_df["A Withdrawal (median, annual USD)"]),
            "A Withdrawal p10": fmt_usd_series(year_df["A Withdrawal p10"]),
            "A Withdrawal p90": fmt_usd_series(year_df["A Withdrawal p90"]),
            "A Year-end Balance (median)": fmt_usd_series(year_df["A Year-end Balance (median USD)"]),
            "A Balance p10": fmt_usd_series(year_df["A Balance p10"]),
            "A Balance p90": fmt_usd_series(year_df["A Balance p90"]),
        })
    else:
        display_df = pd.DataFrame({
            "Age": year_df["Age (year)"],
            "A Withdrawal (median)": fmt_usd_series(year_df["A Withdrawal (median, annual USD)"]),
            "A Year-end Balance (median)": fmt_usd_series(year_df["A Year-end Balance (median USD)"]),
        })

    st.table(display_df)

# -------- Depletion summary --------
def fmt_age(x): return f"{x:.1f}" if x is not None else "—"
A_dep_share = float(np.mean(~np.isnan(np.array(results['A_dep_ages'])))); B_dep_share = float(np.mean(~np.isnan(np.array(results['B_dep_ages']))))

st.markdown("---")
st.subheader("Depletion Summary by Final Age")
dep_tbl = pd.DataFrame({
    "Portfolio": ["A (USD)", "B (BTC units)"],
    "% sims depleted by final age": [f"{A_dep_share*100:.1f}%", f"{B_dep_share*100:.1f}%"],
    "Age when 50% of ALL sims are depleted": [f"{fmt_age(A_cross50)}" if A_cross50 is not None else "—",
                                              f"{fmt_age(B_cross50)}" if B_cross50 is not None else "—"],
})
st.table(dep_tbl)

# ================= Executive PDF Export =================
def _make_exec_pdf(params, results, context):
    """Return PDF bytes for an executive summary."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except Exception as e:
        import io
        # Return a small PDF-like notice if reportlab isn't available
        bio = io.BytesIO()
        try:
            from reportlab.pdfgen import canvas  # second attempt if lazy import helps
        except Exception:
            return b""
    import io, datetime as _dt

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    left = 0.9*inch
    y = H - 1.0*inch

    def line(txt, font="Helvetica", size=10, dy=14):
        nonlocal y
        c.setFont(font, size)
        c.drawString(left, y, str(txt))
        y -= dy

    # ---- Title ----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, "CHEAT CODE — Executive Summary")
    y -= 22
    c.setFont("Helvetica", 10)
    # Executive framework under title (if checkbox selected)
    try:
        import streamlit as _st
        if _st.session_state.get('execFramework', False):
            y = _pdf_exec_summary(c, x=left, y=y-10)
    except Exception:
        pass
    # ---- Inputs ----
    line("Inputs", "Helvetica-Bold", 12, 18)
    line(f"Current age: {context.get('current_age')}")
    line(f"Retirement age: {context.get('retirement_age')}")
    line(f"Final age: {context.get('final_age')}")
    line(f"Inflation (annual): {round(context.get('inflation_pct', 0.0), 2)}%  |  Include inflation after withdrawals: {context.get('include_inflation')}")
    line(f"A Start Balance (USD): {context.get('A_start_balance'):,.0f}")
    line(f"A CAGR: {round(context.get('A_cagr_pct', 0.0), 2)}%  |  A Vol: {round(context.get('A_vol_pct', 0.0), 2)}%")
    line(f"A Withdrawal (USD/month): {context.get('A_withdraw_amount'):,.0f}")
    line(f"B Starting BTC (units): {context.get('B_start_btc'):,.4f}  |  BTC Spot (USD): {context.get('btc_spot'):,.0f}")

    y -= 6
    # ---- Outcomes ----
    line("Key Outcomes", "Helvetica-Bold", 12, 18)

    ages = results['ages']
    final_age_value = float(ages[-1])
    A_end_med = float(np.median(results['A_end_balances'])) if 'A_end_balances' in results else None
    B_end_med_units = float(np.median(results['B_end_units'])) if 'B_end_units' in results else None
    B_end_med_usd = float(np.median(results['B_end_usd'])) if 'B_end_usd' in results else None

    def fmt(v, kind="usd"):
        if v is None: return "—"
        if kind == "usd": return f"${v:,.0f}"
        if kind == "btc": return f"{v:,.4f} BTC"
        if kind == "age": return f"{v:.1f}"
        return str(v)

    # Probability both NOT depleted by final age (success)
    A_deps = np.array(results['A_dep_ages'], dtype=float)
    B_deps = np.array(results['B_dep_ages'], dtype=float)
    success_mask = ~((A_deps <= final_age_value) & (B_deps <= final_age_value))
    success_rate = float(np.nanmean(success_mask)) if success_mask.size else float("nan")

    line(f"Success probability (not both depleted by final age): {success_rate*100:.1f}%")
    line(f"Median end balance A (USD): {fmt(A_end_med, 'usd')}")
    line(f"Median end units B (BTC): {fmt(B_end_med_units, 'btc')}")
    line(f"Median end value B (USD): {fmt(B_end_med_usd, 'usd')}")

    # Depletion medians
    def median_dep(dep_arr, final_age):
        arr = np.array(dep_arr, dtype=float)
        # Treat values that are NaN or >= final_age as 'never depleted'
        arr = arr[np.isfinite(arr) & (arr < float(final_age))]
        return float(np.median(arr)) if arr.size else None

    A_dep_med = median_dep(results['A_dep_ages'], final_age_value)
    B_dep_med = median_dep(results['B_dep_ages'], final_age_value)

    line(f"Median depletion age A: {fmt(A_dep_med, 'age') if A_dep_med is not None else 'Never'}")
    line(f"Median depletion age B: {fmt(B_dep_med, 'age') if B_dep_med is not None else 'Never'}")

    # ---- Footer ----
    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(left, max(0.7*inch, y), "Note: Inflation adjustments are applied only once withdrawals commence (if enabled).")
    c.showPage()
    c.save()
    return buf.getvalue()

# Export UI
st.markdown("---")
with st.expander("📄 Export", expanded=False):
    st.subheader("Executive PDF")
    # Capture current inputs for the PDF
    context_inputs = {
        "current_age": float(current_age),
        "retirement_age": float(retirement_age),
        "final_age": float(final_age),
        "inflation_pct": float(inflation_pct),
        "include_inflation": bool(include_inflation) if 'include_inflation' in locals() else True,
        "A_start_balance": float(A_start_balance),
        "A_cagr_pct": float(A_cagr_pct),
        "A_vol_pct": float(A_vol_pct),
        "A_withdraw_amount": float(A_withdraw_amount),
        "B_start_btc": float(B_start_btc),
        "btc_spot": float(btc_spot),
    }

    if st.button("Create Executive PDF"):
        try:
            pdf_bytes = _make_exec_pdf(params, results, context_inputs)
            if pdf_bytes:
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label="Download Executive PDF",
                    data=pdf_bytes,
                    file_name=f"CHEAT_CODE_Executive_Summary_{ts}.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("ReportLab is not available in this environment. Please `pip install reportlab` and try again.")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
