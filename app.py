# app.py  — Cessna 140A Cost Dashboard (0-dec) with projection & extra charts
import os
import math
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Cessna 140A Cost Dashboard", page_icon="🛩️", layout="wide")
st.title("Cessna 140A — Operating Cost Dashboard
")

# Helpful: shows you're editing the right file and that Streamlit reloaded
import time
st.caption(f"Running: {os.path.abspath(__file__)} | Saved: {time.ctime(os.path.getmtime(__file__))}")

# ---------- helpers ----------
def r0(x):
    try:
        return int(round(float(x), 0))
    except Exception:
        return x

def compute_costs(
    hours_per_year, gph, fuel_price,
    oil_interval_hr, oil_per_change_qt, oil_consumption_qt_hr, oil_price_per_qt,
    tbo_hours, tsmoh_hours, overhaul_cost,
    annual_inspection, annual_tiedown,
    insurance_full, insurance_liability,
    misc_per_hr, owners_count
):
    hours_remaining = max(int(tbo_hours - tsmoh_hours), 1)
    engine_reserve_per_hr = overhaul_cost / hours_remaining
    oil_changes = int(hours_per_year // oil_interval_hr)
    oil_qt_total = oil_changes * oil_per_change_qt + hours_per_year * oil_consumption_qt_hr

    fuel_annual = hours_per_year * gph * fuel_price
    oil_annual = oil_qt_total * oil_price_per_qt
    engine_reserve_annual = hours_per_year * engine_reserve_per_hr
    misc_maint_annual = hours_per_year * misc_per_hr

    base_total = (
        fuel_annual
        + oil_annual
        + engine_reserve_annual
        + misc_maint_annual
        + annual_inspection
        + annual_tiedown
    )

    def row(name, ins_cost):
        total_annual = base_total + ins_cost
        per_hour = total_annual / max(hours_per_year, 1)
        monthly_total = total_annual / 12
        monthly_member = monthly_total / max(owners_count, 1)
        return {
            "Insurance Plan": name,
            "Total Annual (USD)": r0(total_annual),
            "Per Hour (USD/hr)": r0(per_hour),
            "Monthly Total (USD/mo)": r0(monthly_total),
            "Monthly per Member (USD/mo)": r0(monthly_member),
        }

    df = pd.DataFrame(
        [
            row("Full Coverage", insurance_full),
            row("Liability Only", insurance_liability),
        ]
    )

    kpis = {
        "Hours Remaining to TBO": r0(hours_remaining),
        "Engine Reserve $/hr": r0(engine_reserve_per_hr),
        "Base Total (no insurance)": r0(base_total),
    }
    return df, kpis

def compute_projection(df: pd.DataFrame, years: int, inflation_rate: float) -> pd.DataFrame:
    """Return a multi-year projection DataFrame with simple annual inflation."""
    base = df.copy()
    full = int(base.loc[base["Insurance Plan"] == "Full Coverage", "Total Annual (USD)"].iloc[0])
    liab = int(base.loc[base["Insurance Plan"] == "Liability Only", "Total Annual (USD)"].iloc[0])

    records = []
    for y in range(1, years + 1):
        inflation_factor = (1 + inflation_rate / 100.0) ** (y - 1)
        records.append(
            {
                "Year": y,
                "Full Coverage": int(round(full * inflation_factor, 0)),
                "Liability Only": int(round(liab * inflation_factor, 0)),
            }
        )
    proj = pd.DataFrame(records)
    proj["Cumulative Full"] = proj["Full Coverage"].cumsum()
    proj["Cumulative Liability"] = proj["Liability Only"].cumsum()
    return proj

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Inputs")
    hours_per_year      = st.number_input("Flight hours / year", 1, 2000, 150, 1)
    gph                 = st.number_input("Fuel burn (GPH)", 0.1, 30.0, 4.8, 0.1)
    fuel_price          = st.number_input("Fuel price ($/gal)", 0.0, 20.0, 5.22, 0.01)

    st.subheader("Oil")
    oil_interval_hr       = st.number_input("Oil change interval (hrs)", 10, 200, 25, 5)
    oil_per_change_qt     = st.number_input("Oil per change (qt)", 1.0, 12.0, 5.0, 0.5)
    oil_consumption_qt_hr = st.number_input("Oil consumption (qt/hr)", 0.0, 2.0, 0.167, 0.001)
    oil_price_per_qt      = st.number_input("Oil price ($/qt)", 0.0, 100.0, 12.47, 0.01)

    st.subheader("Engine")
    tbo_hours     = st.number_input("TBO (hrs)", 500, 4000, 1800, 50)
    tsmoh_hours   = st.number_input("TSMOH (hrs since OH)", 0, 4000, 400, 10)
    overhaul_cost = st.number_input("Overhaul cost ($)", 0.0, 200000.0, 15000.0, 100.0)

    st.subheader("Fixed costs")
    annual_inspection   = st.number_input("Annual inspection ($)", 0.0, 50000.0, 2400.0, 50.0)
    annual_tiedown      = st.number_input("Annual tie-down ($)", 0.0, 50000.0, 1200.0, 50.0)
    insurance_full      = st.number_input("Insurance (Full) ($)", 0.0, 50000.0, 3433.0, 10.0)
    insurance_liability = st.number_input("Insurance (Liability) ($)", 0.0, 50000.0, 539.0, 10.0)
    misc_per_hr         = st.number_input("Misc maintenance ($/hr)", 0.0, 500.0, 12.60, 0.10)

    owners_count        = st.number_input("Syndicate members", 1, 20, 3, 1)

    st.subheader("Projection")
    years          = st.slider("Projection period (years)", 1, 10, 5)
    inflation_rate = st.slider("Annual inflation rate (%)", 0.0, 10.0, 3.0)

# ---------- Compute ----------
df, kpis = compute_costs(
    hours_per_year, gph, fuel_price,
    oil_interval_hr, oil_per_change_qt, oil_consumption_qt_hr, oil_price_per_qt,
    tbo_hours, tsmoh_hours, overhaul_cost,
    annual_inspection, annual_tiedown,
    insurance_full, insurance_liability,
    misc_per_hr, owners_count
)

# ---------- Summary table & KPIs ----------
col_table, col_kpi = st.columns([1.6, 1])
with col_table:
    st.subheader("Scenarios")
    st.dataframe(df, use_container_width=True)
with col_kpi:
    st.subheader("Quick KPIs")
    st.metric("Hours remaining to TBO", f"{kpis['Hours Remaining to TBO']}")
    st.metric("Engine reserve ($/hr)", f"${kpis['Engine Reserve $/hr']}")
    st.metric("Base total (no insurance)", f"${kpis['Base Total (no insurance)']}")

# ---------- Charts (robust) ----------
# Build a defensive copy in case columns are missing
df_plot = df.copy()
if "Per Hour (USD/hr)" not in df_plot.columns:
    if "Total Annual (USD)" in df_plot.columns:
        df_plot["Per Hour (USD/hr)"] = (df_plot["Total Annual (USD)"] / max(hours_per_year, 1)).round(0).astype(int)
    else:
        df_plot["Per Hour (USD/hr)"] = 0

if "Monthly per Member (USD/mo)" not in df_plot.columns:
    if "Total Annual (USD)" in df_plot.columns:
        df_plot["Monthly per Member (USD/mo)"] = (
            df_plot["Total Annual (USD)"] / 12.0 / max(owners_count, 1)
        ).round(0).astype(int)
    else:
        df_plot["Monthly per Member (USD/mo)"] = 0

col1, col2 = st.columns(2)

# Chart 1: Monthly per member
fig_monthly = px.bar(
    df_plot,
    x="Insurance Plan",
    y="Monthly per Member (USD/mo)",
    text="Monthly per Member (USD/mo)",
    title="Per-member Monthly Cost (0-dec)"
)
fig_monthly.update_traces(texttemplate="$%{text}", textposition="outside")
fig_monthly.update_layout(yaxis_title="USD / month / member", xaxis_title="Insurance Plan")

# Chart 2: Operating cost per hour
fig_hourly = px.bar(
    df_plot,
    x="Insurance Plan",
    y="Per Hour (USD/hr)",
    text="Per Hour (USD/hr)",
    title="Operating Cost per Flight Hour (0-dec)"
)
fig_hourly.update_traces(texttemplate="$%{text}", textposition="outside")
fig_hourly.update_layout(yaxis_title="USD / hr", xaxis_title="Insurance Plan")

with col1:
    st.plotly_chart(fig_monthly, use_container_width=True)
with col2:
    st.plotly_chart(fig_hourly, use_container_width=True)

# ---------- Projection + extra analysis ----------
proj = compute_projection(df, years, inflation_rate)

tab1, tab2, tab3 = st.tabs(["📅 5-Year Projection", "📊 Cumulative Cost", "🍩 Cost Breakdown"])
with tab1:
    st.subheader(f"{years}-Year Projection (Inflation {inflation_rate:.1f}%)")
    fig_proj = px.line(proj, x="Year", y=["Full Coverage", "Liability Only"], markers=True,
                       title="Annual Operating Cost Projection")
    fig_proj.update_layout(yaxis_title="USD per year", xaxis_title="Year")
    st.plotly_chart(fig_proj, use_container_width=True)
    st.dataframe(proj, use_container_width=True)
    st.download_button(
        "Download projection CSV",
        data=proj.to_csv(index=False).encode("utf-8"),
        file_name="C140A_projection.csv",
        mime="text/csv",
    )

with tab2:
    fig_cum = px.line(proj, x="Year", y=["Cumulative Full", "Cumulative Liability"],
                      markers=True, title="Cumulative Ownership Cost")
    fig_cum.update_layout(yaxis_title="Cumulative USD", xaxis_title="Year")
    st.plotly_chart(fig_cum, use_container_width=True)

with tab3:
    st.subheader("Cost Breakdown (Current Year — Full Coverage)")
    # rebuild base pieces (unrounded) for pie
    fuel = hours_per_year * gph * fuel_price
    oil_qt_total = (hours_per_year // oil_interval_hr) * oil_per_change_qt + hours_per_year * oil_consumption_qt_hr
    oil = oil_qt_total * oil_price_per_qt
    engine_reserve = hours_per_year * (overhaul_cost / max(int(tbo_hours - tsmoh_hours), 1))
    misc = hours_per_year * misc_per_hr
    pie = pd.DataFrame({
        "Category": ["Fuel", "Oil", "Engine Reserve", "Misc Maint", "Inspection", "Tie-down", "Insurance"],
        "Cost": [fuel, oil, engine_reserve, misc, annual_inspection, annual_tiedown, insurance_full],
    })
    pie["Cost"] = pie["Cost"].round(0).astype(int)
    fig_pie = px.pie(pie, names="Category", values="Cost", title="Cost Breakdown — Full Coverage")
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------- Download current scenarios ----------
st.download_button(
    "Download CSV (Current Scenarios)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="C140A_cost_scenarios_0dec.csv",
    mime="text/csv",
)
