import math
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cessna 140A Cost Dashboard", page_icon="🛩️", layout="wide")
st.title("Cessna 140A — Operating Cost Dashboard (0-dec)")

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
    hours_remaining = max(tbo_hours - tsmoh_hours, 1)
    engine_reserve_per_hr = overhaul_cost / hours_remaining
    oil_changes = int(hours_per_year // oil_interval_hr)
    oil_qt_total = oil_changes * oil_per_change_qt + hours_per_year * oil_consumption_qt_hr

    fuel_annual = hours_per_year * gph * fuel_price
    oil_annual = oil_qt_total * oil_price_per_qt
    engine_reserve_annual = hours_per_year * engine_reserve_per_hr
    misc_maint_annual = hours_per_year * misc_per_hr

    base_total = fuel_annual + oil_annual + engine_reserve_annual + misc_maint_annual + annual_inspection + annual_tiedown

    def row(name, ins_cost):
        total_annual = base_total + ins_cost
        per_hour = total_annual / hours_per_year
        monthly_total = total_annual / 12
        monthly_member = monthly_total / owners_count
        return {
            "Insurance Plan": name,
            "Total Annual (USD)": r0(total_annual),
            "Per Hour (USD/hr)": r0(per_hour),
            "Monthly Total (USD/mo)": r0(monthly_total),
            "Monthly per Member (USD/mo)": r0(monthly_member),
        }

    df = pd.DataFrame([
        row("Full Coverage", insurance_full),
        row("Liability Only", insurance_liability),
    ])

    kpis = {
        "Hours Remaining to TBO": r0(hours_remaining),
        "Engine Reserve $/hr": r0(engine_reserve_per_hr),
        "Base Total (no insurance)": r0(base_total),
    }
    return df, kpis

# ---------- UI ----------
with st.sidebar:
    st.header("Inputs")
    hours_per_year      = st.number_input("Flight hours / year", 1, 2000, 150, 1)
    gph                 = st.number_input("Fuel burn (GPH)", 0.1, 30.0, 4.8, 0.1)
    fuel_price          = st.number_input("Fuel price ($/gal)", 0.0, 20.0, 5.22, 0.01)

    st.subheader("Oil")
    oil_interval_hr     = st.number_input("Oil change interval (hrs)", 10, 200, 25, 5)
    oil_per_change_qt   = st.number_input("Oil per change (qt)", 1.0, 12.0, 5.0, 0.5)
    oil_consumption_qt_hr = st.number_input("Oil consumption (qt/hr)", 0.0, 2.0, 0.167, 0.001)
    oil_price_per_qt    = st.number_input("Oil price ($/qt)", 0.0, 100.0, 12.47, 0.01)

    st.subheader("Engine")
    tbo_hours           = st.number_input("TBO (hrs)", 500, 4000, 1800, 50)
    tsmoh_hours         = st.number_input("TSMOH (hrs since OH)", 0, 4000, 400, 10)
    overhaul_cost       = st.number_input("Overhaul cost ($)", 0.0, 200000.0, 15000.0, 100.0)

    st.subheader("Fixed costs")
    annual_inspection   = st.number_input("Annual inspection ($)", 0.0, 50000.0, 2400.0, 50.0)
    annual_tiedown      = st.number_input("Annual tie-down ($)", 0.0, 50000.0, 1200.0, 50.0)
    insurance_full      = st.number_input("Insurance (Full) ($)", 0.0, 50000.0, 3433.0, 10.0)
    insurance_liability = st.number_input("Insurance (Liability) ($)", 0.0, 50000.0, 539.0, 10.0)
    misc_per_hr         = st.number_input("Misc maintenance ($/hr)", 0.0, 500.0, 12.60, 0.10)

    owners_count        = st.number_input("Syndicate members", 1, 20, 3, 1)

# compute
df, kpis = compute_costs(
    hours_per_year, gph, fuel_price,
    oil_interval_hr, oil_per_change_qt, oil_consumption_qt_hr, oil_price_per_qt,
    tbo_hours, tsmoh_hours, overhaul_cost,
    annual_inspection, annual_tiedown,
    insurance_full, insurance_liability,
    misc_per_hr, owners_count
)

# ---------- Output ----------
col_table, col_kpi = st.columns([1.6, 1])
with col_table:
    st.subheader("Scenarios")
    st.dataframe(df, use_container_width=True)

with col_kpi:
    st.subheader("Quick KPIs")
    st.metric("Hours remaining to TBO", f"{kpis['Hours Remaining to TBO']}")
    st.metric("Engine reserve ($/hr)", f"${kpis['Engine Reserve $/hr']}")
    st.metric("Base total (no insurance)", f"${kpis['Base Total (no insurance)']}")

# chart
fig = px.bar(
    df,
    x="Insurance Plan",
    y="Monthly per Member (USD/mo)",
    text="Monthly per Member (USD/mo)",
    title="Per-member Monthly Cost (0-dec)",
)
fig.update_traces(texttemplate="$%{text}", textposition="outside")
fig.update_layout(yaxis_title="USD / month / member", xaxis_title="Insurance Plan", uniformtext_minsize=10)
st.plotly_chart(fig, use_container_width=True)

# export
st.download_button(
    "Download CSV (Scenarios)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="C140A_cost_scenarios_0dec.csv",
    mime="text/csv",
)
st.caption("Tip: change TSMOH to see the engine-reserve $/hr adjust based on hours remaining to TBO.")
