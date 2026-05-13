import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import StringIO

st.set_page_config(page_title="EV Charging Finance Model", layout="wide")

# ─────────────────────────────────────────
# BUNDLED CSV DATA
# ─────────────────────────────────────────

CSV_DATA = """Charger,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039
100,18900,18500,18100,17800,17400,17100,16700,16400,16100,15800,15400,15100,14800,14500
500,63500,62500,61000,60000,59000,57500,56500,55500,54000,53000,52000,51000,50000,49000"""

charger_cost = pd.read_csv(StringIO(CSV_DATA)).set_index("Charger")

# ─────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────

with st.sidebar:
    st.header("Inputs")

    project_year = st.selectbox("Project year", list(range(2026, 2040)), index=0)
    charger_size = st.selectbox("Charger size (kW)", [100, 500], index=1)
    charger_quantity = st.slider("Charger quantity", min_value=1, max_value=50, value=10)
    line_km = st.slider("Grid line extension (km)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    util_hrs = st.slider("Daily utilisation (hrs)", min_value=1.0, max_value=24.0, value=6.0, step=0.5)
    elec_price = st.slider("Electricity sale price ($/kWh)", min_value=0.05, max_value=0.50, value=0.15, step=0.01)

# ─────────────────────────────────────────
# CALCULATION ENGINE
# ─────────────────────────────────────────

def run_model(project_year, charger_size, charger_quantity, line_km, util_hrs, electricity_price_per_kwh):
    utilization = util_hrs / 24

    # CAPEX
    hardware_cost_per_charger = charger_cost.loc[charger_size, str(project_year)]
    total_hardware_cost       = hardware_cost_per_charger * charger_quantity
    total_installation_cost   = total_hardware_cost * 0.15

    # Electrical
    power_need = charger_size * charger_quantity
    project_mw = power_need / 1000

    if project_mw <= 1:
        selected_voltage = 11
    elif project_mw <= 5:
        selected_voltage = 33
    elif project_mw <= 10:
        selected_voltage = 66
    elif project_mw <= 50:
        selected_voltage = 132
    else:
        selected_voltage = 220

    processing_fee = 270 if selected_voltage in [33, 11] else 160

    line_fee_per_km = {11: 7000, 33: 45000, 66: 70000, 132: 94000, 220: 127000}
    line_extension_cost     = line_fee_per_km[selected_voltage] * line_km
    gross_substation        = 2_200_000 if selected_voltage < 220 else 5_700_000
    project_substation_cost = gross_substation * 0.25
    total_electrical_cost   = processing_fee + line_extension_cost + project_substation_cost
    total_financing         = total_hardware_cost + total_installation_cost + total_electrical_cost

    # Financing
    debt_ratio    = 0.80
    interest_rate = 0.09
    loan_years    = 15
    debt_amount   = total_financing * debt_ratio
    down_payment  = total_financing * (1 - debt_ratio)

    annual_debt_service = (
        debt_amount * interest_rate
        / (1 - (1 + interest_rate) ** (-loan_years))
    )

    # Amortisation
    balance = debt_amount
    amort = []
    for yr in range(1, loan_years + 1):
        interest  = balance * interest_rate
        principal = annual_debt_service - interest
        balance   = max(0, balance - principal)
        amort.append({"Year": yr, "Interest": interest, "Principal": principal, "Balance": balance})
    amort_df = pd.DataFrame(amort)
    total_interest_paid = amort_df["Interest"].sum()

    # Opex
    annual_land_lease  = (charger_quantity * 18 * 219) / loan_years
    annual_maint_labor = total_hardware_cost * 0.05
    annual_electricity = utilization * charger_size * charger_quantity * 365 * 24 * 0.12
    annual_opex        = annual_land_lease + annual_maint_labor + annual_electricity

    # Revenue & IRR
    annual_kwh_sold = utilization * charger_size * charger_quantity * 365 * 24
    annual_revenue  = annual_kwh_sold * electricity_price_per_kwh
    cash_flows      = [-down_payment] + [
        annual_revenue - annual_opex - annual_debt_service
        for _ in range(loan_years)
    ]
    project_irr = npf.irr(cash_flows)
    cf_df = pd.DataFrame({"Year": range(loan_years + 1), "Cash Flow": cash_flows})

    return dict(
        hardware_cost_per_charger=hardware_cost_per_charger,
        total_hardware_cost=total_hardware_cost,
        total_installation_cost=total_installation_cost,
        total_electrical_cost=total_electrical_cost,
        total_financing=total_financing,
        selected_voltage=selected_voltage,
        debt_amount=debt_amount,
        down_payment=down_payment,
        annual_debt_service=annual_debt_service,
        total_interest_paid=total_interest_paid,
        amort_df=amort_df,
        annual_land_lease=annual_land_lease,
        annual_maint_labor=annual_maint_labor,
        annual_electricity=annual_electricity,
        annual_opex=annual_opex,
        annual_kwh_sold=annual_kwh_sold,
        annual_revenue=annual_revenue,
        project_irr=project_irr,
        cf_df=cf_df,
        annual_net_cf=annual_revenue - annual_opex - annual_debt_service,
    )


m = run_model(project_year, charger_size, charger_quantity, line_km, util_hrs, elec_price)

# ─────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────

def fmt_m(x, _):
    return f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}k"

def capex_chart(m):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    labels = ["Hardware", "Installation", "Electrical"]
    vals   = [m["total_hardware_cost"], m["total_installation_cost"], m["total_electrical_cost"]]
    bars   = ax.barh(labels, vals, color=["#2ecc71", "#3498db", "#f39c12"], height=0.5)
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_m))
    ax.set_title("CAPEX breakdown", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig

def amort_chart(m):
    df  = m["amort_df"]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.plot(df["Year"], df["Balance"], color="#3498db", lw=2, label="Balance")
    ax.bar(df["Year"], df["Interest"],  color="#f39c12", alpha=0.7, label="Interest", width=0.4)
    ax.bar(df["Year"], df["Principal"], color="#2ecc71", alpha=0.7, label="Principal", width=0.4, align="edge")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_m))
    ax.set_title("Loan amortisation", fontsize=10)
    ax.set_xlabel("Year")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def cashflow_chart(m):
    df  = m["cf_df"]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["Cash Flow"]]
    ax.bar(df["Year"], df["Cash Flow"], color=colors, width=0.6)
    ax.axhline(0, color="grey", lw=1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_m))
    ax.set_title("Annual cash flows", fontsize=10)
    ax.set_xlabel("Year")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def opex_chart(m):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    labels = ["Land lease", "Maintenance & labor", "Electricity"]
    vals   = [m["annual_land_lease"], m["annual_maint_labor"], m["annual_electricity"]]
    ax.pie(vals, labels=labels, autopct="%1.0f%%",
           colors=["#2ecc71", "#3498db", "#f39c12"],
           startangle=140, pctdistance=0.75,
           wedgeprops={"linewidth": 2, "edgecolor": "white"})
    ax.set_title("OpEx mix", fontsize=10)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────

st.title("⚡ EV Charging Finance Model")

# KPIs
irr = m["project_irr"]
irr_str = f"{irr*100:.1f}%" if irr is not None and not np.isnan(irr) else "n/a"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total CAPEX",    f"${m['total_financing']:,.0f}")
k2.metric("Annual revenue", f"${m['annual_revenue']:,.0f}")
k3.metric("Annual OpEx",    f"${m['annual_opex']:,.0f}")
k4.metric("Project IRR",    irr_str)
k5.metric("Annual net CF",  f"${m['annual_net_cf']:,.0f}")

st.divider()

# Detail + charts
left, mid, right = st.columns([1.1, 1, 1])

with left:
    st.subheader("Detail")
    st.text(f"""\
CAPEX & FINANCING
─────────────────────────────────
Hardware / charger   ${m['hardware_cost_per_charger']:>12,.0f}
Total hardware       ${m['total_hardware_cost']:>12,.0f}
Installation (15%)   ${m['total_installation_cost']:>12,.0f}
Electrical ({m['selected_voltage']:>3} kV)   ${m['total_electrical_cost']:>12,.0f}
─────────────────────────────────
Total financing      ${m['total_financing']:>12,.0f}
Debt (80%)           ${m['debt_amount']:>12,.0f}
Down payment (20%)   ${m['down_payment']:>12,.0f}
Annual debt service  ${m['annual_debt_service']:>12,.0f}
Total interest paid  ${m['total_interest_paid']:>12,.0f}

ANNUAL P&L
─────────────────────────────────
Revenue              ${m['annual_revenue']:>12,.0f}
OpEx                 ${m['annual_opex']:>12,.0f}
Debt service         ${m['annual_debt_service']:>12,.0f}
─────────────────────────────────
Net cash flow        ${m['annual_net_cf']:>12,.0f}
""")

with mid:
    st.pyplot(capex_chart(m))
    st.pyplot(cashflow_chart(m))

with right:
    st.pyplot(amort_chart(m))
    st.pyplot(opex_chart(m))

st.divider()

# Cash flow table
st.subheader("Cash flow schedule")
cf_display = m["cf_df"].copy()
cf_display["Cash Flow"] = cf_display["Cash Flow"].map(lambda x: f"${x:,.0f}")
st.dataframe(cf_display, use_container_width=True, hide_index=True)