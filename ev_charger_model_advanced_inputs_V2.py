import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(page_title="EV Charger Model — Advanced", layout="wide")

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #f8f9fa; }
  .metric-block { background: #ffffff; border: 1px solid #e9ecef; border-radius: 10px;
                  padding: 16px 20px; margin-bottom: 10px; }
  .metric-label { font-size: 12px; color: #6c757d; font-weight: 500;
                  text-transform: uppercase; letter-spacing: 0.04em; }
  .metric-primary { font-size: 26px; font-weight: 600; color: #111; margin: 2px 0 1px; }
  .metric-secondary { font-size: 12px; color: #adb5bd; }
  .metric-badge { display: inline-block; font-size: 11px; font-weight: 600;
                  padding: 2px 8px; border-radius: 20px; margin-top: 4px; }
  .badge-green { background: #d1fae5; color: #065f46; }
  .badge-amber { background: #fef3c7; color: #92400e; }
  .badge-red   { background: #fee2e2; color: #991b1b; }
  .section-head { font-size: 12px; font-weight: 600; color: #374151;
                  text-transform: uppercase; letter-spacing: 0.05em;
                  margin: 1.2rem 0 0.5rem; border-bottom: 1px solid #e9ecef; padding-bottom: 5px; }
  .override-box { background: #fffbeb; border: 1px solid #fde68a;
                  border-radius: 8px; padding: 10px 12px; margin-top: 6px; }
  .explainer { font-size: 11px; color: #9ca3af; margin-top: -6px; margin-bottom: 6px; }
  .default-tag { font-size: 10px; color: #9ca3af; font-style: italic; }
  h1 { font-size: 22px !important; font-weight: 600 !important; color: #111 !important; }
  .adv-badge { display:inline-block; font-size:10px; font-weight:600; padding:2px 7px;
               border-radius:20px; background:#ede9fe; color:#5b21b6; margin-left:6px;
               vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
USD_TO_INR   = 90
PROJECT_YEARS = 15

CHARGER_COSTS_DEFAULT = {
    50:  {y: v for y, v in zip(range(2026,2040), [12973,12714,12459,12210,11966,11966,11966,11966,11966,11966,11966,11966,11966,11966])},
    100: {y: v for y, v in zip(range(2026,2040), [18900,18500,18100,17800,17400,17100,16700,16400,16100,15800,15400,15100,14800,14500])},
    250: {y: v for y, v in zip(range(2026,2040), [34462,33773,33097,32435,31787,31787,31787,31787,31787,31787,31787,31787,31787,31787])},
    500: {y: v for y, v in zip(range(2026,2040), [63500,62500,61000,60000,59000,57500,56500,55500,54000,53000,52000,51000,50000,49000])},
}

LINE_FEE_PER_KM_DEFAULT = {11: 7000, 33: 45000, 66: 70000, 132: 94000, 220: 127000}
GROSS_SUBSTATION        = {True: 5_700_000, False: 2_200_000}

TARIFF_DATA = {
    "Tamil Nadu":     {"demand_charge": 304.0, "energy_charge": 8.12},
    "Rajasthan":      {"demand_charge": 150.0, "energy_charge": 6.08},
    "Andhra Pradesh": {"demand_charge":   0.0, "energy_charge": 6.70},
    "Karnataka":      {"demand_charge": 200.0, "energy_charge": 4.50},
    "Maharashtra":    {"demand_charge":   0.0, "energy_charge": 8.24},
    "Telangana":      {"demand_charge": 100.0, "energy_charge": 6.00},
    "Delhi":          {"demand_charge":   0.0, "energy_charge": 7.50},
    "Gujarat":        {"demand_charge":   0.0, "energy_charge": 5.50},
    "Uttar Pradesh":  {"demand_charge":   0.0, "energy_charge": 6.90},
    "West Bengal":    {"demand_charge":   0.0, "energy_charge": 5.80},
    "Madhya Pradesh": {"demand_charge":   0.0, "energy_charge": 8.60},
    "Kerala":         {"demand_charge":   0.0, "energy_charge": 10.30},
    "Haryana":        {"demand_charge":   0.0, "energy_charge": 7.10},
    "Jharkhand":      {"demand_charge":   0.0, "energy_charge": 8.80},
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def fmt_inr(val, decimals=0):
    if abs(val) >= 1e7:
        return f"₹{val/1e7:.{max(decimals,1)}f} Cr"
    elif abs(val) >= 1e5:
        return f"₹{val/1e5:.{max(decimals,1)}f} L"
    return f"₹{val:,.{decimals}f}"

def fmt_usd(val):
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:.1f}K"
    return f"${val:,.0f}"

def calc_irr(cashflows):
    cf = np.array(cashflows, dtype=float)
    signs = np.sign(cf[cf != 0])
    if len(np.unique(signs)) < 2:
        return None
    rate = 0.3
    for _ in range(1000):
        t   = np.arange(len(cf))
        pv  = np.sum(cf / (1 + rate) ** t)
        dpv = np.sum(-t * cf / (1 + rate) ** (t + 1))
        if abs(dpv) < 1e-12:
            break
        new_rate = rate - pv / dpv
        if abs(new_rate - rate) < 1e-8:
            rate = new_rate
            break
        rate = max(new_rate, -0.999)
    if not (-0.999 < rate < 100):
        return None
    return rate

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Advanced inputs")
    st.markdown('<span class="adv-badge">Advanced</span>', unsafe_allow_html=True)

    # ── Project config ──
    st.markdown('<div class="section-head">Project configuration</div>', unsafe_allow_html=True)

    charger_size = st.selectbox(
        "Charger size (kW)",
        options=[50, 100, 250, 500], index=3,
        format_func=lambda x: {50:"50 kW", 100:"100 kW", 250:"250 kW", 500:"500 kW"}[x]
    )
    selected_state = st.selectbox(
        "State", options=sorted(TARIFF_DATA.keys()),
        index=sorted(TARIFF_DATA.keys()).index("Rajasthan")
    )
    charger_quantity = st.slider("Number of chargers", 1, 50, 10, 1)
    daily_util_hrs   = st.slider("Daily utilization (hrs/day)", 1.0, 20.0, 5.0, 0.5)
    st.markdown('<div class="explainer">Average hours per charger actively dispensing power per day.</div>', unsafe_allow_html=True)

    # ── Hardware cost ──
    st.markdown('<div class="section-head">Hardware cost</div>', unsafe_allow_html=True)
    default_hw_usd = CHARGER_COSTS_DEFAULT[charger_size][2026]
    override_hw = st.checkbox("Override hardware cost per charger", value=False)
    if override_hw:
        hw_cost_per_charger_usd = st.number_input(
            "Cost per charger ($)", min_value=1000, max_value=500000,
            value=default_hw_usd, step=500,
        )
        st.markdown(f'<div class="default-tag">Model default: ${default_hw_usd:,}</div>', unsafe_allow_html=True)
    else:
        hw_cost_per_charger_usd = default_hw_usd
        st.markdown(f'<div class="explainer">Using model default: ${default_hw_usd:,} / charger</div>', unsafe_allow_html=True)

    # ── Installation cost ──
    st.markdown('<div class="section-head">Installation cost</div>', unsafe_allow_html=True)
    override_install = st.checkbox("Override installation cost", value=False)
    if override_install:
        install_pct = st.number_input(
            "Installation cost (% of hardware)", min_value=0.0, max_value=100.0,
            value=15.0, step=1.0, format="%.1f"
        )
        st.markdown('<div class="default-tag">Model default: 15% of hardware</div>', unsafe_allow_html=True)
    else:
        install_pct = 15.0
        st.markdown('<div class="explainer">Using model default: 15% of hardware cost</div>', unsafe_allow_html=True)

    # ── Grid connection ──
    st.markdown('<div class="section-head">Grid connection</div>', unsafe_allow_html=True)
    line_km = st.slider("Grid line extension (km)", 0.0, 20.0, 1.0, 0.5)
    st.markdown('<div class="explainer">Distance to nearest transformer or substation.</div>', unsafe_allow_html=True)

    override_line = st.checkbox("Override line extension cost per km", value=False)
    if override_line:
        line_cost_per_km_usd = st.number_input(
            "Line extension cost ($/km)", min_value=0, max_value=500000,
            value=45000, step=1000
        )
        st.markdown('<div class="default-tag">Auto-default: voltage-tier based (e.g. $45K/km at 33 kV)</div>', unsafe_allow_html=True)
    else:
        line_cost_per_km_usd = None  # resolved in model

    override_proc = st.checkbox("Override processing fee", value=False)
    if override_proc:
        processing_fee_usd = st.number_input(
            "Processing fee ($)", min_value=0, max_value=100000,
            value=270, step=10
        )
        st.markdown('<div class="default-tag">Model default: $160–$270 depending on voltage</div>', unsafe_allow_html=True)
    else:
        processing_fee_usd = None  # resolved in model

    # ── Substation ──
    st.markdown('<div class="section-head">Substation</div>', unsafe_allow_html=True)
    include_substation = st.checkbox("Include substation cost share", value=True)
    if include_substation:
        substation_share_pct = st.slider(
            "Project share of substation cost (%)", 0, 100, 25, 5
        )
        st.markdown('<div class="explainer">Gross substation cost: $2.2M (≤132 kV) or $5.7M (220 kV). Your share is apportioned here.</div>', unsafe_allow_html=True)
    else:
        substation_share_pct = 0

    # ── Land cost ──
    st.markdown('<div class="section-head">Land lease</div>', unsafe_allow_html=True)
    override_land = st.checkbox("Override land cost per m²", value=False)
    if override_land:
        land_cost_per_m2_usd = st.number_input(
            "Land cost ($/m²)", min_value=1, max_value=5000,
            value=219, step=10
        )
        land_area_per_charger = st.number_input(
            "Area per charger (m²)", min_value=5, max_value=200,
            value=18, step=1
        )
        st.markdown('<div class="default-tag">Model defaults: $219/m², 18 m²/charger</div>', unsafe_allow_html=True)
    else:
        land_cost_per_m2_usd    = 219
        land_area_per_charger   = 18
        st.markdown('<div class="explainer">Using model defaults: $219/m², 18 m²/charger</div>', unsafe_allow_html=True)

    # ── Maintenance ──
    st.markdown('<div class="section-head">Maintenance & labour</div>', unsafe_allow_html=True)
    override_maint = st.checkbox("Override maintenance cost", value=False)
    if override_maint:
        maint_pct = st.number_input(
            "Annual maintenance (% of hardware cost)", min_value=0.0, max_value=30.0,
            value=5.0, step=0.5, format="%.1f"
        )
        st.markdown('<div class="default-tag">Model default: 5% of hardware cost p.a.</div>', unsafe_allow_html=True)
    else:
        maint_pct = 5.0
        st.markdown('<div class="explainer">Using model default: 5% of hardware cost p.a.</div>', unsafe_allow_html=True)

    # ── Debt structure ──
    st.markdown('<div class="section-head">Debt structure</div>', unsafe_allow_html=True)
    debt_ratio    = st.slider("Debt ratio (%)", 0, 100, 80, 5) / 100
    interest_rate = st.slider("Interest rate (%)", 1.0, 20.0, 9.0, 0.5) / 100
    loan_years    = st.slider("Loan term (years)", 5, 25, 15, 1)
    st.markdown(f'<div class="explainer">Equity = {100-int(debt_ratio*100)}% of capex · {int(interest_rate*100)}% p.a. · {loan_years}-yr amortisation</div>', unsafe_allow_html=True)

    # ── Pricing ──
    st.markdown('<div class="section-head">Pricing</div>', unsafe_allow_html=True)
    electricity_sale_price_inr = st.number_input(
        "Electricity sale price (₹/kWh)",
        min_value=1.0, max_value=50.0, value=13.5, step=0.5, format="%.1f",
        help="Price charged to customers. Critical revenue lever."
    )
    electricity_sale_price_usd = electricity_sale_price_inr / USD_TO_INR
    st.markdown(f'<div class="explainer">= ${electricity_sale_price_usd:.2f}/kWh at ₹{USD_TO_INR}/$ exchange rate</div>', unsafe_allow_html=True)

# ─── Model ─────────────────────────────────────────────────────────────────────
utilization = daily_util_hrs / 24
electricity_sale_price_inr_calc = electricity_sale_price_usd * USD_TO_INR

# Voltage tier
power_mw = (charger_size * charger_quantity) / 1000
if power_mw <= 1:      voltage = 11
elif power_mw <= 5:    voltage = 33
elif power_mw <= 10:   voltage = 66
elif power_mw <= 50:   voltage = 132
else:                  voltage = 220

# CAPEX
total_hw_usd   = hw_cost_per_charger_usd * charger_quantity
install_usd    = total_hw_usd * (install_pct / 100)

proc_fee_usd   = processing_fee_usd if processing_fee_usd is not None else (160 if voltage >= 66 else 270)
line_rate_usd  = line_cost_per_km_usd if line_cost_per_km_usd is not None else LINE_FEE_PER_KM_DEFAULT[voltage]
line_ext_usd   = line_rate_usd * line_km
substation_usd = GROSS_SUBSTATION[voltage == 220] * (substation_share_pct / 100)
electrical_usd = proc_fee_usd + line_ext_usd + substation_usd

total_capex_usd = total_hw_usd + install_usd + electrical_usd
total_capex_inr = total_capex_usd * USD_TO_INR
total_hw_inr    = total_hw_usd   * USD_TO_INR
install_inr     = install_usd    * USD_TO_INR
electrical_inr  = electrical_usd * USD_TO_INR

# Debt service
debt_usd   = total_capex_usd * debt_ratio
equity_usd = total_capex_usd * (1 - debt_ratio)
annual_debt_usd = (debt_usd * interest_rate /
                   (1 - (1 + interest_rate) ** (-loan_years))) if debt_usd > 0 else 0
annual_debt_inr = annual_debt_usd * USD_TO_INR

# OPEX
tariff = TARIFF_DATA[selected_state]
annual_kwh      = utilization * charger_size * charger_quantity * 365 * 24
energy_cost_inr = annual_kwh * tariff["energy_charge"]
demand_cost_inr = charger_size * charger_quantity * tariff["demand_charge"] * 12
elec_cost_inr   = energy_cost_inr + demand_cost_inr

land_lease_inr   = (charger_quantity * land_area_per_charger * land_cost_per_m2_usd * USD_TO_INR) / loan_years
maintenance_inr  = total_hw_usd * (maint_pct / 100) * USD_TO_INR
total_opex_inr   = elec_cost_inr + land_lease_inr + maintenance_inr

# Revenue & cash flows
annual_rev_inr   = annual_kwh * electricity_sale_price_inr_calc
equity_inr       = equity_usd * USD_TO_INR
annual_net       = annual_rev_inr - total_opex_inr - annual_debt_inr
cashflows_inr    = [-equity_inr] + [annual_net] * PROJECT_YEARS
irr              = calc_irr(cashflows_inr)

# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# EV Charging Station — Advanced Model")
st.markdown(
    f"**{charger_quantity}× {charger_size} kW · {selected_state} · "
    f"{daily_util_hrs:.1f} hrs/day · {int(debt_ratio*100)}% debt @ {int(interest_rate*100)}% · {loan_years}-yr loan**"
)
st.divider()

# ── KPI cards ──────────────────────────────────────────────────────────────────
irr_str   = "N/A"
irr_badge = "badge-red"
irr_label = "Unviable"
if irr is not None:
    irr_pct   = irr * 100
    irr_str   = f"{irr_pct:.1f}%"
    irr_badge = "badge-green" if irr_pct > 20 else "badge-amber" if irr_pct > 0 else "badge-red"
    irr_label = "Strong" if irr_pct > 20 else "Moderate" if irr_pct > 0 else "Negative"

def kpi_card(col, label, inr_val, usd_val=None, extra_html=""):
    usd_str = f'<div class="metric-secondary">{fmt_usd(usd_val)}</div>' if usd_val is not None else ""
    col.markdown(f"""
    <div class="metric-block">
      <div class="metric-label">{label}</div>
      <div class="metric-primary">{fmt_inr(inr_val)}</div>
      {usd_str}{extra_html}
    </div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi_card(c1, "Total capex",       total_capex_inr, total_capex_usd)
kpi_card(c2, "Annual revenue",    annual_rev_inr,  annual_rev_inr / USD_TO_INR)
kpi_card(c3, "Annual opex",       total_opex_inr,  total_opex_inr / USD_TO_INR)
c4.markdown(f"""
<div class="metric-block">
  <div class="metric-label">15-yr project IRR <span title="IRR on equity over {PROJECT_YEARS}-year project life." style="cursor:help;color:#adb5bd;">ⓘ</span></div>
  <div class="metric-primary">{irr_str}</div>
  <span class="metric-badge {irr_badge}">{irr_label}</span>
</div>""", unsafe_allow_html=True)
kpi_card(c5, "Annual debt service", annual_debt_inr, annual_debt_usd)

st.divider()

# ─── Charts ────────────────────────────────────────────────────────────────────
NAVY = "#1B2B4B"; TEAL = "#1D9E75"; AMBER = "#EF9F27"
CORAL = "#D85A30"; SLATE = "#64748b"; GREEN = "#22c55e"; RED = "#ef4444"
BG = "rgba(0,0,0,0)"

base_layout = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
)

col_a, col_b, col_c = st.columns(3)

# ── Capex breakdown bar ────────────────────────────────────────────────────────
with col_a:
    st.markdown("#### Capex breakdown")
    fig_cap = go.Figure(go.Bar(
        x=["Hardware", "Installation", "Electrical &<br>grid connection"],
        y=[total_hw_inr, install_inr, electrical_inr],
        marker_color=[NAVY, TEAL, AMBER],
        text=[fmt_inr(v) for v in [total_hw_inr, install_inr, electrical_inr]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{text}<extra></extra>",
    ))
    fig_cap.update_layout(**{**base_layout,
        "yaxis": dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, visible=False),
        "height": 280,
    })
    st.plotly_chart(fig_cap, use_container_width=True)

    cap_rows = [
        ("Hardware",                   total_hw_inr,   total_hw_usd),
        (f"Installation ({install_pct:.0f}% of hw)", install_inr, install_usd),
        ("Electrical & grid",          electrical_inr, electrical_usd),
        ("Substation share",           substation_usd * USD_TO_INR, substation_usd),
        ("Total",                      total_capex_inr, total_capex_usd),
    ]
    cap_html = """
    <style>
      body{margin:0;font-family:sans-serif;}
      table{width:100%;font-size:13px;border-collapse:collapse;}
      th{text-align:left;padding:5px 4px;color:#6b7280;font-weight:500;border-bottom:1px solid #e5e7eb;}
      th.r{text-align:right;}
      td{padding:5px 4px;border-bottom:0.5px solid #f3f4f6;}
      td.r{text-align:right;}
      .tot td{border-top:1px solid #e5e7eb;border-bottom:none;font-weight:600;}
      .usd{color:#9ca3af;}
    </style>
    <table><thead><tr><th>Item</th><th class="r">₹</th><th class="r">USD</th></tr></thead><tbody>
    """
    for i, (label, inr, usd) in enumerate(cap_rows):
        is_tot = label == "Total"
        rc = "tot" if is_tot else ""
        cap_html += f'<tr class="{rc}"><td>{label}</td><td class="r">{fmt_inr(inr)}</td><td class="r usd">{fmt_usd(usd)}</td></tr>'
    cap_html += "</tbody></table>"
    components.html(cap_html, height=175, scrolling=False)

# ── Opex donut ────────────────────────────────────────────────────────────────
with col_b:
    st.markdown("#### Annual opex breakdown")
    opex_items = [
        ("Energy cost",    energy_cost_inr,  CORAL),
        ("Demand charges", demand_cost_inr,  AMBER),
        ("Land lease",     land_lease_inr,   SLATE),
        ("O&M / labour",   maintenance_inr,  TEAL),
    ]
    fig_opex = go.Figure(go.Pie(
        labels=[x[0] for x in opex_items],
        values=[x[1] for x in opex_items],
        marker=dict(colors=[x[2] for x in opex_items], line=dict(color="#fff", width=2)),
        hole=0.55, textinfo="percent", textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{label}<br>%{customdata}<br>%{percent}<extra></extra>",
        customdata=[fmt_inr(x[1]) for x in opex_items],
        direction="clockwise", sort=False,
    ))
    fig_opex.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        annotations=[dict(
            text=f"<b>{fmt_inr(total_opex_inr)}</b><br><span style='font-size:10px;color:#9ca3af'>total opex</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#374151"),
            xanchor="center", yanchor="middle",
        )],
    )
    st.plotly_chart(fig_opex, use_container_width=True)

    opex_table_rows = [(lbl, val, val / USD_TO_INR, val / total_opex_inr * 100) for lbl, val, _ in opex_items]
    opex_table_rows.append(("Total", total_opex_inr, total_opex_inr / USD_TO_INR, 100.0))

    opex_html = """
    <style>
      body{margin:0;font-family:sans-serif;}
      table{width:100%;font-size:13px;border-collapse:collapse;}
      th{text-align:left;padding:5px 4px;color:#6b7280;font-weight:500;border-bottom:1px solid #e5e7eb;}
      th.r{text-align:right;}
      td{padding:6px 4px;}
      td.r{text-align:right;}
      .pct{font-size:10px;color:#9ca3af;line-height:1.4;}
      .usd{color:#9ca3af;}
      .tot td{border-top:1px solid #e5e7eb;font-weight:600;}
      .row td{border-bottom:0.5px solid #f3f4f6;}
    </style>
    <table><thead><tr><th>Item</th><th class="r">₹</th><th class="r">USD</th></tr></thead><tbody>
    """
    for label, inr, usd, pct in opex_table_rows:
        is_tot = label == "Total"
        rc = "tot" if is_tot else "row"
        pct_html = f'<div class="pct">{pct:.0f}% of total</div>' if not is_tot else ""
        opex_html += f'<tr class="{rc}"><td>{label}{pct_html}</td><td class="r">{fmt_inr(inr)}</td><td class="r usd">{fmt_usd(usd)}</td></tr>'
    opex_html += "</tbody></table>"
    components.html(opex_html, height=175, scrolling=False)

# ── Cash flow chart ────────────────────────────────────────────────────────────
with col_c:
    st.markdown("#### Annual cash flows")
    year_labels = [f"Yr {y}" for y in range(len(cashflows_inr))]
    bar_colors  = [RED if v < 0 else GREEN for v in cashflows_inr]

    fig_cf = go.Figure(go.Bar(
        x=year_labels, y=cashflows_inr,
        marker_color=bar_colors,
        hovertemplate="Year %{x}<br>%{customdata}<extra></extra>",
        customdata=[fmt_inr(v) for v in cashflows_inr],
    ))
    fig_cf.add_hline(y=0, line_width=1, line_color="#94a3b8")

    cf_max = max(abs(v) for v in cashflows_inr)
    cr_range = cf_max / 1e7
    cr_step  = 0.5 if cr_range < 3 else (1 if cr_range < 10 else 5)
    cr_max_t = math.ceil(cf_max / 1e7 / cr_step) * cr_step
    cr_min_t = math.floor(min(cashflows_inr) / 1e7 / cr_step) * cr_step
    tick_vals = [round(i * cr_step, 2) for i in range(int(cr_min_t / cr_step), int(cr_max_t / cr_step) + 2)]

    fig_cf.update_layout(**{**base_layout,
        "showlegend": False,
        "yaxis": dict(
            showgrid=True, gridcolor="#f1f5f9", zeroline=True, zerolinecolor="#94a3b8",
            tickvals=[v * 1e7 for v in tick_vals],
            ticktext=[f"₹{v:.1f}Cr" for v in tick_vals],
        ),
        "margin": dict(l=55, r=10, t=10, b=10),
        "height": 280,
    })
    st.plotly_chart(fig_cf, use_container_width=True)

    equity_pct = int((1 - debt_ratio) * 100)
    st.markdown(f"""
    <div style="background:#f8fafc;border-radius:8px;padding:12px 16px;font-size:13px;
                color:#374151;border:1px solid #e2e8f0;">
      <b>Yr 0</b> = equity down payment ({equity_pct}% of capex)<br>
      <b>Yrs 1–{PROJECT_YEARS}</b> = revenue − opex − debt service<br>
      <span style="color:#6b7280;font-size:11px;">
        {int(debt_ratio*100)}% debt @ {int(interest_rate*100)}% · {loan_years}-yr loan · {voltage} kV grid
      </span>
    </div>""", unsafe_allow_html=True)

# ─── Assumption summary ────────────────────────────────────────────────────────
st.divider()
with st.expander("View all active assumptions"):
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**Capex assumptions**")
        st.markdown(f"- Hardware: ${hw_cost_per_charger_usd:,}/charger {'*(overridden)*' if override_hw else ''}")
        st.markdown(f"- Installation: {install_pct:.0f}% of hardware {'*(overridden)*' if override_install else ''}")
        st.markdown(f"- Line ext: ${line_rate_usd:,}/km {'*(overridden)*' if override_line else '(auto)'}")
        st.markdown(f"- Processing fee: ${proc_fee_usd:,} {'*(overridden)*' if override_proc else '(auto)'}")
        st.markdown(f"- Substation share: {substation_share_pct}% {'(excluded)' if not include_substation else ''}")
    with a2:
        st.markdown("**Opex assumptions**")
        st.markdown(f"- Land: ${land_cost_per_m2_usd}/m² · {land_area_per_charger} m²/charger {'*(overridden)*' if override_land else ''}")
        st.markdown(f"- Maintenance: {maint_pct:.1f}% of hardware p.a. {'*(overridden)*' if override_maint else ''}")
        st.markdown(f"- Energy tariff: ₹{TARIFF_DATA[selected_state]['energy_charge']}/kWh ({selected_state})")
        st.markdown(f"- Demand charge: ₹{TARIFF_DATA[selected_state]['demand_charge']}/kW/mo ({selected_state})")
    with a3:
        st.markdown("**Financing assumptions**")
        st.markdown(f"- Debt ratio: {int(debt_ratio*100)}%")
        st.markdown(f"- Interest rate: {int(interest_rate*100)}% p.a.")
        st.markdown(f"- Loan term: {loan_years} years")
        st.markdown(f"- Equity: {equity_pct}% = {fmt_inr(equity_usd * USD_TO_INR)}")
        st.markdown(f"- Annual debt service: {fmt_inr(annual_debt_inr)}")

st.markdown(f"""
<div style="font-size:11px;color:#9ca3af;margin-top:8px;">
  USD/INR: {USD_TO_INR} · Annual kWh: {annual_kwh:,.0f} · Grid voltage: {voltage} kV · 
  Project life: {PROJECT_YEARS} yrs
</div>""", unsafe_allow_html=True)
