import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="EV Charger Financial Model", layout="wide")

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
  .section-head { font-size: 13px; font-weight: 600; color: #374151;
                  text-transform: uppercase; letter-spacing: 0.05em;
                  margin: 1.4rem 0 0.6rem; border-bottom: 1px solid #e9ecef; padding-bottom: 6px; }
  .explainer { font-size: 11px; color: #9ca3af; margin-top: -8px; margin-bottom: 6px; }
  h1 { font-size: 22px !important; font-weight: 600 !important; color: #111 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
USD_TO_INR = 90
PROJECT_YEARS = 15
DEBT_RATIO = 0.80
INTEREST_RATE = 0.09

CHARGER_COSTS = {
    50:  {y: v for y, v in zip(range(2026,2040), [12973,12714,12459,12210,11966,11966,11966,11966,11966,11966,11966,11966,11966,11966])},
    100: {y: v for y, v in zip(range(2026,2040), [18900,18500,18100,17800,17400,17100,16700,16400,16100,15800,15400,15100,14800,14500])},
    250: {y: v for y, v in zip(range(2026,2040), [34462,33773,33097,32435,31787,31787,31787,31787,31787,31787,31787,31787,31787,31787])},
    500: {y: v for y, v in zip(range(2026,2040), [63500,62500,61000,60000,59000,57500,56500,55500,54000,53000,52000,51000,50000,49000])},
}

LINE_FEE_PER_KM = {11: 7000, 33: 45000, 66: 70000, 132: 94000, 220: 127000}
GROSS_SUBSTATION = {True: 5_700_000, False: 2_200_000}

TARIFF_DATA = {
    "Tamil Nadu":       {"demand_charge": 304.0, "energy_charge": 8.12},
    "Rajasthan":        {"demand_charge": 150.0, "energy_charge": 6.08},
    "Andhra Pradesh":   {"demand_charge":   0.0, "energy_charge": 6.70},
    "Karnataka":        {"demand_charge": 200.0, "energy_charge": 4.50},
    "Maharashtra":      {"demand_charge":   0.0, "energy_charge": 8.24},
    "Telangana":        {"demand_charge": 100.0, "energy_charge": 6.00},
    "Delhi":            {"demand_charge":   0.0, "energy_charge": 7.50},
    "Gujarat":          {"demand_charge":   0.0, "energy_charge": 5.50},
    "Uttar Pradesh":    {"demand_charge":   0.0, "energy_charge": 6.90},
    "West Bengal":      {"demand_charge":   0.0, "energy_charge": 5.80},
    "Madhya Pradesh":   {"demand_charge":   0.0, "energy_charge": 8.60},
    "Kerala":           {"demand_charge":   0.0, "energy_charge": 10.30},
    "Haryana":          {"demand_charge":   0.0, "energy_charge": 7.10},
    "Jharkhand":        {"demand_charge":   0.0, "energy_charge": 8.80},
}

def fmt_inr(val, decimals=0):
    """Format as ₹ with Cr/L shorthand."""
    if abs(val) >= 1e7:
        return f"₹{val/1e7:.{max(decimals,1)}f} Cr"
    elif abs(val) >= 1e5:
        return f"₹{val/1e5:.{max(decimals,1)}f} L"
    else:
        return f"₹{val:,.{decimals}f}"

def fmt_usd(val):
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:.1f}K"
    return f"${val:,.0f}"

def calc_irr(cashflows):
    """Newton-Raphson IRR. Returns None if no valid solution exists."""
    cf = np.array(cashflows, dtype=float)

    # IRR requires at least one sign change in cash flows
    signs = np.sign(cf[cf != 0])
    if len(np.unique(signs)) < 2:
        return None

    rate = 0.3
    for _ in range(1000):
        t = np.arange(len(cf))
        pv  = np.sum(cf / (1 + rate) ** t)
        dpv = np.sum(-t * cf / (1 + rate) ** (t + 1))
        if abs(dpv) < 1e-12:
            break
        new_rate = rate - pv / dpv
        if abs(new_rate - rate) < 1e-8:
            rate = new_rate
            break
        rate = max(new_rate, -0.999)

    # Sanity check — reject obviously wrong results from divergence
    if not (-0.999 < rate < 100):
        return None
    return rate

def run_model(charger_size, charger_quantity, line_km, daily_util_hrs,
              electricity_sale_price_usd, selected_state, project_year=2026):

    utilization = daily_util_hrs / 24
    electricity_sale_price_inr = electricity_sale_price_usd * USD_TO_INR

    # ── CAPEX ──────────────────────────────────────────────────────────────────
    hw_cost_usd = CHARGER_COSTS[charger_size][project_year]
    total_hw_usd = hw_cost_usd * charger_quantity
    install_usd = total_hw_usd * 0.15

    power_mw = (charger_size * charger_quantity) / 1000
    if power_mw <= 1:
        voltage = 11
    elif power_mw <= 5:
        voltage = 33
    elif power_mw <= 10:
        voltage = 66
    elif power_mw <= 50:
        voltage = 132
    else:
        voltage = 220

    processing_fee_usd = 160 if voltage >= 66 else 270
    line_ext_usd = LINE_FEE_PER_KM[voltage] * line_km
    substation_usd = GROSS_SUBSTATION[voltage == 220] * 0.25
    electrical_usd = processing_fee_usd + line_ext_usd + substation_usd

    total_capex_usd = total_hw_usd + install_usd + electrical_usd
    total_capex_inr = total_capex_usd * USD_TO_INR
    total_hw_inr    = total_hw_usd * USD_TO_INR
    install_inr     = install_usd * USD_TO_INR
    electrical_inr  = electrical_usd * USD_TO_INR

    # ── DEBT SERVICE ───────────────────────────────────────────────────────────
    debt_usd = total_capex_usd * DEBT_RATIO
    equity_usd = total_capex_usd * (1 - DEBT_RATIO)
    annual_debt_usd = (debt_usd * INTEREST_RATE /
                       (1 - (1 + INTEREST_RATE) ** (-PROJECT_YEARS)))
    annual_debt_inr = annual_debt_usd * USD_TO_INR

    # ── OPEX ───────────────────────────────────────────────────────────────────
    tariff = TARIFF_DATA[selected_state]
    energy_charge_inr_per_kwh = tariff["energy_charge"]
    demand_charge_inr_per_kw  = tariff["demand_charge"]

    annual_kwh = utilization * charger_size * charger_quantity * 365 * 24
    energy_cost_inr  = annual_kwh * energy_charge_inr_per_kwh
    demand_cost_inr  = (charger_size * charger_quantity * demand_charge_inr_per_kw * 12)
    elec_cost_inr    = energy_cost_inr + demand_cost_inr

    land_lease_inr   = (charger_quantity * 18 * 219 * USD_TO_INR) / PROJECT_YEARS
    maintenance_inr  = total_hw_usd * 0.05 * USD_TO_INR
    total_opex_inr   = elec_cost_inr + land_lease_inr + maintenance_inr

    # ── REVENUE ────────────────────────────────────────────────────────────────
    annual_kwh_sold  = utilization * charger_size * charger_quantity * 365 * 24
    annual_rev_inr   = annual_kwh_sold * electricity_sale_price_inr

    # ── CASH FLOWS & IRR ───────────────────────────────────────────────────────
    equity_inr   = equity_usd * USD_TO_INR
    annual_net   = annual_rev_inr - total_opex_inr - annual_debt_inr
    cashflows_inr = [-equity_inr] + [annual_net] * PROJECT_YEARS
    irr = calc_irr(cashflows_inr)

    return {
        "total_capex_inr": total_capex_inr,
        "hw_inr": total_hw_inr,
        "install_inr": install_inr,
        "electrical_inr": electrical_inr,
        "total_capex_usd": total_capex_usd,
        "annual_rev_inr": annual_rev_inr,
        "annual_rev_usd": annual_rev_inr / USD_TO_INR,
        "total_opex_inr": total_opex_inr,
        "total_opex_usd": total_opex_inr / USD_TO_INR,
        "energy_cost_inr": energy_cost_inr,
        "demand_cost_inr": demand_cost_inr,
        "land_lease_inr": land_lease_inr,
        "maintenance_inr": maintenance_inr,
        "annual_debt_inr": annual_debt_inr,
        "annual_debt_usd": annual_debt_usd,
        "irr": irr,
        "cashflows_inr": cashflows_inr,
        "voltage_kv": voltage,
        "annual_kwh_sold": annual_kwh_sold,
    }

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Project inputs")

    st.markdown('<div class="section-head">Charger configuration</div>', unsafe_allow_html=True)

    charger_size = st.selectbox(
        "Charger size (kW)",
        options=[50, 100, 250, 500],
        index=3,
        format_func=lambda x: {50:"50 kW", 100:"100 kW", 250:"250 kW", 500:"500 kW"}[x]
    )

    selected_state = st.selectbox(
        "State",
        options=sorted(TARIFF_DATA.keys()),
        index=sorted(TARIFF_DATA.keys()).index("Rajasthan")
    )

    charger_quantity = st.slider("Number of chargers", min_value=1, max_value=50, value=10, step=1)

    st.markdown('<div class="section-head">Utilization</div>', unsafe_allow_html=True)

    daily_util_hrs = st.slider("Daily utilization (hrs/day)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    st.markdown('<div class="explainer">Average hours each charger actively dispenses power per day.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">Pricing</div>', unsafe_allow_html=True)
    electricity_sale_price_inr = st.number_input(
        "Electricity sale price (₹/kWh)",
        min_value=1.0, max_value=50.0, value=13.5, step=0.5,
        format="%.1f",
        help="Price charged to customers in ₹/kWh. This is the critical revenue lever."
    )
    electricity_sale_price_usd = electricity_sale_price_inr / USD_TO_INR
    st.markdown(f'<div class="explainer">= ${electricity_sale_price_usd:.2f}/kWh at ₹{USD_TO_INR}/$ exchange rate</div>', unsafe_allow_html=True)

# ─── Model run ─────────────────────────────────────────────────────────────────
m = run_model(
    charger_size=charger_size,
    charger_quantity=charger_quantity,
    line_km=1.0,
    daily_util_hrs=daily_util_hrs,
    electricity_sale_price_usd=electricity_sale_price_usd,
    selected_state=selected_state,
)

# ─── Main content ──────────────────────────────────────────────────────────────
st.markdown("# EV Charging Station — Financial Model")
st.markdown(f"**{charger_quantity}× {charger_size} kW chargers · {selected_state} · {daily_util_hrs:.1f} hrs/day utilization**")
st.divider()

# ── KPI row ────────────────────────────────────────────────────────────────────
irr = m["irr"]
if irr is None:
    irr_pct_str = "N/A"
    irr_badge   = "badge-red"
    irr_label   = "Unviable"
else:
    irr_pct     = irr * 100
    irr_pct_str = f"{irr_pct:.1f}%"
    irr_badge   = "badge-green" if irr_pct > 20 else "badge-amber" if irr_pct > 0 else "badge-red"
    irr_label   = "Strong" if irr_pct > 20 else "Moderate" if irr_pct > 0 else "Negative"

col1, col2, col3, col4, col5 = st.columns(5)

def kpi_card(col, label, inr_val, usd_val=None, badge_html=""):
    usd_str = f'<div class="metric-secondary">{fmt_usd(usd_val)}</div>' if usd_val is not None else ""
    col.markdown(f"""
    <div class="metric-block">
      <div class="metric-label">{label}</div>
      <div class="metric-primary">{fmt_inr(inr_val)}</div>
      {usd_str}{badge_html}
    </div>""", unsafe_allow_html=True)

kpi_card(col1, "Total capex",    m["total_capex_inr"], m["total_capex_usd"])
kpi_card(col2, "Annual revenue", m["annual_rev_inr"],  m["annual_rev_usd"])
kpi_card(col3, "Annual opex",    m["total_opex_inr"],  m["total_opex_usd"])

col4.markdown(f"""
<div class="metric-block">
  <div class="metric-label">15-yr project IRR <span title="Internal rate of return on equity over a 15-year project life, with 80% debt at 9% over 15 years." style="cursor:help; color:#adb5bd;">ⓘ</span></div>
  <div class="metric-primary">{irr_pct_str}</div>
  <span class="metric-badge {irr_badge}">{irr_label}</span>
</div>""", unsafe_allow_html=True)

kpi_card(col5, "Annual debt service", m["annual_debt_inr"], m["annual_debt_usd"])

st.divider()

# ── Charts row ─────────────────────────────────────────────────────────────────
NAVY    = "#1B2B4B"
TEAL    = "#1D9E75"
AMBER   = "#EF9F27"
CORAL   = "#D85A30"
SLATE   = "#64748b"
GREEN   = "#22c55e"
RED     = "#ef4444"
CHART_BG = "rgba(0,0,0,0)"

chart_layout = dict(
    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    margin=dict(l=10, r=10, t=36, b=10),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5,
                font=dict(size=11)),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, tickprefix="₹"),
)

col_a, col_b, col_c = st.columns(3)

# ── 1. Capex breakdown ─────────────────────────────────────────────────────────
with col_a:
    st.markdown("#### Capex breakdown")
    fig_capex = go.Figure(go.Bar(
        x=["Hardware", "Installation", "Electrical &<br>grid connection"],
        y=[m["hw_inr"], m["install_inr"], m["electrical_inr"]],
        marker_color=[NAVY, TEAL, AMBER],
        text=[fmt_inr(v) for v in [m["hw_inr"], m["install_inr"], m["electrical_inr"]]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>"
    ))
    fig_capex.update_layout(**{**chart_layout,
        "showlegend": False,
        "yaxis": dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, visible=False),
        "margin": dict(l=10, r=10, t=10, b=10),
        "height": 280,
    })
    st.plotly_chart(fig_capex, use_container_width=True)

    # Capex table
    capex_rows = [
        ("Hardware", m["hw_inr"], m["hw_inr"] / USD_TO_INR),
        ("Installation (15% of hw)", m["install_inr"], m["install_inr"] / USD_TO_INR),
        ("Electrical & grid connection", m["electrical_inr"], m["electrical_inr"] / USD_TO_INR),
        ("**Total**", m["total_capex_inr"], m["total_capex_usd"]),
    ]
    df_cap = pd.DataFrame(capex_rows, columns=["Item", "₹", "USD"])
    df_cap["₹"]   = df_cap["₹"].apply(fmt_inr)
    df_cap["USD"] = df_cap["USD"].apply(fmt_usd)
    st.dataframe(df_cap, hide_index=True, use_container_width=True)

# ── 2. Opex breakdown ──────────────────────────────────────────────────────────
with col_b:
    st.markdown("#### Annual opex breakdown")

    opex_items = [
        ("Energy cost",    m["energy_cost_inr"],  CORAL),
        ("Demand charges", m["demand_cost_inr"],  AMBER),
        ("Land lease",     m["land_lease_inr"],   SLATE),
        ("O&M / labour",   m["maintenance_inr"],  TEAL),
    ]
    opex_labels = [x[0] for x in opex_items]
    opex_vals   = [x[1] for x in opex_items]
    opex_colors = [x[2] for x in opex_items]

    fig_opex = go.Figure(go.Pie(
        labels=opex_labels,
        values=opex_vals,
        marker=dict(colors=opex_colors, line=dict(color="#ffffff", width=2)),
        hole=0.55,
        textinfo="percent",
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{label}<br>%{customdata}<br>%{percent}<extra></extra>",
        customdata=[fmt_inr(v) for v in opex_vals],
        direction="clockwise",
        sort=False,
    ))
    fig_opex.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        annotations=[dict(
            text=f"<b>{fmt_inr(m['total_opex_inr'])}</b><br><span style='font-size:10px;color:#9ca3af'>total opex</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#374151"),
            xanchor="center", yanchor="middle",
        )],
    )
    st.plotly_chart(fig_opex, use_container_width=True)

    # Table with % of total as light gray subtext
    total_opex = m["total_opex_inr"]
    opex_rows = [
        (label, val, val / USD_TO_INR, val / total_opex * 100)
        for label, val, _ in opex_items
    ]
    opex_rows.append(("**Total**", total_opex, total_opex / USD_TO_INR, 100.0))

    table_html = """
    <style>
      body { margin: 0; font-family: sans-serif; }
      table { width:100%; font-size:13px; border-collapse:collapse; }
      th { text-align:left; padding:5px 4px; color:#6b7280; font-weight:500;
           border-bottom:1px solid #e5e7eb; }
      th.r { text-align:right; }
      td { padding:6px 4px; }
      td.r { text-align:right; }
      .pct { font-size:10px; color:#9ca3af; line-height:1.4; }
      .usd { color:#9ca3af; }
      .total td { border-top:1px solid #e5e7eb; font-weight:600; }
      .row td { border-bottom:0.5px solid #f3f4f6; }
    </style>
    <table>
      <thead>
        <tr><th>Item</th><th class="r">₹</th><th class="r">USD</th></tr>
      </thead>
      <tbody>
    """
    for label, val_inr, val_usd, pct in opex_rows:
        is_total = label == "**Total**"
        row_class = "total" if is_total else "row"
        pct_html  = f'<div class="pct">{pct:.0f}% of total</div>' if not is_total else ""
        table_html += f"""
        <tr class="{row_class}">
          <td>{label.replace("**", "")}{pct_html}</td>
          <td class="r">{fmt_inr(val_inr)}</td>
          <td class="r usd">{fmt_usd(val_usd)}</td>
        </tr>"""
    table_html += "</tbody></table>"
    components.html(table_html, height=175, scrolling=False)

# ── 3. Cash flows ──────────────────────────────────────────────────────────────
with col_c:
    st.markdown("#### Annual cash flows")
    cfs = m["cashflows_inr"]
    years = list(range(len(cfs)))
    year_labels = [f"Yr {y}" for y in years]
    bar_colors = [RED if v < 0 else GREEN for v in cfs]

    fig_cf = go.Figure(go.Bar(
        x=year_labels,
        y=cfs,
        marker_color=bar_colors,
        hovertemplate="Year %{x}<br>" + "%{customdata}<extra></extra>",
        customdata=[fmt_inr(v) for v in cfs],
    ))
    fig_cf.add_hline(y=0, line_width=1, line_color="#94a3b8")

    cf_max = max(abs(v) for v in cfs)
    cr_range = cf_max / 1e7
    cr_step = 0.5 if cr_range < 3 else (1 if cr_range < 10 else 5)
    import math
    cr_max_tick = math.ceil(cf_max / 1e7 / cr_step) * cr_step
    cr_min_tick = math.floor(min(cfs) / 1e7 / cr_step) * cr_step
    tick_vals = [round(i * cr_step, 2) for i in range(int(cr_min_tick / cr_step), int(cr_max_tick / cr_step) + 2)]

    fig_cf.update_layout(**{**chart_layout,
        "showlegend": False,
        "yaxis": dict(
            showgrid=True, gridcolor="#f1f5f9", zeroline=True,
            zerolinecolor="#94a3b8",
            tickvals=[v * 1e7 for v in tick_vals],
            ticktext=[f"₹{v:.1f}Cr" for v in tick_vals],
        ),
        "margin": dict(l=55, r=10, t=10, b=10),
        "height": 280,
    })
    st.plotly_chart(fig_cf, use_container_width=True)

    st.markdown(f"""
    <div style="background:#f8fafc; border-radius:8px; padding:12px 16px; font-size:13px; color:#374151; border: 1px solid #e2e8f0;">
      <b>Yr 0</b> = equity down payment ({int((1-DEBT_RATIO)*100)}% of capex)<br>
      <b>Yrs 1–{PROJECT_YEARS}</b> = revenue − opex − debt service<br>
      <span style="color:#6b7280; font-size:11px;">80% debt @ 9% over 15 yrs · {m['voltage_kv']} kV grid connection</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown(f"""
<div style="font-size:11px; color:#9ca3af;">
  USD/INR rate: {USD_TO_INR} · State tariff: {selected_state} · 
  Annual kWh sold: {m['annual_kwh_sold']:,.0f} · 
  Grid voltage: {m['voltage_kv']} kV · 
  Debt structure: {int(DEBT_RATIO*100)}% debt @ {int(INTEREST_RATE*100)}% over {PROJECT_YEARS} yrs
</div>
""", unsafe_allow_html=True)
