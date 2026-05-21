import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="EV Charging Finance", layout="wide")

# ── DATA ──────────────────────────────────────────────────────────────────────

CHARGER_COSTS = {
    100: {2026:18900,2027:18500,2028:18100,2029:17800,2030:17400,2031:17100,
          2032:16700,2033:16400,2034:16100,2035:15800,2036:15400,2037:15100,
          2038:14800,2039:14500},
    500: {2026:63500,2027:62500,2028:61000,2029:60000,2030:59000,2031:57500,
          2032:56500,2033:55500,2034:54000,2035:53000,2036:52000,2037:51000,
          2038:50000,2039:49000},
}

TARIFFS = {
    "Tamil Nadu":       {"demand": 304.0, "energy": 8.12},
    "Rajasthan":        {"demand": 150.0, "energy": 6.08},
    "Andhra Pradesh":   {"demand":   0.0, "energy": 6.70},
    "Karnataka":        {"demand": 200.0, "energy": 4.50},
    "Kerala":           {"demand":   0.0, "energy": 10.30},
    "West Bengal":      {"demand":   0.0, "energy": 5.80},
    "Haryana":          {"demand":   0.0, "energy": 7.10},
    "Jharkhand":        {"demand":   0.0, "energy": 8.80},
    "Maharashtra":      {"demand":   0.0, "energy": 8.24},
    "Telangana":        {"demand": 100.0, "energy": 6.00},
    "Madhya Pradesh":   {"demand":   0.0, "energy": 8.60},
}

USD = 90
AVG_D = sum(t["demand"] for t in TARIFFS.values()) / len(TARIFFS)
AVG_E = sum(t["energy"] for t in TARIFFS.values()) / len(TARIFFS)

SCEN_KEYS  = ["base", "ppa", "managed", "solar"]
SCEN_NAMES = {
    "base":    "Grid (no RE)",
    "ppa":     "PPA",
    "managed": "Managed charging",
    "solar":   "Solar + storage",
}
SCEN_COLORS = {
    "base":    "#888780",
    "ppa":     "#378ADD",
    "managed": "#1D9E75",
    "solar":   "#BA7517",
}
SCEN_COLORS_LIGHT = {
    "base":    "rgba(136,135,128,0.25)",
    "ppa":     "rgba(55,138,221,0.25)",
    "managed": "rgba(29,158,117,0.25)",
    "solar":   "rgba(186,117,23,0.25)",
}

# ── CALCULATIONS ──────────────────────────────────────────────────────────────

def elec_cost(scen, pkw, csz, qty, ann_kwh):
    if scen == "ppa":
        gross = 1_550_000 * (pkw / 1000)
        return ((3.20 * gross) + (0.61 + 0.62 + 1.36 + 1.44) * gross + 6060 * 12) / USD
    if scen == "solar":
        bat = pkw * 2 / 24
        return max(0, (ann_kwh - bat * 365 * 24) * (AVG_E / USD) + csz * qty * (AVG_D / USD) * 12)
    if scen == "managed":
        return (ann_kwh * (AVG_E / USD) + csz * qty * (AVG_D / USD) * 12) * 0.8
    return ann_kwh * (AVG_E / USD) + csz * qty * (AVG_D / USD) * 12


def calc_irr(cash_flows):
    from numpy_financial import irr
    result = irr(cash_flows)
    return result if result is not None and result == result else None


def run_model(yr, csz, qty, km, util_hrs, sale_price, active_scen):
    util = util_hrs / 24
    hw   = CHARGER_COSTS[csz][yr] * qty
    inst = hw * 0.15
    pkw  = csz * qty
    pmw  = pkw / 1000

    if pmw <= 1:    kv = 11
    elif pmw <= 5:  kv = 33
    elif pmw <= 10: kv = 66
    elif pmw <= 50: kv = 132
    else:           kv = 220

    proc_fee  = 270 if kv <= 33 else 160
    line_rate = {11: 7_000, 33: 45_000, 66: 70_000, 132: 94_000, 220: 127_000}
    line_ext  = line_rate[kv] * km
    subst     = (2_200_000 if kv < 220 else 5_700_000) * 0.25
    elec_conn = proc_fee + line_ext + subst
    capex     = hw + inst + elec_conn

    debt  = capex * 0.8
    down  = capex * 0.2
    rate  = 0.09
    lyrs  = 15
    dsvc  = debt * rate / (1 - (1 + rate) ** (-lyrs))

    land  = (qty * 18 * 219) / lyrs
    maint = hw * 0.05
    ann_kwh = util * csz * qty * 365 * 24
    rev     = ann_kwh * sale_price

    results = {}
    for s in SCEN_KEYS:
        ec      = elec_cost(s, pkw, csz, qty, ann_kwh)
        opex    = land + maint + ec
        total   = dsvc + opex
        ncf     = rev - opex - dsvc
        cfs     = [-down] + [ncf] * lyrs
        results[s] = {
            "ec": ec, "opex": opex, "total": total,
            "ncf": ncf, "irr": calc_irr(cfs),
        }

    return {
        "hw": hw, "inst": inst, "elec_conn": elec_conn,
        "capex": capex, "kv": kv,
        "debt": debt, "down": down, "dsvc": dsvc,
        "land": land, "maint": maint,
        "ann_kwh": ann_kwh, "rev": rev,
        "scenarios": results,
        "active": results[active_scen],
    }


def fmt(n):
    a = abs(n)
    s = "-" if n < 0 else ""
    if a >= 1_000_000: return f"{s}${a/1_000_000:.1f}M"
    if a >= 1_000:     return f"{s}${a/1_000:.0f}k"
    return f"{s}${a:.0f}"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Inputs")

    active_scen = st.selectbox(
        "RE scenario",
        options=SCEN_KEYS,
        format_func=lambda k: SCEN_NAMES[k],
        help="Your chosen scenario is highlighted. Others shown as context.",
    )

    st.markdown("---")

    yr        = st.selectbox("Project year", list(range(2026, 2040)), index=0)
    csz       = st.selectbox("Charger size (kW)", [100, 500], index=1)
    qty       = st.slider("Charger quantity", 1, 50, 10)
    km        = st.slider("Grid line extension (km)", 0.0, 20.0, 2.0, step=0.5)
    util_hrs  = st.slider("Daily utilisation (hrs)", 1.0, 24.0, 6.0, step=0.5)
    sale_price= st.slider("Sale price ($/kWh)", 0.05, 0.50, 0.15, step=0.01)

# ── MODEL ─────────────────────────────────────────────────────────────────────

m = run_model(yr, csz, qty, km, util_hrs, sale_price, active_scen)
sc = m["scenarios"]
sel = sc[active_scen]

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown(f"## Scenario comparison")
st.caption(
    f"**{SCEN_NAMES[active_scen]}** highlighted · others shown as context"
)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total CAPEX",      fmt(m["capex"]))
k2.metric("Annual revenue",   fmt(m["rev"]))
k3.metric("Debt service/yr",  fmt(m["dsvc"]))
k4.metric(f"Electricity cost · {SCEN_NAMES[active_scen]}", fmt(sel["ec"]))
k5.metric(f"Total annual cost · {SCEN_NAMES[active_scen]}", fmt(sel["total"]))

st.markdown("---")

# ── MAIN SCENARIO CHART ───────────────────────────────────────────────────────

st.markdown("#### Electricity cost & total annual cost — all scenarios")
st.caption(
    "Solid bar = electricity cost only (the variable that changes). "
    "Outlined bar = total annual cost including debt service. "
    "Selected scenario highlighted; others dimmed."
)

fig = go.Figure()

labels   = [SCEN_NAMES[s] for s in SCEN_KEYS]
ec_vals  = [sc[s]["ec"]    for s in SCEN_KEYS]
tot_vals = [sc[s]["total"] for s in SCEN_KEYS]

for i, s in enumerate(SCEN_KEYS):
    selected = s == active_scen
    col      = SCEN_COLORS[s]       if selected else "#C8C7C3"
    col_fade = SCEN_COLORS_LIGHT[s] if selected else "rgba(200,199,195,0.3)"
    bw       = 2.0                  if selected else 0

    fig.add_trace(go.Bar(
        name=f"{SCEN_NAMES[s]} — electricity",
        x=[SCEN_NAMES[s]],
        y=[sc[s]["ec"]],
        marker_color=col,
        marker_line_width=0,
        width=0.28,
        offset=-0.16,
        showlegend=(i == 0),
        legendgroup="elec",
        legendgrouptitle_text="Electricity cost" if i == 0 else None,
        hovertemplate=f"<b>{SCEN_NAMES[s]}</b><br>Electricity: %{{y:$,.0f}}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name=f"{SCEN_NAMES[s]} — total",
        x=[SCEN_NAMES[s]],
        y=[sc[s]["total"]],
        marker_color=col_fade,
        marker_line_color=col,
        marker_line_width=bw,
        width=0.28,
        offset=0.02,
        showlegend=(i == 0),
        legendgroup="total",
        legendgrouptitle_text="Total annual cost" if i == 0 else None,
        hovertemplate=(
            f"<b>{SCEN_NAMES[s]}</b><br>"
            f"Total annual cost: %{{y:$,.0f}}<br>"
            f"(debt service + OpEx)<extra></extra>"
        ),
    ))

    if selected:
        fig.add_annotation(
            x=SCEN_NAMES[s], y=sc[s]["ec"],
            text=fmt(sc[s]["ec"]),
            showarrow=False, yshift=10,
            font=dict(size=12, color=col, family="Arial"),
        )
        fig.add_annotation(
            x=SCEN_NAMES[s], y=sc[s]["total"],
            text=fmt(sc[s]["total"]),
            showarrow=False, yshift=10,
            font=dict(size=12, color=col, family="Arial"),
        )

fig.update_layout(
    barmode="overlay",
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=380,
    margin=dict(t=20, b=40, l=60, r=20),
    yaxis=dict(
        tickprefix="$",
        tickformat=",.0f",
        gridcolor="#f0efe8",
        title="$/yr",
    ),
    xaxis=dict(tickfont=dict(size=13)),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
    ),
    font=dict(family="Arial, sans-serif", size=12),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── CAPEX BREAKDOWN ───────────────────────────────────────────────────────────

st.markdown("#### CAPEX breakdown")
st.caption(
    f"Total CAPEX: **{fmt(m['capex'])}** · "
    f"Grid connection voltage: {m['kv']} kV · "
    f"Annual debt service: {fmt(m['dsvc'])}/yr over 15 years at 9%"
)

capex_labels = ["Hardware", "Installation (15%)", "Electrical connection"]
capex_values = [m["hw"], m["inst"], m["elec_conn"]]
capex_colors = ["#378ADD", "#1D9E75", "#BA7517"]
capex_pcts   = [v / m["capex"] * 100 for v in capex_values]

c1, c2 = st.columns([1.6, 1])

with c1:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=capex_values,
        y=capex_labels,
        orientation="h",
        marker_color=capex_colors,
        text=[f"{fmt(v)}  ({p:.0f}%)" for v, p in zip(capex_values, capex_pcts)],
        textposition="outside",
        hovertemplate="%{y}: %{x:$,.0f}<extra></extra>",
    ))
    fig2.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=220,
        margin=dict(t=10, b=30, l=10, r=120),
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#f0efe8"),
        yaxis=dict(tickfont=dict(size=13)),
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
    )
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    fig3 = go.Figure(go.Pie(
        labels=capex_labels,
        values=capex_values,
        marker_colors=capex_colors,
        hole=0.55,
        textinfo="percent",
        hovertemplate="%{label}<br>%{value:$,.0f} (%{percent})<extra></extra>",
    ))
    fig3.add_annotation(
        text=fmt(m["capex"]), x=0.5, y=0.5,
        font=dict(size=14, color="#2c2c2a"),
        showarrow=False,
    )
    fig3.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=0, r=0),
        showlegend=False,
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption(
    "Electrical connection includes processing fee, line extension, "
    f"and {m['kv']}kV substation cost share (25%). "
    "CAPEX financed at 80% debt / 20% equity, 9% interest, 15-year term."
)