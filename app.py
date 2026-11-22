import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NIFTY 50 Gamma & Options Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# CONSTANTS & NSE HOLIDAY SCHEDULE
# ----------------------------------------------------------------------
RISK_FREE_RATE = 0.065
NIFTY_LOT_SIZE = 65

NSE_HOLIDAYS = {
    "2026-01-26", "2026-03-03", "2026-03-20", "2026-04-02",
    "2026-04-14", "2026-05-01", "2026-08-15", "2026-10-02",
    "2026-10-20", "2026-11-10", "2026-11-24", "2026-12-25"
}

def get_nifty_expiry(current_dt: datetime):
    today_date = current_dt.date()
    days_to_tuesday = 1 - today_date.weekday()
    this_tuesday = today_date + timedelta(days=days_to_tuesday)

    if this_tuesday.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        candidate_expiry = this_tuesday - timedelta(days=1)
    else:
        candidate_expiry = this_tuesday

    cutoff_dt = datetime(candidate_expiry.year, candidate_expiry.month, candidate_expiry.day, 15, 30, 0)
    if current_dt > cutoff_dt:
        next_tuesday = this_tuesday + timedelta(days=7)
        candidate_expiry = next_tuesday - timedelta(days=1) if next_tuesday.strftime("%Y-%m-%d") in NSE_HOLIDAYS else next_tuesday

    return candidate_expiry

# ----------------------------------------------------------------------
# CLOSED-FORM BSM GREEKS
# ----------------------------------------------------------------------
def bsm_greeks(S, K, T, sigma, r=RISK_FREE_RATE):
    if sigma <= 0.001 or T <= 1e-6:
        call_delta = 1.0 if S > K else 0.0
        put_delta = 0.0 if S > K else -1.0
        return call_delta, put_delta, 0.0, 0.0

    tau = max(T, 1e-6)
    vol = max(sigma, 0.01)

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(tau))
    vega = S * norm.pdf(d1) * np.sqrt(tau) / 100.0

    return call_delta, put_delta, gamma, vega

# ----------------------------------------------------------------------
# LIVE SPOT & CHAIN LOADER
# ----------------------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_spot():
    try:
        spot_df = yf.download("^NSEI", period="1d", interval="1m", progress=False)
        if not spot_df.empty:
            return float(spot_df["Close"].squeeze().dropna().iloc[-1])
    except Exception:
        pass
    return 25044.65  # Fallback if market is closed / feed unavailable

def generate_demo_chain(spot):
    """Generates a realistic NIFTY weekly chain around live spot for public demo visitors."""
    atm = round(spot / 50) * 50
    strikes = np.arange(atm - 1000, atm + 1050, 50)
    np.random.seed(42)

    call_oi = np.where(strikes >= atm, np.random.randint(40000, 180000, len(strikes)), np.random.randint(5000, 35000, len(strikes)))
    put_oi = np.where(strikes <= atm, np.random.randint(40000, 190000, len(strikes)), np.random.randint(5000, 35000, len(strikes)))
    
    # Add realistic skew
    moneyness = (strikes - spot) / spot
    call_iv = np.clip(14.5 - 25 * moneyness + 80 * (moneyness**2), 9.0, 45.0)
    put_iv = np.clip(14.8 - 30 * moneyness + 90 * (moneyness**2), 9.0, 48.0)

    return pd.DataFrame({
        "STRIKE": strikes,
        "CALL_OI": call_oi,
        "PUT_OI": put_oi,
        "CALL_CHG_OI": np.random.randint(-10000, 25000, len(strikes)),
        "PUT_CHG_OI": np.random.randint(-10000, 25000, len(strikes)),
        "CALL_IV": np.round(call_iv, 2),
        "PUT_IV": np.round(put_iv, 2),
        "CALL_LTP": np.maximum(spot - strikes, 0) + 65 * np.exp(-abs(moneyness) * 18),
        "PUT_LTP": np.maximum(strikes - spot, 0) + 65 * np.exp(-abs(moneyness) * 18)
    })

# ----------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------
st.sidebar.header("Engine Controls")
uploaded_file = st.sidebar.file_uploader("Upload NSE Option Chain CSV (Optional)", type=["csv"])
auto_refresh = st.sidebar.toggle("Live Market Auto-Refresh (30s)", value=False)
strike_window = st.sidebar.slider("Strike Window (± Points from Spot)", 500, 2000, 1000, step=100)

live_spot = fetch_live_spot()
spot = st.sidebar.number_input("Spot Price Override (₹)", value=float(round(live_spot, 2)), step=5.0)
r_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=6.5, step=0.1) / 100.0

# ----------------------------------------------------------------------
# DATA PROCESSING
# ----------------------------------------------------------------------
now = datetime.now()
expiry_date = get_nifty_expiry(now)
days_to_expiry = max((expiry_date - now.date()).days, 0)
expiry_dt = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 15, 30, 0)
seconds_to_expiry = max((expiry_dt - now).total_seconds(), 3600.0)
T = seconds_to_expiry / (365.25 * 86400.0)

if uploaded_file is not None:
    chain = pd.read_csv(uploaded_file).replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    chain = generate_demo_chain(spot)

chain["DISTANCE"] = (chain["STRIKE"] - spot).abs()
atm = chain.loc[chain["DISTANCE"].idxmin()]
atm_strike = float(atm["STRIKE"])

pcr = chain["PUT_OI"].sum() / max(chain["CALL_OI"].sum(), 1)
atm_iv = (float(atm["CALL_IV"]) + float(atm["PUT_IV"])) / 2.0
atm_iv = 14.5 if atm_iv <= 0 else atm_iv
expected_move = spot * (atm_iv / 100.0) * np.sqrt(T)

# Compute Greeks
greeks = [
    bsm_greeks(spot, row["STRIKE"], T, max(row["CALL_IV"] if row["CALL_IV"] > 0 else atm_iv, 1.0) / 100.0, r_rate)
    for _, row in chain.iterrows()
]
chain["CALL_DELTA"] = [g[0] for g in greeks]
chain["PUT_DELTA"] = [g[1] for g in greeks]
chain["GAMMA"] = [g[2] for g in greeks]
chain["VEGA"] = [g[3] for g in greeks]

chain["CALL_GEX_CR"] = chain["CALL_OI"] * chain["GAMMA"] * (spot**2) * 0.01 * NIFTY_LOT_SIZE / 1e7
chain["PUT_GEX_CR"] = -chain["PUT_OI"] * chain["GAMMA"] * (spot**2) * 0.01 * NIFTY_LOT_SIZE / 1e7
chain["NET_GEX_CR"] = chain["CALL_GEX_CR"] + chain["PUT_GEX_CR"]
chain["DEX_CR"] = (chain["CALL_OI"] * chain["CALL_DELTA"] + chain["PUT_OI"] * chain["PUT_DELTA"]) * spot * NIFTY_LOT_SIZE / 1e7

gex_total = round(chain["NET_GEX_CR"].sum(), 2)
dex_total = round(chain["DEX_CR"].sum(), 2)
call_wall = chain.loc[chain["CALL_GEX_CR"].idxmax(), "STRIKE"]
put_wall = chain.loc[chain["PUT_GEX_CR"].abs().idxmax(), "STRIKE"]

sorted_chain = chain.sort_values("STRIKE").reset_index(drop=True)
sign_changes = np.where(np.diff(np.sign(sorted_chain["NET_GEX_CR"])))[0]
gamma_flip = sorted_chain.loc[sign_changes[0], "STRIKE"] if len(sign_changes) > 0 else np.nan

# ----------------------------------------------------------------------
# DASHBOARD UI
# ----------------------------------------------------------------------
st.title("NIFTY 50 Gamma Exposure & Options Analytics Engine")
st.caption(f"Active Expiry: **{expiry_date.strftime('%d %b %Y')}** ({days_to_expiry} DTE) | Last Updated: {now.strftime('%H:%M:%S IST')}")

# Top KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NIFTY Spot", f"₹{spot:,.2f}", f"ATM: {atm_strike:.0f}")
c2.metric("Net Dealer GEX", f"₹{gex_total:,.2f} Cr", "Long Gamma" if gex_total > 0 else "Short Gamma")
c3.metric("Dealer Delta (DEX)", f"₹{dex_total:,.2f} Cr", f"PCR: {pcr:.2f}")
c4.metric("Expected Move", f"±{expected_move:.1f} pts", f"ATM IV: {atm_iv:.2f}%")
c5.metric("Call / Put Walls", f"{call_wall:.0f} / {put_wall:.0f}", f"Flip: {gamma_flip:.0f}" if pd.notna(gamma_flip) else "Flip: N/A")

st.divider()

# Dual-Panel Gamma Profile Plot
window = chain[(chain["STRIKE"] >= spot - strike_window) & (chain["STRIKE"] <= spot + strike_window)].copy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
plt.style.use("dark_background")
fig.patch.set_facecolor("#0e1117")
ax1.set_facecolor("#0e1117")
ax2.set_facecolor("#0e1117")

width = 22
ax1.bar(window["STRIKE"] - width/2, window["CALL_GEX_CR"], width=width, color="#00E676", alpha=0.85, label="Call Gamma (Dealer Long)")
ax1.bar(window["STRIKE"] + width/2, window["PUT_GEX_CR"], width=width, color="#FF5252", alpha=0.85, label="Put Gamma (Dealer Short)")
ax1.axvline(spot, color="#40C4FF", linestyle="--", linewidth=1.8, label=f"Spot: {spot:.1f}")
ax1.axvline(call_wall, color="#1DE9B6", linestyle=":", linewidth=1.8, label=f"Call Wall: {call_wall:.0f}")
ax1.axvline(put_wall, color="#FF4081", linestyle=":", linewidth=1.8, label=f"Put Wall: {put_wall:.0f}")
ax1.set_title("Dealer Gamma Exposure by Strike (₹ Crores per 1% Move)", fontsize=12, fontweight="bold", color="white")
ax1.set_ylabel("GEX (₹ Cr)", color="lightgray")
ax1.grid(True, linestyle="--", alpha=0.2)
ax1.legend(loc="upper left", framealpha=0.3)

ax2.plot(window["STRIKE"], window["NET_GEX_CR"], color="#82B1FF", linewidth=2.2, marker="o", markersize=4, label="Net Dealer Gamma")
ax2.fill_between(window["STRIKE"], window["NET_GEX_CR"], 0, where=(window["NET_GEX_CR"] >= 0), color="#00E676", alpha=0.2, interpolate=True)
ax2.fill_between(window["STRIKE"], window["NET_GEX_CR"], 0, where=(window["NET_GEX_CR"] < 0), color="#FF5252", alpha=0.2, interpolate=True)
ax2.axhline(0, color="gray", linestyle="-", linewidth=1.0)
ax2.axvline(spot, color="#40C4FF", linestyle="--", linewidth=1.5)
if pd.notna(gamma_flip):
    ax2.axvline(gamma_flip, color="#FFAB40", linestyle="-.", linewidth=1.8, label=f"Gamma Flip: {gamma_flip:.0f}")

ax2.set_xlabel("Strike Price", fontsize=11, color="lightgray")
ax2.set_ylabel("Net GEX (₹ Cr)", color="lightgray")
ax2.grid(True, linestyle="--", alpha=0.2)
ax2.legend(loc="upper left", framealpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Data Table & EOD CSV Download
st.subheader("📊 Strike-by-Strike Greeks & Dealer Exposure Table")
display_cols = ["STRIKE", "CALL_OI", "PUT_OI", "CALL_IV", "PUT_IV", "CALL_DELTA", "PUT_DELTA", "GAMMA", "CALL_GEX_CR", "PUT_GEX_CR", "NET_GEX_CR", "DEX_CR"]
st.dataframe(window[display_cols].style.format("{:.2f}", subset=["CALL_IV", "PUT_IV", "CALL_DELTA", "PUT_DELTA", "CALL_GEX_CR", "PUT_GEX_CR", "NET_GEX_CR", "DEX_CR"]), use_container_width=True)

csv_data = chain.to_csv(index=False).encode("utf-8")
st.download_button("Download EOD Settled Chain (CSV)", data=csv_data, file_name=f"nifty_eod_chain_{now.strftime('%Y-%m-%d')}.csv", mime="text/csv")

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NIFTY 50 Quant & Options Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# CONSTANTS & NSE HOLIDAY SCHEDULE
# ----------------------------------------------------------------------
RISK_FREE_RATE = 0.065
NIFTY_LOT_SIZE = 65

NSE_HOLIDAYS = {
    "2026-01-26", "2026-03-03", "2026-03-20", "2026-04-02",
    "2026-04-14", "2026-05-01", "2026-08-15", "2026-10-02",
    "2026-10-20", "2026-11-10", "2026-11-24", "2026-12-25"
}

def get_nifty_expiry(current_dt: datetime):
    today_date = current_dt.date()
    days_to_tuesday = 1 - today_date.weekday()
    this_tuesday = today_date + timedelta(days=days_to_tuesday)

    if this_tuesday.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        candidate_expiry = this_tuesday - timedelta(days=1)
    else:
        candidate_expiry = this_tuesday

    cutoff_dt = datetime(candidate_expiry.year, candidate_expiry.month, candidate_expiry.day, 15, 30, 0)
    if current_dt > cutoff_dt:
        next_tuesday = this_tuesday + timedelta(days=7)
        candidate_expiry = next_tuesday - timedelta(days=1) if next_tuesday.strftime("%Y-%m-%d") in NSE_HOLIDAYS else next_tuesday

    return candidate_expiry

# ----------------------------------------------------------------------
# CLOSED-FORM BSM ENGINE (PRICE & GREEKS)
# ----------------------------------------------------------------------
def bsm_full_metrics(S, K, T, sigma, option_type="CE", r=RISK_FREE_RATE):
    opt = option_type.upper()
    if sigma <= 0.001 or T <= 1e-7:
        intrinsic = max(S - K, 0.0) if opt == "CE" else max(K - S, 0.0)
        delta = (1.0 if S > K else 0.0) if opt == "CE" else (-1.0 if S < K else 0.0)
        return intrinsic, delta, 0.0, 0.0, 0.0

    tau = max(T, 1e-7)
    vol = max(sigma, 0.01)

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)

    gamma = norm.pdf(d1) / (S * vol * np.sqrt(tau))
    vega = S * norm.pdf(d1) * np.sqrt(tau) / 100.0
    common_theta = -(S * norm.pdf(d1) * vol) / (2 * np.sqrt(tau))

    if opt == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        delta = norm.cdf(d1)
        annual_theta = common_theta - r * K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        annual_theta = common_theta + r * K * np.exp(-r * tau) * norm.cdf(-d2)

    daily_theta = abs(annual_theta) / 365.0
    return price, delta, gamma, vega, daily_theta

# ----------------------------------------------------------------------
# LIVE SPOT & CHAIN LOADER
# ----------------------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_spot():
    try:
        spot_df = yf.download("^NSEI", period="1d", interval="1m", progress=False)
        if not spot_df.empty:
            return float(spot_df["Close"].squeeze().dropna().iloc[-1])
    except Exception:
        pass
    return 25044.65

def generate_demo_chain(spot, T, r=RISK_FREE_RATE):
    """Generates a realistic NIFTY weekly chain around live spot for public demo visitors."""
    atm = round(spot / 50) * 50
    strikes = np.arange(atm - 1000, atm + 1050, 50)
    np.random.seed(42)

    call_oi = np.where(strikes >= atm, np.random.randint(40000, 180000, len(strikes)), np.random.randint(5000, 35000, len(strikes)))
    put_oi = np.where(strikes <= atm, np.random.randint(40000, 190000, len(strikes)), np.random.randint(5000, 35000, len(strikes)))

    moneyness = (strikes - spot) / spot
    call_iv = np.clip(14.5 - 25 * moneyness + 80 * (moneyness**2), 9.0, 45.0)
    put_iv = np.clip(14.8 - 30 * moneyness + 90 * (moneyness**2), 9.0, 48.0)

    call_ltp = [bsm_full_metrics(spot, k, T, iv / 100.0, "CE", r)[0] for k, iv in zip(strikes, call_iv)]
    put_ltp = [bsm_full_metrics(spot, k, T, iv / 100.0, "PE", r)[0] for k, iv in zip(strikes, put_iv)]

    return pd.DataFrame({
        "STRIKE": strikes,
        "CALL_OI": call_oi,
        "PUT_OI": put_oi,
        "CALL_CHG_OI": np.random.randint(-10000, 25000, len(strikes)),
        "PUT_CHG_OI": np.random.randint(-10000, 25000, len(strikes)),
        "CALL_IV": np.round(call_iv, 2),
        "PUT_IV": np.round(put_iv, 2),
        "CALL_LTP": np.round(call_ltp, 2),
        "PUT_LTP": np.round(put_ltp, 2)
    })

# ----------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------
st.sidebar.header("Engine Controls")
uploaded_file = st.sidebar.file_uploader("Upload NSE Option Chain CSV (Optional)", type=["csv"])
auto_refresh = st.sidebar.toggle("Live Market Auto-Refresh (30s)", value=False)
strike_window = st.sidebar.slider("Strike Window (± Points from Spot)", 500, 2000, 1000, step=100)

live_spot = fetch_live_spot()
spot = st.sidebar.number_input("Spot Price Override (₹)", value=float(round(live_spot, 2)), step=5.0)
r_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=6.5, step=0.1) / 100.0

# ----------------------------------------------------------------------
# GLOBAL DATA PROCESSING
# ----------------------------------------------------------------------
now = datetime.now()
expiry_date = get_nifty_expiry(now)
days_to_expiry = max((expiry_date - now.date()).days, 0)
expiry_dt = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 15, 30, 0)
seconds_to_expiry = max((expiry_dt - now).total_seconds(), 3600.0)
T = seconds_to_expiry / (365.25 * 86400.0)

if uploaded_file is not None:
    chain = pd.read_csv(uploaded_file).replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    chain = generate_demo_chain(spot, T, r_rate)

chain["DISTANCE"] = (chain["STRIKE"] - spot).abs()
atm = chain.loc[chain["DISTANCE"].idxmin()]
atm_strike = float(atm["STRIKE"])

pcr = chain["PUT_OI"].sum() / max(chain["CALL_OI"].sum(), 1)
atm_iv = (float(atm["CALL_IV"]) + float(atm["PUT_IV"])) / 2.0
atm_iv = 14.5 if atm_iv <= 0 else atm_iv
expected_move = spot * (atm_iv / 100.0) * np.sqrt(T)

# Compute Greeks across entire chain
call_deltas, put_deltas, gammas, vegas, call_thetas = [], [], [], [], []
for _, row in chain.iterrows():
    iv = max(row["CALL_IV"] if row["CALL_IV"] > 0 else atm_iv, 1.0) / 100.0
    _, c_del, gam, veg, c_th = bsm_full_metrics(spot, row["STRIKE"], T, iv, "CE", r_rate)
    _, p_del, _, _, _ = bsm_full_metrics(spot, row["STRIKE"], T, iv, "PE", r_rate)
    call_deltas.append(c_del)
    put_deltas.append(p_del)
    gammas.append(gam)
    vegas.append(veg)
    call_thetas.append(c_th)

chain["CALL_DELTA"] = call_deltas
chain["PUT_DELTA"] = put_deltas
chain["GAMMA"] = gammas
chain["VEGA"] = vegas
chain["CALL_THETA"] = call_thetas

chain["CALL_GEX_CR"] = chain["CALL_OI"] * chain["GAMMA"] * (spot**2) * 0.01 * NIFTY_LOT_SIZE / 1e7
chain["PUT_GEX_CR"] = -chain["PUT_OI"] * chain["GAMMA"] * (spot**2) * 0.01 * NIFTY_LOT_SIZE / 1e7
chain["NET_GEX_CR"] = chain["CALL_GEX_CR"] + chain["PUT_GEX_CR"]
chain["DEX_CR"] = (chain["CALL_OI"] * chain["CALL_DELTA"] + chain["PUT_OI"] * chain["PUT_DELTA"]) * spot * NIFTY_LOT_SIZE / 1e7

gex_total = round(chain["NET_GEX_CR"].sum(), 2)
dex_total = round(chain["DEX_CR"].sum(), 2)
call_wall = chain.loc[chain["CALL_GEX_CR"].idxmax(), "STRIKE"]
put_wall = chain.loc[chain["PUT_GEX_CR"].abs().idxmax(), "STRIKE"]

sorted_chain = chain.sort_values("STRIKE").reset_index(drop=True)
sign_changes = np.where(np.diff(np.sign(sorted_chain["NET_GEX_CR"])))[0]
gamma_flip = sorted_chain.loc[sign_changes[0], "STRIKE"] if len(sign_changes) > 0 else np.nan

# ----------------------------------------------------------------------
# HEADER & TOP KPIS
# ----------------------------------------------------------------------
st.title("NIFTY 50 Quant, Gamma Exposure & Erosion Engine")
st.caption(f"Active Expiry: **{expiry_date.strftime('%d %b %Y')}** ({days_to_expiry} DTE) | Last Updated: {now.strftime('%H:%M:%S IST')}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("NIFTY Spot", f"₹{spot:,.2f}", f"ATM: {atm_strike:.0f}")
c2.metric("Net Dealer GEX", f"₹{gex_total:,.2f} Cr", "Long Gamma" if gex_total > 0 else "Short Gamma")
c3.metric("Dealer Delta (DEX)", f"₹{dex_total:,.2f} Cr", f"PCR: {pcr:.2f}")
c4.metric("Expected Move", f"±{expected_move:.1f} pts", f"ATM IV: {atm_iv:.2f}%")
c5.metric("Call / Put Walls", f"{call_wall:.0f} / {put_wall:.0f}", f"Flip: {gamma_flip:.0f}" if pd.notna(gamma_flip) else "Flip: N/A")

st.divider()

# ----------------------------------------------------------------------
# TABS: 1. DEALER GEX PROFILE | 2. THETA EROSION & EXPIRY SIMULATOR
# ----------------------------------------------------------------------
tab1, tab2 = st.tabs(["Dealer Gamma Exposure (GEX)", "Theta Erosion & Expiry Simulator"])

# ======================================================================
# TAB 1: DEALER GEX PROFILE
# ======================================================================
with tab1:
    window = chain[(chain["STRIKE"] >= spot - strike_window) & (chain["STRIKE"] <= spot + strike_window)].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0e1117")
    ax1.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")

    width = 22
    ax1.bar(window["STRIKE"] - width/2, window["CALL_GEX_CR"], width=width, color="#00E676", alpha=0.85, label="Call Gamma (Dealer Long)")
    ax1.bar(window["STRIKE"] + width/2, window["PUT_GEX_CR"], width=width, color="#FF5252", alpha=0.85, label="Put Gamma (Dealer Short)")
    ax1.axvline(spot, color="#40C4FF", linestyle="--", linewidth=1.8, label=f"Spot: {spot:.1f}")
    ax1.axvline(call_wall, color="#1DE9B6", linestyle=":", linewidth=1.8, label=f"Call Wall: {call_wall:.0f}")
    ax1.axvline(put_wall, color="#FF4081", linestyle=":", linewidth=1.8, label=f"Put Wall: {put_wall:.0f}")
    ax1.set_title("Dealer Gamma Exposure by Strike (₹ Crores per 1% Move)", fontsize=12, fontweight="bold", color="white")
    ax1.set_ylabel("GEX (₹ Cr)", color="lightgray")
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.legend(loc="upper left", framealpha=0.3)

    ax2.plot(window["STRIKE"], window["NET_GEX_CR"], color="#82B1FF", linewidth=2.2, marker="o", markersize=4, label="Net Dealer Gamma")
    ax2.fill_between(window["STRIKE"], window["NET_GEX_CR"], 0, where=(window["NET_GEX_CR"] >= 0), color="#00E676", alpha=0.2, interpolate=True)
    ax2.fill_between(window["STRIKE"], window["NET_GEX_CR"], 0, where=(window["NET_GEX_CR"] < 0), color="#FF5252", alpha=0.2, interpolate=True)
    ax2.axhline(0, color="gray", linestyle="-", linewidth=1.0)
    ax2.axvline(spot, color="#40C4FF", linestyle="--", linewidth=1.5)
    if pd.notna(gamma_flip):
        ax2.axvline(gamma_flip, color="#FFAB40", linestyle="-.", linewidth=1.8, label=f"Gamma Flip: {gamma_flip:.0f}")

    ax2.set_xlabel("Strike Price", fontsize=11, color="lightgray")
    ax2.set_ylabel("Net GEX (₹ Cr)", color="lightgray")
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.legend(loc="upper left", framealpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Strike-by-Strike Greeks & Dealer Exposure Table")
    display_cols = ["STRIKE", "CALL_OI", "PUT_OI", "CALL_IV", "PUT_IV", "CALL_DELTA", "PUT_DELTA", "GAMMA", "CALL_GEX_CR", "PUT_GEX_CR", "NET_GEX_CR", "DEX_CR"]
    st.dataframe(window[display_cols].style.format("{:.2f}", subset=["CALL_IV", "PUT_IV", "CALL_DELTA", "PUT_DELTA", "CALL_GEX_CR", "PUT_GEX_CR", "NET_GEX_CR", "DEX_CR"]), use_container_width=True)

    csv_data = chain.to_csv(index=False).encode("utf-8")
    st.download_button("Download EOD Settled Chain (CSV)", data=csv_data, file_name=f"nifty_eod_chain_{now.strftime('%Y-%m-%d')}.csv", mime="text/csv")

# ======================================================================
# TAB 2: THETA EROSION & EXPIRY SIMULATOR
# ======================================================================
with tab2:
    st.subheader("Intraday Premium Erosion & Expiry Payoff Simulator")
    st.caption("Auto-populated from your live NIFTY option chain. Select any strike or override parameters below to stress-test Theta decay.")

    # Interactive Controls Row
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)

    strike_list = sorted(chain["STRIKE"].unique().tolist())
    default_strike_idx = strike_list.index(atm_strike) if atm_strike in strike_list else len(strike_list) // 2

    with col_p1:
        selected_strike = st.selectbox("Select Strike (K)", strike_list, index=default_strike_idx)
    with col_p2:
        opt_type = st.selectbox("Option Type", ["CE", "PE"], index=0)

    # Pull live chain defaults for the selected strike
    row_sel = chain[chain["STRIKE"] == selected_strike].iloc[0]
    default_prem = float(row_sel["CALL_LTP"] if opt_type == "CE" else row_sel["PUT_LTP"])
    default_iv = float(row_sel["CALL_IV"] if opt_type == "CE" else row_sel["PUT_IV"])
    if default_iv <= 0:
        default_iv = atm_iv
    if default_prem <= 0:
        default_prem = max(bsm_full_metrics(spot, selected_strike, max(T, 1/365), default_iv / 100.0, opt_type, r_rate)[0], 5.0)

    with col_p3:
        sim_premium = st.number_input("Entry Premium (₹)", value=round(default_prem, 2), step=1.0, min_value=0.05)
    with col_p4:
        sim_iv = st.number_input("Implied Volatility (%)", value=round(default_iv, 2), step=0.25, min_value=1.0)
    with col_p5:
        sim_dte = st.number_input("Days to Expiry (DTE)", value=max(float(days_to_expiry), 1.0), step=0.5, min_value=0.1)

    # Run Simulation Calculations
    sim_T = sim_dte / 365.0
    sim_vol = sim_iv / 100.0
    bs_fair, sim_delta, sim_gamma, sim_vega, true_daily_theta = bsm_full_metrics(spot, selected_strike, sim_T, sim_vol, opt_type, r_rate)

    intrinsic_now = max(spot - selected_strike, 0.0) if opt_type == "CE" else max(selected_strike - spot, 0.0)
    extrinsic_now = max(sim_premium - intrinsic_now, 0.0)
    move_to_beat_theta = true_daily_theta / max(abs(sim_delta), 1e-4)
    breakeven_spot = (selected_strike + sim_premium) if opt_type == "CE" else (selected_strike - sim_premium)

    # Erosion KPI Cards
    ek1, ek2, ek3, ek4, ek5 = st.columns(5)
    ek1.metric("Extrinsic (Time) Value", f"₹{extrinsic_now:.2f}", f"Intrinsic: ₹{intrinsic_now:.2f}")
    ek2.metric("True Daily Theta", f"-₹{true_daily_theta:.2f} / day", f"BSM Fair: ₹{bs_fair:.2f}")
    ek3.metric("Delta / Gamma", f"{sim_delta:.3f}", f"Γ: {sim_gamma:.5f}")
    ek4.metric("Move to Beat 1D Theta", f"{move_to_beat_theta:.1f} pts", "Spot move needed")
    ek5.metric("Expiry Breakeven", f"{breakeven_spot:,.2f}", f"{breakeven_spot - spot:+.1f} pts from spot")

    # Build Intraday Time Decay Schedule (0% to 90% elapsed)
    decay_fractions = np.round(np.arange(0.0, 1.0, 0.1), 1)
    decay_rows = []
    for frac in decay_fractions:
        rem_T = sim_T * (1.0 - frac)
        bs_val_t = bsm_full_metrics(spot, selected_strike, rem_T, sim_vol, opt_type, r_rate)[0]
        linear_prem = max(sim_premium - (true_daily_theta * frac * sim_dte), intrinsic_now)
        decay_rows.append({
            "Time Elapsed (%)": f"{int(frac*100)}%",
            "Remaining DTE": round(sim_dte * (1.0 - frac), 2),
            "Linear Erosion Prem (₹)": round(linear_prem, 2),
            "True BSM Prem (₹)": round(bs_val_t, 2),
            "Extrinsic Lost (₹)": round(sim_premium - bs_val_t, 2)
        })
    decay_df = pd.DataFrame(decay_rows)

    # Build Expiry Spot Simulation Grid (-200 to +200 pts in 20-pt steps)
    sim_spots = np.arange(round(spot / 20) * 20 - 200, round(spot / 20) * 20 + 220, 20)
    sim_rows = []
    for s_exp in sim_spots:
        intrinsic_exp = max(s_exp - selected_strike, 0.0) if opt_type == "CE" else max(selected_strike - s_exp, 0.0)
        u_move = s_exp - spot
        delta_move = sim_delta * u_move
        dg_move = (sim_delta * u_move) + (0.5 * sim_gamma * (u_move**2))
        net_pnl = intrinsic_exp - sim_premium

        sim_rows.append({
            "Expiry Spot": s_exp,
            "Intrinsic at Expiry (₹)": round(intrinsic_exp, 2),
            "Net Expiry PnL (₹)": round(net_pnl, 2),
            "Underlying Move (pts)": round(u_move, 2),
            "Option Move (Delta)": round(delta_move, 2),
            "Option Move (Delta+Gamma)": round(dg_move, 2),
            "Move to Beat Theta": round(move_to_beat_theta, 2)
        })
    sim_df = pd.DataFrame(sim_rows)

    # Dual Chart: Time Decay Curve + Expiry PnL Profile
    fig_e, (eax1, eax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    plt.style.use("dark_background")
    fig_e.patch.set_facecolor("#0e1117")
    eax1.set_facecolor("#0e1117")
    eax2.set_facecolor("#0e1117")

    x_ticks = np.arange(len(decay_df))
    eax1.plot(x_ticks, decay_df["Linear Erosion Prem (₹)"], linestyle="--", color="#FFAB40", marker="o", label="Linear Theta Decay")
    eax1.plot(x_ticks, decay_df["True BSM Prem (₹)"], linewidth=2.5, color="#40C4FF", marker="s", label="True BSM Decay (√t)")
    eax1.axhline(intrinsic_now, color="#00E676", linestyle=":", label=f"Intrinsic Floor (₹{intrinsic_now:.1f})")
    eax1.set_xticks(x_ticks)
    eax1.set_xticklabels(decay_df["Time Elapsed (%)"])
    eax1.set_title(f"Time Decay Trajectory ({selected_strike:.0f} {opt_type})", fontweight="bold", color="white")
    eax1.set_xlabel("Time Elapsed Toward Expiry", color="lightgray")
    eax1.set_ylabel("Option Premium (₹)", color="lightgray")
    eax1.grid(True, alpha=0.2)
    eax1.legend(framealpha=0.3)

    eax2.plot(sim_df["Expiry Spot"], sim_df["Net Expiry PnL (₹)"], linewidth=2.5, color="#82B1FF", label="Net PnL at Expiry (₹)")
    eax2.fill_between(sim_df["Expiry Spot"], sim_df["Net Expiry PnL (₹)"], 0,
                      where=(sim_df["Net Expiry PnL (₹)"] >= 0), color="#00E676", alpha=0.25, interpolate=True)
    eax2.fill_between(sim_df["Expiry Spot"], sim_df["Net Expiry PnL (₹)"], 0,
                      where=(sim_df["Net Expiry PnL (₹)"] < 0), color="#FF5252", alpha=0.25, interpolate=True)
    eax2.axhline(0, color="gray", linewidth=1)
    eax2.axvline(spot, color="#40C4FF", linestyle="--", label=f"Current Spot ({spot:.1f})")
    eax2.axvline(breakeven_spot, color="#FFAB40", linestyle="-.", label=f"Breakeven ({breakeven_spot:.1f})")
    eax2.set_title("Expiry Spot Simulation & Net PnL", fontweight="bold", color="white")
    eax2.set_xlabel("NIFTY Spot at Expiry", color="lightgray")
    eax2.set_ylabel("Profit / Loss (₹)", color="lightgray")
    eax2.grid(True, alpha=0.2)
    eax2.legend(framealpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_e)

    # Display Tables Side-by-Side
    t_col1, t_col2 = st.columns([1, 1.4])
    with t_col1:
        st.markdown("#### Intraday Time Decay Schedule")
        st.dataframe(decay_df, use_container_width=True, hide_index=True)
    with t_col2:
        st.markdown("#### Expiry Spot & Delta/Gamma Simulation Grid")
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
