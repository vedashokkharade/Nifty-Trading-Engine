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
st.sidebar.header("⚙️ Engine Controls")
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
st.title("📈 NIFTY 50 Gamma Exposure & Options Analytics Engine")
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
st.download_button("⬇️ Download EOD Settled Chain (CSV)", data=csv_data, file_name=f"nifty_eod_chain_{now.strftime('%Y-%m-%d')}.csv", mime="text/csv")

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
