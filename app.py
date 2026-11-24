import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
# CONSTANTS, TIMEZONE & NSE HOLIDAY SCHEDULE
# ----------------------------------------------------------------------
IST = ZoneInfo("Asia/Kolkata")
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

    cutoff_dt = datetime(candidate_expiry.year, candidate_expiry.month, candidate_expiry.day, 15, 30, 0, tzinfo=IST)
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

def bsm_vectorized_surface(S, K, T_grid, sigma_grid, option_type="CE", r=RISK_FREE_RATE):
    """Vectorized BSM valuation for 2D Theta vs. IV-Crush surface grids."""
    T_safe = np.maximum(T_grid, 1e-7)
    vol_safe = np.maximum(sigma_grid, 1e-4)

    d1 = (np.log(S / K) + (r + 0.5 * vol_safe**2) * T_safe) / (vol_safe * np.sqrt(T_safe))
    d2 = d1 - vol_safe * np.sqrt(T_safe)

    if option_type.upper() == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)

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
# GLOBAL DATA PROCESSING (IST TIMEZONE-AWARE)
# ----------------------------------------------------------------------
now = datetime.now(IST)
expiry_date = get_nifty_expiry(now)
days_to_expiry = max((expiry_date - now.date()).days, 0)
expiry_dt = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 15, 30, 0, tzinfo=IST)
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

c1, c2, c3, c4, c5 = st.
