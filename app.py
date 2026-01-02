# app.py
# pip install streamlit requests scipy

import math
import time
import requests
import streamlit as st
from datetime import datetime, timezone
from scipy.stats import norm

DERIBIT_HTTP = "https://www.deribit.com/api/v2"  # switch to https://test.deribit.com/api/v2 for testnet


def rpc(method: str, params: dict | None = None, timeout=10):
    payload = {"jsonrpc": "2.0", "id": int(time.time() * 1000), "method": method, "params": params or {}}
    r = requests.post(DERIBIT_HTTP, json=payload, timeout=timeout)

    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} from Deribit: {r.text}")

    data = r.json()
    if "error" in data:
        raise RuntimeError(f"Deribit error: {data['error']}")
    return data["result"]

@st.cache_data(ttl=3)
def get_ticker(instr: str):
    return rpc("public/ticker", {"instrument_name": instr})


@st.cache_data(ttl=3)
def get_btc_index_price():
    return rpc("public/get_index_price", {"index_name": "btc_usd"})["index_price"]


@st.cache_data(ttl=60)
def load_btc_option_instruments():
    inst = rpc("public/get_instruments", {"currency": "BTC", "kind": "option", "expired": False})
    out = []
    for x in inst:
        out.append(
            {
                "instrument_name": x["instrument_name"],
                "expiration_timestamp": x["expiration_timestamp"],
                "strike": x["strike"],
                "option_type": x["option_type"],  # "call" / "put"
            }
        )
    out.sort(key=lambda z: (z["expiration_timestamp"], z["strike"], z["option_type"]))
    return out


def deribit_inverse_bs_price_btc(F, K, T, sigma, is_call: bool):
    """
    Deribit support formula output is in BTC:
      Call: C = N(d1) - (K/F)*N(d2)
      Put : P = (K/F)*N(-d2) - N(-d1)
    """
    if F <= 0 or K <= 0:
        return 0.0

    if T <= 0 or sigma <= 0:
        intrinsic_usd = max(0.0, F - K) if is_call else max(0.0, K - F)
        return intrinsic_usd / F

    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        return norm.cdf(d1) - (K / F) * norm.cdf(d2)
    else:
        return (K / F) * norm.cdf(-d2) - norm.cdf(-d1)


# ---------------- UI ----------------
st.set_page_config(page_title="Deribit BTC Inverse Option Calculator", layout="wide")
st.title("Deribit BTC Inverse Option Calculator")

instruments = load_btc_option_instruments()

expiries = sorted({x["expiration_timestamp"] for x in instruments})
expiry_labels = {
    ts: datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%d %b %Y")
    for ts in expiries
}

colA, colB, colC, colD = st.columns([1.2, 1.2, 1.0, 1.0])

with colA:
    expiry_ts = st.selectbox("Expiry", expiries, format_func=lambda ts: expiry_labels[ts])

filtered_by_expiry = [x for x in instruments if x["expiration_timestamp"] == expiry_ts]
strikes = sorted({int(x["strike"]) for x in filtered_by_expiry})

with colC:
    opt_type = st.selectbox("Option Type", ["call", "put"])

# We need a forward (F_live) to choose ATM strike.
# We do it in 2 steps:
# 1) pick any instrument for this expiry (based on option type) to read underlying_price (= forward)
# 2) select ATM strike based on that forward, then re-fetch ticker for the exact selected strike

# Pick a sample instrument for this expiry + option type (any strike is fine)
sample_candidates = [x for x in filtered_by_expiry if x["option_type"] == opt_type]
if not sample_candidates:
    st.error("No options found for that expiry/type (unexpected).")
    st.stop()

sample_instr = sample_candidates[0]["instrument_name"]

# (Optional, only for display) - derived label, may not exist as a real future instrument
parts = sample_instr.split("-")
future_name = f"{parts[0]}-{parts[1]}"

# Fetch sample ticker to obtain Deribit forward (underlying_price)
sample_ticker = get_ticker(sample_instr)

# Deribit forward (F) for option pricing (this matches the option chain header "Underlying future")
F_live = float(sample_ticker.get("underlying_price") or 0.0)
if F_live <= 0:
    F_live = float(get_btc_index_price())  # fallback

# Auto-select ATM strike closest to forward
if strikes:
    atm_strike = min(strikes, key=lambda k: abs(k - int(round(F_live))))
    atm_index = strikes.index(atm_strike)
else:
    atm_index = 0

with colB:
    strike = st.selectbox("Strike (USD)", strikes, index=atm_index, format_func=lambda s: f"{s:,}")

# Now find the exact instrument for expiry/strike/type
matches = [
    x for x in filtered_by_expiry
    if int(x["strike"]) == int(strike) and x["option_type"] == opt_type
]
if not matches:
    st.error("No instrument found for that expiry/strike/type (unexpected).")
    st.stop()

instrument_name = matches[0]["instrument_name"]
ticker = get_ticker(instrument_name)

# Update F_live using the exact selected instrument (should be same, but keep it consistent)
F_live = float(ticker.get("underlying_price") or F_live)

# Deribit forward (F) for option pricing:
# This is the "Underlying future" shown on the option chain header.
F_live = float(ticker.get("underlying_price") or 0.0)

# Fallback if missing (rare)
if F_live <= 0:
    F_live = float(get_btc_index_price())

with colD:
    st.text_input("Selected Instrument", instrument_name, disabled=True)

index_price = get_btc_index_price()

mark_iv_pct = ticker.get("mark_iv", None)          # % (e.g. 47.1)
underlying_price = ticker.get("underlying_price", None)
mark_price_btc = ticker.get("mark_price", None)    # BTC premium (inverse option mark)

st.caption("Live data")
m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC Index (btc_usd)", f"{index_price:,.2f}")
m2.metric("Underlying future", f"{F_live:,.2f}")
if mark_iv_pct is not None:
    m3.metric("Option Mark IV (%)", f"{float(mark_iv_pct):.2f}")
if mark_price_btc is not None:
    m4.metric("Deribit Mark Price (BTC)", f"{float(mark_price_btc):.8f}")

st.caption("Forward used for pricing = option ticker underlying_price (Deribit 'Underlying future')")

st.divider()

# Time to expiry (Deribit uses delivery/expiry at 08:00 UTC in their description,
# but Deribit API expiration_timestamp already represents the instrument expiry timestamp.
now_ms = int(time.time() * 1000)
T_sec = max(0.0, (expiry_ts - now_ms) / 1000.0)
T_years = T_sec / (365.0 * 24 * 3600)

left, right = st.columns([1.2, 1.0])

with left:
    # Load Default Data button
    load_default = st.button("Load Default Data", type="secondary")

    # Initialize session state for inputs (empty strings initially)
    if 'forward_price_text' not in st.session_state:
        st.session_state.forward_price_text = ""
    if 'iv_text' not in st.session_state:
        st.session_state.iv_text = ""

    # Set default values when button is clicked
    if load_default:
        default_forward = float(F_live) if F_live > 0 else float(index_price)
        default_iv = float(mark_iv_pct) if mark_iv_pct is not None else 50.0
        st.session_state.forward_price_text = f"{default_forward:.2f}"
        st.session_state.iv_text = f"{default_iv:.2f}"

    forward_text = st.text_input(
        "Forward price F (editable)",
        value=st.session_state.forward_price_text,
        placeholder="Enter forward price"
    )
    st.session_state.forward_price_text = forward_text

    iv_text = st.text_input(
        "Implied Vol (%)",
        value=st.session_state.iv_text,
        placeholder="Enter implied volatility %"
    )
    st.session_state.iv_text = iv_text

    # Parse values for calculations
    try:
        expected_forward = float(forward_text) if forward_text else None
    except ValueError:
        expected_forward = None
        if forward_text:
            st.error("Invalid forward price. Please enter a number.")

    try:
        iv_pct = float(iv_text) if iv_text else None
    except ValueError:
        iv_pct = None
        if iv_text:
            st.error("Invalid implied volatility. Please enter a number.")

with right:
    # Calculate days, hours, minutes
    days = int(T_sec // (24 * 3600))
    remaining_sec = T_sec % (24 * 3600)
    hours = int(remaining_sec // 3600)
    minutes = int((remaining_sec % 3600) // 60)

    # Determine if quarterly (March, June, September, December)
    expiry_datetime = datetime.fromtimestamp(expiry_ts / 1000, tz=timezone.utc)
    is_quarterly = expiry_datetime.month in [3, 6, 9, 12]
    quarterly_text = " (Quarterly)" if is_quarterly else ""

    st.write(f"Time to Expiry: {days}d {hours}h {minutes}m{quarterly_text}")

calc = st.button("Calculate", type="primary")

if calc:
    # Validate inputs
    if expected_forward is None or expected_forward <= 0:
        st.error("Please enter a valid Forward price (must be > 0)")
    elif iv_pct is None or iv_pct <= 0:
        st.error("Please enter a valid Implied Vol % (must be > 0)")
    else:
        K = float(strike)
        sigma = float(iv_pct) / 100.0
        is_call = (opt_type == "call")

        # Calculate using Underlying future from API
        F_market = float(F_live) if F_live > 0 else float(index_price)
        btc_premium_market = deribit_inverse_bs_price_btc(F=F_market, K=K, T=T_years, sigma=sigma, is_call=is_call)
        usd_equiv_market = btc_premium_market * F_market

        # Calculate using user-entered Forward price
        F_user = float(expected_forward)
        btc_premium_user = deribit_inverse_bs_price_btc(F=F_user, K=K, T=T_years, sigma=sigma, is_call=is_call)
        usd_equiv_user = btc_premium_user * F_user

        # Results using Underlying future
        st.subheader("Results - Underlying future")
        r1, r2 = st.columns(2)
        r1.metric("Theoretical Option Value (BTC)", f"{btc_premium_market:.4f}")
        r2.metric("USD Equivalent (using F)", f"{usd_equiv_market:,.2f}")

        st.divider()

        # Results using user-entered Forward price
        st.subheader("Results - Custom Forward Price")
        r3, r4 = st.columns(2)
        r3.metric("Theoretical Option Value (BTC)", f"{btc_premium_user:.4f}")
        r4.metric("USD Equivalent (using F)", f"{usd_equiv_user:,.2f}")
