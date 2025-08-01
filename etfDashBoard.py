import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import requests
from io import BytesIO

st.set_page_config(layout="wide")

# Markdown CSS
st.markdown(
    """
    <style>
    body {
        line-height: 1.0;
        padding-top: 0;
        padding-bottom: 0;
    }
    hr {
        margin: 0px 0px;
        padding: 0px 0px;
    }
    .stMarkdown {
        padding: 0 0;
    }
    .stHeadingWithActionElements {
        margin: 0 0;
        padding: 0 0;
    }
    .stHeading {
        margin: 0 0;
        padding-top: 1.0rem;
    }
    .stMainBlockContainer {
        line-height: 1.0;
        padding-top: 0;
        padding-bottom: 0;
    }
    .stElementContainer {
        line-height: 1.0;
        margin-top: 0;
        padding-top: 0;
    }
    .stVerticalBlock {
        line-height: 1.0;
        margin-top: 0;
        padding-top: 0;
        padding-bottom: 0; 
    }
    div[data-testid="column"] {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-box {
        padding: 0.75em 1em;
        border-radius: 6px;
        font-size: 0.9em;
    }
    .result-box.success {
        background-color: #1f3f2b;
        color: #d7ffd9;
    }
    .result-box.warning {
        background-color: #3f1f1f;
        color: #ffdada;
    }
    .isin-success {
        background-color: #1f3d2d;
        color: #d2f0d2;
        padding: 0.75em 1em;
        border-left: 4px solid #5cb85c;
        border-radius: 6px;
        font-weight: 500;
        margin-top: 0.5em;
        margin-bottom: 0;  
    }
    .isin-warning {
        background-color: #332323;
        color: #f8cfcf;
        padding: 0.75em 1.0em;
        border-left: 4px solid #e06666;
        border-radius: 6px;
        font-weight: 500;
        margin-top: 0.5em;
        margin-bottom: 0;
        height: 100%;   
    }
    
    
    div[data-testid="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }

    div[data-testid="stMarkdownContainer"] > .isin-success,
    div[data-testid="stMarkdownContainer"] > .isin-warning {
        margin-bottom: auto;
    }
       
    </style>
    """,
    unsafe_allow_html=True
)

#st.title("🇬🇧 ETF Summary Dashboard")

st.title("📈 ETF Summary Dashboard")

api_key = st.secrets["openfigi_api_key"]

# --- Initialise ETF input state if not set ---
if "etf_input_val" not in st.session_state:
    st.session_state.etf_input_val = "SGLS.L,CSH2.L,PHPP.L,ARMG.L,DAPP.L,BKCN.L,ARCI.L,CYBP.L,XAIX.L,NATP.L,DFNX.L,ESGB.L,WEBP.L,ARKK"

# XAIX.L,GBSP.L,BLKC.L,

# yfinance suffix mapping
yf_suffix_mapping = {
    'NYQ': '',     # NYSE
    'NYS': '',     # NYSE (alt code)
    'NMS': '',     # NASDAQ
    'NAS': '',     # NASDAQ (alt code)
    'ASE': '',     # AMEX
    'BATS': '',    # Cboe BATS

    'LSE': 'L',    # London Stock Exchange
    'LN': 'L',     # London Stock Exchange
    'I2': 'L',     # LSE IOB (International Order Book)
    
    'MIL': 'MI',   # Borsa Italiana (Milan)
    'XETR': 'DE',  # Deutsche Börse XETRA
    'FRA': 'F',    # Frankfurt
    'GER': 'DE',   # Germany (alt)
    
    'AMS': 'AS',   # Euronext Amsterdam
    'BRU': 'BR',   # Euronext Brussels
    'PAR': 'PA',   # Euronext Paris

    'TSE': 'T',    # Tokyo Stock Exchange
    'JPX': 'T',    # Japan Exchange Group

    'HKG': 'HK',   # Hong Kong Exchange
    'HKEX': 'HK',  # Hong Kong Exchange

    'TOR': 'TO',   # Toronto Stock Exchange
    'TSX': 'TO',   # Toronto Stock Exchange (alt)

    'ASX': 'AX',   # Australian Securities Exchange
    'NZE': 'NZ',   # New Zealand Exchange
    'SGX': 'SI',   # Singapore Exchange
}

# ETF input
etfs_input = st.text_input(
    "Enter ETF tickers to analyse (comma-separated):",
    value=st.session_state.etf_input_val,
    key="etf_input"
)
etf_list = [e.strip().upper() for e in etfs_input.split(",") if e.strip()]

# ---- ISIN to Ticker Layout ----
col1, col2, col3 = st.columns([0.35, 0.45, 0.2])

def isin_to_ticker(isin, api_key):
    url = "https://api.openfigi.com/v3/mapping"
    headers = {
        "Content-Type": "application/json",
        "X-OPENFIGI-APIKEY": api_key
    }
    payload = [{
        "idType": "ID_ISIN",
        "idValue": isin.upper()
    }]
    try:
        resp = requests.post(url, json=payload, headers=headers)
        data = resp.json()
        # Response is a list; first element contains .get('data')
        items = data[0].get("data", [])
        if items:
            # Return the ticker of the first match
            return items[0].get("ticker"), items[0].get("exchCode"), items[0].get("name")
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error fetching from OpenFIGI: {e}")
        return None, None, None

def map_openfigi_ticker_to_yf_ticker(foundTicker, openFigiExchangeCode):
    suffix = yf_suffix_mapping.get(openFigiExchangeCode)
    if suffix is None:
        # Log or handle unknown exchange code
        print(f"⚠️ Unknown OpenFIGI exchange code: '{openFigiExchangeCode}'")
        return foundTicker  # Fallback: return ticker without suffix
    return f"{foundTicker}.{suffix}" if suffix else foundTicker

# ISIN Lookup
with col1:
    isin_input = st.text_input("Lookup ETF ticker (Enter ISIN):", key="isin_input")

# --- Lookup logic ---
found_ticker, exchange, long_name, openFigiName = None, None, None, None
if isin_input:
    found_ticker, exchange, openFigiName = isin_to_ticker(isin_input.strip(), api_key)
    if found_ticker:
        # Convert OpenFIGI ticker to yfinance ticker using exchange code
        found_ticker = map_openfigi_ticker_to_yf_ticker(found_ticker, exchange)
        try:
            yf_info = yf.Ticker(found_ticker).info
            long_name = yf_info.get("longName") or yf_info.get("shortName")
        except Exception:
            long_name = "Name unavailable"

# --- Ticker Found Output ---
with col2:
    if isin_input:
        if found_ticker:
            # success component
            st.markdown(f"""
                <div class="isin-success">
                    <strong>Found:</strong> <strong>{found_ticker}</strong> on exchange: {exchange},<br><i>{long_name}</i> (yFinance) / <i>{openFigiName}</i> (OpenFIGI)
                </div>
            """, unsafe_allow_html=True)            
        else:
            # warning component
            st.markdown(f"""
                <div class="isin-warning">
                    <strong>No ticker symbol found for the provided ISIN.</strong><br>Check the ISIN is correct.
                </div>
            """, unsafe_allow_html=True)   

# --- Add Button ---
with col3:
    if isin_input and found_ticker:
        in_list = found_ticker in etf_list
        label = f"Add {found_ticker}" if not in_list else "Ticker Already In List!"
        if st.button(label, disabled=in_list):
            new_list = etf_list + [found_ticker] if found_ticker not in etf_list else etf_list
            st.session_state.etf_input_val = ", ".join(new_list)
            st.rerun()

momentum_periods = {
    "1 Week": 5,
    "1 Month": 21,
    "3 Months (with Sharpe Ratio)": 63
}

# Dataframes Helper Functions

@st.cache_data
def fetch_close_prices(tickers, total_days_needed):
    df = pd.DataFrame()
    invalid = []
    for t in tickers:
        try:
            data = yf.download(t, period="4mo", progress=False, auto_adjust=True)
            if data.empty or "Close" not in data or len(data) < total_days_needed + 1:
                invalid.append(t)
                continue
            df[t] = data["Close"]
        except Exception:
            invalid.append(t)
    return df, invalid

@st.cache_data
def fetch_names(tickers):
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            name = info.get("longName") or info.get("shortName") or "Unknown"
            names[t] = name
        except Exception:
            names[t] = "Unknown"
    return names

@st.cache_data
def fetch_volume_and_aum(_tickers):
    volume = {}
    aum = {}
    for t in _tickers:
        try:
            info = yf.Ticker(t).info
            volume[t] = info.get("volume", np.nan)
            aum[t] = info.get("totalAssets", np.nan)
        except Exception:
            volume[t] = np.nan
            aum[t] = np.nan
    return volume, aum

def calculate_max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def make_sparkline(series, figsize=(3.0, 0.3)):
    plt.figure(figsize=figsize)
    plt.plot(series, color='blue')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def style_momentum(df):
    return df.style.format({"Momentum %": "{:.2%}"})\
        .apply(lambda x: ["background-color: #c7f0d8" if v > 0 else "background-color: #f7c7c7" for v in x] if x.name == "Momentum %" else [""]*len(x), axis=0)\
        .set_properties(**{"font-size": "12px"})

def getSrColor(val):
    if val < 0.7:
        return "#ab0000"
    elif val >= 0.7 and val < 2.0:
        return "#ffffff"
    else:
        return "#30c06c"
    
def getMomentumColor(colIndex,val):
    # 1 week momentums
    if colIndex == 0:
        if val < -1.0:
            return "#a33636"
        elif val >= -1.0 and val < 2.0:
            return "#ffffff"
        else:
            return "#30c06c"
    # 1 month momentums
    elif colIndex == 1:
        if val < 0.0:
            return "#a33636"
        elif val >= 0.0 and val < 5.0:
            return "#ffffff"
        else:
            return "#30c06c"
    # 3 month momentums
    elif colIndex == 2:
        if val < 1.0:
            return "#a33636"
        elif val >= 1.0 and val < 10.0:
            return "#ffffff"
        else:
            return "#30c06c"

def getScoreColor(val):
    if val < 0.3:
        return "#ab0000"
    elif val >= 0.3 and val < 0.6:
        return "#ff891a"
    elif val >= 0.6 and val < 0.9:
        return "#ffffff"
    elif val >= 0.9 and val < 1.2:
        return "#e1ff00"
    else:
        return "#30c06c"

def getMaxDdColorDot(val):
    if val < 0.10:
        return f'<span style="color: #30c06c; font-size:14px;">●</span>'
    elif val >= 0.10 and val < 0.15:
        return f'<span style="color: #ffffff; font-size:14px;">●</span>'
    elif val >= 0.15 and val < 0.25:
        return f'<span style="color: #ff891a; font-size:14px;">●</span>'
    else:
        return f'<span style="color: #ab0000; font-size:14px;">●</span>'

def formatLargeNum(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

# Dataframe Build Logic
if etf_list:
    max_days = max(momentum_periods.values())
    with st.spinner("Fetching data..."):
        price_data, invalid = fetch_close_prices(etf_list, total_days_needed=max_days)
        names = fetch_names(etf_list)

    if invalid:
        st.warning(f"⚠️ These tickers had issues: {', '.join(invalid)}. Try manually looking up or removing any suffix from the ticker.")

    if not price_data.empty:
        cols = st.columns(4)
        for i, (label, days) in enumerate(momentum_periods.items()):
            
            interval_data = price_data.iloc[-days - 1 :].copy()

            # Momentum
            momentum = (interval_data.iloc[-1] / interval_data.iloc[0]) - 1

            # Sharpe Ratio
            returns = interval_data.pct_change(fill_method=None).dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

            spark_data = price_data.iloc[-days:].copy()

            # Build DataFrame
            data = pd.DataFrame({
                "Ticker": momentum.index,
                "Name": [names.get(t, "Unknown") for t in momentum.index],
                "Momentum %": momentum.values,
                "Sharpe Ratio": sharpe[momentum.index].round(2),
                "Sparkline": [make_sparkline(spark_data[t]) for t in momentum.index]
                }).sort_values(by="Momentum %", ascending=False).reset_index(drop=True)

            data = data.sort_values(by="Momentum %", ascending=False).reset_index(drop=True)

            # 1 Week / 1 Month / £ Month colums
            with cols[i]:
                st.markdown(f"<h5 style='margin-bottom: 2px; padding-bottom: 4px;'>{label}</h5>", unsafe_allow_html=True)
                for idx, row in data.iterrows():
                    spark_img = f"<img src='data:image/png;base64,{row['Sparkline']}' style='height:23px;'>"
                    mColor = getMomentumColor(i, (row['Momentum %'] * 100))
                    srString = f"{row['Sharpe Ratio']:.2f}" if i == 2 else ""
                    srColor = getSrColor(row['Sharpe Ratio'])
                    st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; align-items: center; margin: 0px 0;'>
                            <div style='flex: 2;'>
                                <strong style='font-size: 12px;'>{row['Ticker']}</strong><br>
                                <span style='font-size: 12px;'><i>{row['Name']}</i></span>
                            </div>
                            <div style='flex: 1; text-align: center; font-size: 16px; color: {mColor};'><strong>{row['Momentum %']:.2%}</strong></div>
                            <div style='flex: 0; text-align: center; padding-left: 0.2rem; font-size: 16px; color: {srColor}'>{srString}</div>
                        </div>
                        <div style='margin: 0; padding-top: 2px; padding-bottom: 1px;'>{spark_img}</div>
                        <hr style='margin: 2px 0;'>
                        """, unsafe_allow_html=True)
            
        # Top ETF Summary Tables
        with cols[3]:

            # Get 3-month period data again
            interval_data = price_data.iloc[-63 - 1 :].copy()
            returns = interval_data.pct_change(fill_method=None).dropna()
            sharpe_3mo = (returns.mean() / returns.std()) * np.sqrt(252)
            
            # "Consider Buying" Section
            st.markdown("<h5 style='margin-top: 0px;'>Consider Buy/Sell (Based on 3-Month Data)</h5>", unsafe_allow_html=True)

            # Compute 60/40 Sharpe/Momentum Combo Score
            momentum_3mo = (interval_data.iloc[-1] / interval_data.iloc[0]) - 1
            combo_score = (0.6 * sharpe_3mo) + (0.4 * momentum_3mo)

            # Fetch additional metrics (Volume, Assets Under Management)
            volume_data, aum_data = fetch_volume_and_aum(combo_score.index)

            # Compute Max Drawdown
            max_drawdowns = {}
            for t in combo_score.index:
                try:
                    dd = calculate_max_drawdown(price_data[t].iloc[-63:])
                    max_drawdowns[t] = dd
                except Exception as e:
                    st.warning(f"Drawdown error for {t}: {e}")
                    max_drawdowns[t] = np.nan

            rows = []
            for t in combo_score.index:
                rows.append({
                    "Ticker": t,
                    "Name": names.get(t, "Unknown"),
                    "Combo Score": combo_score[t],
                    "Volume": volume_data.get(t, np.nan),
                    "AUM": aum_data.get(t, np.nan),
                    "Max Drawdown": max_drawdowns.get(t, np.nan)
                })

            combo_df = pd.DataFrame(rows)
            combo_df = combo_df.sort_values(by="Combo Score", ascending=False).reset_index(drop=True)

            col_sizes = [1.1, 3.1, 1.1, 1.2, 1.1, 1.05]

            # 'Consider Buying' Headers
            header_cols = st.columns(col_sizes)
            headers = ["Ticker", "Name", "Combo", "Max DD", "Volume", "Assets"]
            for col, h in zip(header_cols, headers):
                col.markdown(f"<p style='font-size:12px; font-weight:bold; border-bottom: 1px solid #ccc; margin-bottom:2px;'>{h}</p>", unsafe_allow_html=True)

            # 'Consider Buying' ETF Data
            for idx, row in combo_df.iterrows():
                data_cols = st.columns(col_sizes)
                data_cols[0].markdown(f"<p style='font-size:11px'><strong>{row['Ticker']}</strong></p>", unsafe_allow_html=True)
                data_cols[1].markdown(f"<p style='font-size:11px'><i>{row['Name']}</i></p>", unsafe_allow_html=True)
                data_cols[2].markdown(f"<p style='font-size:14px; text-align: right; color: {getScoreColor(row['Combo Score'])};'><strong>{row['Combo Score']:.2f}</strong></p>", unsafe_allow_html=True)
                data_cols[3].markdown(f"<p style='font-size:13px; text-align: right;'>{getMaxDdColorDot(abs(row['Max Drawdown']))} {abs(row['Max Drawdown']):.1%}</p>", unsafe_allow_html=True)
                data_cols[4].markdown(f"<p style='font-size:11px; text-align: right;'>{formatLargeNum(int(row['Volume'])) if not np.isnan(row['Volume']) else 'N/A'}</p>", unsafe_allow_html=True)
                data_cols[5].markdown(f"<p style='font-size:11px; text-align: right;'>{('£'+formatLargeNum(int(row['AUM']))) if not np.isnan(row['AUM']) else 'N/A'}</p>", unsafe_allow_html=True)

            # Top ETFs by 3-Month Sharpe Ratio Table
            st.markdown("<h5 style='margin-bottom: 0px; padding-bottom: 4px;'>Top ETFs by 3-Month Sharpe Ratio</h5>", unsafe_allow_html=True)
            sharpe_df = pd.DataFrame({
                "Sharpe Ratio": sharpe_3mo
            }).sort_values(by="Sharpe Ratio", ascending=False)

            for t, row in sharpe_df.iterrows():
                name = names.get(t, "Unknown")
                srColor = getSrColor(row['Sharpe Ratio'])
                st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; margin: 2px 0;'>
                        <div>
                            <strong style='font-size: 11px;'>{t}</strong><br>
                            <span style='font-size: 12px;'><i>{name}</i></span>
                        </div>
                        <div style='text-align: right; font-size: 14px; color: {srColor};'>{row['Sharpe Ratio']:.2f}</div>
                    </div>
                    <hr style='margin: 1px 0;'>
                """, unsafe_allow_html=True)
            

# info = """
#     <h6>To Identify ETFs worth Buying:</h6>
#     <p>1. Momentum Filter:<br>
#     -Rank ETFs by 3-month price return (momentum).<br>
#     -Select top 20% or top 10 ETFs by return.
#     </p><br>
#     <p>2. Sharpe Ratio Filter:<br>
#     -Among those, rank by 3-month Sharpe Ratio.<br>
#     -Select top X (e.g., top 5 ETFs) with Sharpe > 1.0, preferably > 1.5.
#     </p><br>
#     <p>Optional 3. Additional Filters:<br>
#     -Exclude ETFs with:<br>
#       -Low volume or AUM (liquidity risk).<br>
#       -Extreme volatility (max drawdown filter)<br>
#       -Highly correlated assets (for diversification)
#     </p><br>
#     <h6>To Identify ETFs to Sell:</h6>
#     <p>1. Negative Momentum:<br>
#     -ETFs with negative or very low 3-month returns (e.g., < +2%).<br>
#     </p><br>
#     <p>2. Low or Negative Sharpe Ratio:<br>
#     -Sharpe < 0.5 — means return not compensating for risk.<br>
#     -Sharpe < 0 — return is negative on a risk-adjusted basis.
#     </p><br>
#     <p>3. Confirm with Trend:<br>
#     -Exclude ETFs with:<br>
#       -Price below 50-day or 100-day moving average? Another bearish sign.<br>
#       -Extreme volatility (max drawdown filter)<br>
#       -Increased drawdowns or volatility spikes? Red flag.
#     </p><br>

#     <h6>Example:</h6>
#     <p>Let’s say you have a universe of 100 ETFs. You:<br>
#     -Calculate 3-month returns and Sharpe Ratios.<br>
#     -Rank all ETFs by 3-month return. Take top 20 (top 20%).<br>
#     -From those, rank by Sharpe Ratio. Pick top 5 with Sharpe > 1.2.<br>
#     -Those 5 are your candidates to buy or hold.
#     </p><br>
#     <p>Meanwhile, you:<br>
#     -Check your current holdings.<br>
#     -Sell ETFs that have dropped out of the top 40% in momentum and have Sharpe < 0.5.
#     </p><br>

#     <h6>Backtest-Proven Variants (Used in Real Strategies):</h6>
#     <p>
#     -Dual Momentum: Combines relative strength (vs. other ETFs) and absolute momentum (vs. cash).<br>
#     -Risk-Adjusted Momentum: Return / Volatility instead of raw return. Sometimes called "volatility-adjusted momentum."<br>
#     -Sharpe Momentum Combo Score: Weight Sharpe and momentum equally or 60/40, then rank.
#     </p><br>
# """

# st.markdown(info, unsafe_allow_html=True)
