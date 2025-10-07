# markets_advanced_complete.py
"""
Advanced Market Analysis with:
- Separate analysis by trailer type (Van, Reefer, Flatbed)
- Statistical significance filtering (minimum volume thresholds)
- Seasonal forecasting with Holt-Winters
- Volume-weighted shock detection
- Dynamic equilibrium (adaptive, rolling)
- Multiple timeframe support (Weekly, Monthly, Quarterly)
- Future heat anticipation signals
- Regional heat maps
- Markov Regime Switching analytical tool
- Configurable market rankings
- Specific future period analysis
- ENHANCED: Single market selection in filter pane with state information
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Advanced Market Analysis")

st.title("🔄 Dynamic Market Analysis with Regime Detection")
st.markdown("**Adaptive equilibrium with heat maps & Markov regime switching**")


# -------------------------
# Data Loading Functions (Same as before)
# -------------------------

@st.cache_data
def read_excel_safely(uploaded):
    try:
        df = pd.read_excel(uploaded)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def load_lt(df_raw):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "date" in lc:
            col_map[c] = "date"
        elif "type" in lc and "trailer" in lc:
            col_map[c] = "type_of_trailer"
        elif "market" in lc:
            col_map[c] = "market"
        elif "load" in lc and "ratio" not in lc:
            col_map[c] = "loads"
        elif "truck" in lc:
            col_map[c] = "trucks"
        elif "ratio" in lc:
            col_map[c] = "ratio"
    df = df.rename(columns=col_map)
    df = df.replace(["No Match", "#DIV/0!", "#VALUE!", "#N/A", "NA", "N/A", "", " "], np.nan)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["type_of_trailer"] = df["type_of_trailer"].astype(str).str.strip().str.upper()
    df["market"] = df["market"].astype(str).str.strip()
    for col in ["loads", "trucks", "ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    mask_calc = df["ratio"].isna() & df["loads"].notna() & df["trucks"].notna() & (df["trucks"] != 0)
    df.loc[mask_calc, "ratio"] = df.loc[mask_calc, "loads"] / df.loc[mask_calc, "trucks"]
    df.loc[df["trucks"] == 0, "ratio"] = np.nan
    df = df.sort_values(["market", "type_of_trailer", "date"]).reset_index(drop=True)
    return df


def load_map(df_raw):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "market" in lc:
            col_map[c] = "Market"
        elif lc in ("abb", "abbr", "state_code", "stateabbr", "state_abbr"):
            col_map[c] = "Abb"
        elif "state" == lc:
            col_map[c] = "State"
    df = df.rename(columns=col_map)
    df["Market"] = df["Market"].astype(str).str.strip()
    if "Abb" in df.columns:
        df["Abb"] = df["Abb"].astype(str).str.strip().str.upper()
    return df


def safe_interpolate_group(g):
    g = g.copy()
    for c in ["loads", "trucks", "ratio"]:
        if c in g.columns:
            g[c] = g[c].interpolate(limit_direction="both")
            g[c] = g[c].fillna(method="ffill").fillna(method="bfill")
    return g


def preprocess_and_reindex(lt_df):
    df = lt_df.copy()
    all_frames = []
    for (market, typ), g in df.groupby(["market", "type_of_trailer"]):
        g = g.set_index("date").sort_index()
        if pd.isna(g.index.min()) or pd.isna(g.index.max()):
            continue
        full_idx = pd.date_range(start=g.index.min(), end=g.index.max(), freq="D")
        g = g.reindex(full_idx)
        g["market"] = market
        g["type_of_trailer"] = typ
        g = safe_interpolate_group(g)
        g = g.reset_index().rename(columns={"index": "date"})
        all_frames.append(g)
    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=df.columns)


def aggregate_df(df, freq):
    """Enhanced aggregation with multiple timeframe support"""
    out = []
    for (market, typ), g in df.groupby(["market", "type_of_trailer"]):
        g = g.set_index("date").sort_index()

        if freq == "W":
            res = g.resample("W-MON").agg({"loads": "sum", "trucks": "sum", "ratio": "mean"}).reset_index()
        elif freq == "M":
            res = g.resample("M").agg({"loads": "sum", "trucks": "sum", "ratio": "mean"}).reset_index()
        elif freq == "Q":
            res = g.resample("Q").agg({"loads": "sum", "trucks": "sum", "ratio": "mean"}).reset_index()
        else:
            res = g.resample("M").agg({"loads": "sum", "trucks": "sum", "ratio": "mean"}).reset_index()

        res["market"] = market
        res["type_of_trailer"] = typ
        out.append(res)

    if out:
        return pd.concat(out, ignore_index=True)
    else:
        return pd.DataFrame(columns=["date", "loads", "trucks", "ratio", "market", "type_of_trailer"])


# -------------------------
# ENHANCEMENT: Create market selection with state information
# -------------------------

def create_market_selection_with_states(analysis_df, map_df):
    """
    Create enhanced market selection with state information
    """
    if analysis_df.empty or map_df.empty:
        return None, None

    # Merge market data with state information
    market_state_data = analysis_df.merge(
        map_df[['Market', 'State', 'Abb']].drop_duplicates(),
        left_on='market', right_on='Market', how='left'
    )

    # Create display names with state information
    market_state_data['display_name'] = market_state_data.apply(
        lambda x: f"{x['market']} ({x['State']})" if pd.notna(x['State']) else x['market'],
        axis=1
    )

    # Sort by state then market for better organization
    market_state_data = market_state_data.sort_values(['State', 'market'])

    return market_state_data


# -------------------------
# ENHANCEMENT 1: Markov Regime Switching Function - FIXED VERSION
# -------------------------

def markov_regime_switching_analysis(ts_data, n_regimes=3):
    """
    Markov Regime Switching Analysis using Gaussian Mixture Models
    Provides statistical regime detection with transition probabilities

    KEY FIX: Corrected regime stability calculation to use diagonal of transition matrix
    instead of incorrect transition counting logic
    """
    if len(ts_data) < 24:
        return {
            'regimes': pd.Series(["insufficient_data"] * len(ts_data), index=ts_data.index),
            'transition_matrix': None,
            'regime_means': None,
            'regime_persistence': None,
            'current_regime': "insufficient_data",
            'regime_stability': 0.0
        }

    try:
        from sklearn.mixture import GaussianMixture

        # Prepare data for GMM (simplified Markov switching)
        X = ts_data.values.reshape(-1, 1)

        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type='full')
        regimes_numeric = gmm.fit_predict(X)

        # Calculate regime probabilities and means
        regime_probs = gmm.predict_proba(X)
        regime_means = {f"regime_{i}": float(gmm.means_[i][0]) for i in range(n_regimes)}

        # Create regime labels based on mean values
        sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1])
        regime_mapping = {}
        for i, (regime_key, mean_val) in enumerate(sorted_regimes):
            if i == 0:
                regime_mapping[list(regime_means.keys()).index(regime_key)] = "low_volatility"
            elif i == len(sorted_regimes) - 1:
                regime_mapping[list(regime_means.keys()).index(regime_key)] = "high_volatility"
            else:
                regime_mapping[list(regime_means.keys()).index(regime_key)] = "normal_volatility"

        # Apply regime labels
        regime_labels = [regime_mapping[regime] for regime in regimes_numeric]
        regimes = pd.Series(regime_labels, index=ts_data.index)

        # Calculate transition probabilities
        transitions = []
        for i in range(1, len(regimes)):
            transitions.append((regimes.iloc[i - 1], regimes.iloc[i]))

        if transitions:
            transition_matrix = pd.crosstab(
                pd.Series([t[0] for t in transitions]),
                pd.Series([t[1] for t in transitions]),
                normalize='index'
            ).round(3)
        else:
            transition_matrix = None

        # Calculate regime persistence (average duration in each regime)
        regime_persistence = {}
        for regime in regimes.unique():
            regime_data = regimes[regimes == regime]
            if len(regime_data) > 1:
                persistence = regime_data.groupby((regime_data != regime_data.shift()).cumsum()).size().mean()
            else:
                persistence = 1.0
            regime_persistence[regime] = persistence

        # ✅ FIXED: Correct regime stability calculation
        current_regime = regimes.iloc[-1] if not regimes.empty else "unknown"
        if transition_matrix is not None and current_regime in transition_matrix.index:
            # Stability = probability of staying in current regime
            stability_score = transition_matrix.loc[current_regime, current_regime]
        else:
            # Fallback: use persistence measure
            stability_score = regime_persistence.get(current_regime, 0.0) / 10.0  # Normalize persistence

        # Ensure stability is between 0 and 1
        stability_score = max(0.0, min(1.0, stability_score))

        return {
            'regimes': regimes,
            'transition_matrix': transition_matrix,
            'regime_means': regime_means,
            'regime_persistence': regime_persistence,
            'current_regime': current_regime,
            'regime_stability': stability_score
        }

    except Exception as e:
        # Fallback to simple regime detection
        st.warning(f"Markov switching failed: {e}. Using fallback method.")
        return simple_regime_detection_enhanced(ts_data)


def simple_regime_detection_enhanced(ts_data):
    """
    Enhanced simple regime detection with more statistical rigor
    and FIXED stability calculation
    """
    if len(ts_data) < 12:
        return {
            'regimes': pd.Series(["insufficient_data"] * len(ts_data), index=ts_data.index),
            'transition_matrix': None,
            'regime_means': None,
            'regime_persistence': None,
            'current_regime': "insufficient_data",
            'regime_stability': 0.0
        }

    # Statistical threshold-based regime detection
    median = ts_data.median()
    std = ts_data.std()
    q25, q75 = np.percentile(ts_data, [25, 75])

    regimes = []
    for value in ts_data:
        if value > q75 + 0.5 * std:
            regimes.append("high_volatility")
        elif value < q25 - 0.5 * std:
            regimes.append("low_volatility")
        else:
            regimes.append("normal_volatility")

    regimes_series = pd.Series(regimes, index=ts_data.index)

    # Calculate basic transition probabilities
    transitions = []
    for i in range(1, len(regimes)):
        transitions.append((regimes[i - 1], regimes[i]))

    if transitions:
        transition_matrix = pd.crosstab(
            pd.Series([t[0] for t in transitions]),
            pd.Series([t[1] for t in transitions]),
            normalize='index'
        ).round(3)
    else:
        transition_matrix = None

    # ✅ FIXED: Correct stability calculation for simple method
    current_regime = regimes[-1] if regimes else "unknown"
    if transition_matrix is not None and current_regime in transition_matrix.index:
        stability_score = transition_matrix.loc[current_regime, current_regime]
    else:
        # Estimate stability based on regime distribution
        regime_counts = pd.Series(regimes).value_counts(normalize=True)
        stability_score = regime_counts.get(current_regime, 0.5)

    # Simplified regime means
    regime_means = {
        'low_volatility': ts_data[regimes_series == "low_volatility"].mean() if any(
            r == "low_volatility" for r in regimes) else median - std,
        'normal_volatility': ts_data[regimes_series == "normal_volatility"].mean() if any(
            r == "normal_volatility" for r in regimes) else median,
        'high_volatility': ts_data[regimes_series == "high_volatility"].mean() if any(
            r == "high_volatility" for r in regimes) else median + std
    }

    # Calculate actual persistence
    regime_persistence = {}
    for regime in regimes_series.unique():
        regime_data = regimes_series[regimes_series == regime]
        if len(regime_data) > 1:
            persistence = regime_data.groupby((regime_data != regime_data.shift()).cumsum()).size().mean()
        else:
            persistence = 1.0
        regime_persistence[regime] = persistence

    return {
        'regimes': regimes_series,
        'transition_matrix': transition_matrix,
        'regime_means': regime_means,
        'regime_persistence': regime_persistence,
        'current_regime': current_regime,
        'regime_stability': stability_score
    }


# -------------------------
# ENHANCEMENT 2: Extended Future Period Analysis
# -------------------------

def analyze_future_period(future_signals_df, start_date, end_date, analysis_type='heated', top_n=10):
    """
    Analyze specific future period for market conditions
    """
    # Filter for the specific period
    period_signals = future_signals_df[
        (future_signals_df['forecast_date'] >= pd.Timestamp(start_date)) &
        (future_signals_df['forecast_date'] <= pd.Timestamp(end_date))
        ]

    if period_signals.empty:
        return pd.DataFrame()

    if analysis_type == 'heated':
        # Find markets with highest average deviation in the period
        period_analysis = period_signals.groupby('market').agg({
            'deviation_sigma': 'mean',
            'forecast_ratio': 'mean',
            'volatility': 'mean',
            'trailer_type': 'first'
        }).reset_index()

        period_analysis['abs_deviation'] = period_analysis['deviation_sigma'].abs()
        result = period_analysis.nlargest(top_n, 'abs_deviation')

    elif analysis_type == 'volatile':
        # Find markets with highest volatility in the period
        period_analysis = period_signals.groupby('market').agg({
            'volatility': 'mean',
            'deviation_sigma': 'mean',
            'forecast_ratio': 'mean',
            'trailer_type': 'first'
        }).reset_index()

        result = period_analysis.nlargest(top_n, 'volatility')

    elif analysis_type == 'stable':
        # Find markets with lowest volatility in the period
        period_analysis = period_signals.groupby('market').agg({
            'volatility': 'mean',
            'deviation_sigma': 'mean',
            'forecast_ratio': 'mean',
            'trailer_type': 'first'
        }).reset_index()

        result = period_analysis.nsmallest(top_n, 'volatility')

    return result


# -------------------------
# Existing Analysis Functions (Keeping everything else the same)
# -------------------------

def compute_dynamic_equilibrium(ts_data, window=12, method='adaptive'):
    """
    Compute dynamic equilibrium that adapts to market shifts
    """
    ratios = ts_data.dropna()

    if len(ratios) < window:
        return pd.Series([ratios.median()] * len(ratios), index=ratios.index), ratios.median(), []

    if method == 'adaptive':
        equilibrium = []
        regime_shifts = []  # Track potential regime changes

        for i in range(len(ratios)):
            if i < window:
                current_data = ratios.iloc[:i + 1]
                eq = current_data.median()
                regime = "initial"
            else:
                window_data = ratios.iloc[i - window:i + 1]
                first_half = window_data.iloc[:window // 2].median()
                second_half = window_data.iloc[window // 2:].median()
                regime_change = abs(second_half - first_half) > window_data.std()

                if regime_change:
                    eq = second_half
                    regime = "shift"
                else:
                    eq = window_data.median()
                    regime = "stable"

                regime_shifts.append(regime)

            equilibrium.append(eq)

        equilibrium = pd.Series(equilibrium, index=ratios.index)
        current_eq = equilibrium.iloc[-1]

    return equilibrium, current_eq, regime_shifts


def simple_regime_detection(ts_data, n_regimes=2):
    """
    Simple regime detection (placeholder for Markov switching)
    This will be replaced with proper Markov regime switching model
    """
    if len(ts_data) < 24:
        return pd.Series(["single_regime"] * len(ts_data), index=ts_data.index)

    # Simple threshold-based regime detection
    median = ts_data.median()
    std = ts_data.std()

    regimes = []
    for value in ts_data:
        if value > median + 0.5 * std:
            regimes.append("high_regime")
        elif value < median - 0.5 * std:
            regimes.append("low_regime")
        else:
            regimes.append("normal_regime")

    return pd.Series(regimes, index=ts_data.index)


def fit_seasonal_forecast(series, steps=12, seasonal_periods=12):
    """Holt-Winters seasonal forecasting"""
    if len(series) < 2 * seasonal_periods:
        last_year = series.iloc[-seasonal_periods:] if len(series) >= seasonal_periods else series
        forecast_values = []
        for i in range(steps):
            forecast_values.append(last_year.iloc[i % len(last_year)] if len(last_year) > 0 else series.iloc[-1])
        return pd.Series(forecast_values,
                         index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=30),
                                             periods=steps, freq='M'))

    try:
        model = ExponentialSmoothing(
            series,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps)
        return forecast
    except:
        last_year = series.iloc[-seasonal_periods:]
        forecast_values = []
        for i in range(steps):
            forecast_values.append(last_year.iloc[i % len(last_year)])
        return pd.Series(forecast_values,
                         index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=30),
                                             periods=steps, freq='M'))


def analyze_market_by_type_dynamic(lt_agg, trailer_type, timeframe, min_significance=0.3):
    """Enhanced analysis with dynamic equilibrium and future signals"""
    type_data = lt_agg[lt_agg['type_of_trailer'] == trailer_type]

    markets_analysis = []
    future_signals = []

    for market in type_data['market'].unique():
        market_data = type_data[type_data['market'] == market].set_index('date').sort_index()

        if len(market_data) < 16:
            continue

        # Calculate dynamic equilibrium
        ts_data = market_data['ratio'].dropna()
        dynamic_eq_series, current_equilibrium, regime_shifts = compute_dynamic_equilibrium(ts_data, window=12,
                                                                                            method='adaptive')

        # ENHANCEMENT: Use Markov Regime Switching instead of simple detection
        regime_analysis = markov_regime_switching_analysis(ts_data)

        # Calculate dynamic volatility
        volatility = ts_data.rolling(window=12, min_periods=1).apply(
            lambda x: np.percentile(x, 75) - np.percentile(x, 25)
        ).iloc[-1]

        # Significance based on volume consistency
        volume_data = market_data['trucks'].dropna()
        volume_cv = volume_data.std() / volume_data.mean() if volume_data.mean() > 0 else 1
        significance_score = 1 / (1 + volume_cv)

        if significance_score >= min_significance:
            current_ratio = ts_data.iloc[-1]
            current_deviation = (current_ratio - current_equilibrium) / volatility if volatility > 0 else 0

            # Generate seasonal forecast
            forecast = fit_seasonal_forecast(ts_data, steps=24)  # Extended to 24 months for future analysis

            # Future signals for heat anticipation
            for i, fcst in enumerate(forecast):
                future_dev = (fcst - current_equilibrium) / volatility if volatility > 0 else 0
                future_signals.append({
                    'market': market,
                    'trailer_type': trailer_type,
                    'forecast_date': forecast.index[i],
                    'forecast_ratio': fcst,
                    'deviation_sigma': future_dev,
                    'volatility': volatility,  # Include current volatility for future analysis
                    'timeframe': timeframe
                })

            markets_analysis.append({
                'market': market,
                'trailer_type': trailer_type,
                'current_ratio': current_ratio,
                'dynamic_equilibrium': current_equilibrium,
                'deviation_sigma': current_deviation,
                'volatility': volatility,
                'significance': significance_score,
                'avg_volume': volume_data.mean(),
                'timeframe': timeframe,
                'current_regime': regime_analysis['current_regime'],
                'regime_stability': regime_analysis['regime_stability'],
                'transition_matrix': regime_analysis['transition_matrix'],
                'regime_means': regime_analysis['regime_means'],
                'forecast_trend': 'increasing' if forecast.iloc[-1] > current_ratio else 'decreasing'
            })

    analysis_df = pd.DataFrame(markets_analysis)
    future_df = pd.DataFrame(future_signals)

    return analysis_df, future_df


def identify_future_heated_markets(future_signals_df, lookahead_periods=4, heat_threshold=2.0):
    """Identify markets expected to be heated in future periods"""
    future_cutoff = datetime.now() + timedelta(days=30 * lookahead_periods)

    future_signals = future_signals_df[
        (future_signals_df['forecast_date'] <= future_cutoff) &
        (future_signals_df['deviation_sigma'].abs() >= heat_threshold)
        ]

    if future_signals.empty:
        return pd.DataFrame()

    heated_markets = []
    for market in future_signals['market'].unique():
        market_signals = future_signals[future_signals['market'] == market]
        first_heat = market_signals.loc[market_signals['forecast_date'].idxmin()]

        heat_periods = len(market_signals)
        avg_severity = market_signals['deviation_sigma'].abs().mean()

        heated_markets.append({
            'market': market,
            'trailer_type': first_heat['trailer_type'],
            'first_heat_date': first_heat['forecast_date'],
            'heat_periods': heat_periods,
            'avg_severity': avg_severity,
            'timeframe': first_heat['timeframe']
        })

    return pd.DataFrame(heated_markets).sort_values('first_heat_date')


def create_regional_heat_map(analysis_df, map_df, metric='deviation_sigma'):
    """Create regional heat map for market analysis"""
    # Merge with map data
    regional_data = analysis_df.merge(
        map_df[['Market', 'State', 'Abb']].drop_duplicates(),
        left_on='market', right_on='Market', how='left'
    )

    if regional_data.empty or 'Abb' not in regional_data.columns:
        return None

    # Aggregate by state for the heat map
    state_heat = regional_data.groupby(['State', 'Abb']).agg({
        'deviation_sigma': 'mean',
        'volatility': 'mean',
        'market': 'count'
    }).reset_index()

    state_heat = state_heat.rename(columns={'market': 'market_count'})

    # Create the heat map
    fig = px.choropleth(
        state_heat,
        locations='Abb',
        locationmode='USA-states',
        color='deviation_sigma',
        hover_name='State',
        hover_data={
            'deviation_sigma': ':.2f',
            'volatility': ':.2f',
            'market_count': True
        },
        scope='usa',
        color_continuous_scale='reds',
        title='Regional Market Heat Map (Average Deviation from Equilibrium)'
    )

    return fig


def get_top_markets_by_type(analysis_df, trailer_type, metric='deviation_sigma', top_n=5):
    """Get top markets for a specific trailer type"""
    type_data = analysis_df[analysis_df['trailer_type'] == trailer_type]
    if metric == 'deviation_sigma':
        type_data = type_data.copy()
        type_data['abs_deviation'] = type_data['deviation_sigma'].abs()
        return type_data.nlargest(top_n, 'abs_deviation')
    else:
        return type_data.nlargest(top_n, metric)


def generate_dynamic_insights(market_data, future_signals, trailer_type, timeframe):
    """Generate insights with dynamic equilibrium context"""
    insights = []

    current_ratio = market_data['current_ratio']
    equilibrium = market_data['dynamic_equilibrium']
    volatility = market_data['volatility']
    deviation_sigma = market_data['deviation_sigma']
    current_regime = market_data['current_regime']
    regime_stability = market_data['regime_stability']

    # Current state with dynamic context
    if deviation_sigma > 2.5:
        insights.append(f"🚨 **Severely overheated** - {deviation_sigma:.1f}σ above dynamic equilibrium")
    elif deviation_sigma > 1.5:
        insights.append(f"🔴 **Overheated** - {deviation_sigma:.1f}σ above dynamic equilibrium")
    elif deviation_sigma < -2.5:
        insights.append(f"💤 **Severely underutilized** - {abs(deviation_sigma):.1f}σ below dynamic equilibrium")
    elif deviation_sigma < -1.5:
        insights.append(f"🟢 **Underutilized** - {abs(deviation_sigma):.1f}σ below dynamic equilibrium")
    else:
        insights.append(f"🟡 **Near dynamic equilibrium** - {deviation_sigma:.1f}σ deviation")

    # ENHANCEMENT: Markov Regime Switching insights
    insights.append(f"🏛️ **Markov Regime**: {current_regime.replace('_', ' ').title()}")
    if regime_stability > 0.8:
        insights.append("📊 **High regime stability** - Market likely to maintain current state")
    elif regime_stability < 0.5:
        insights.append("⚡ **Low regime stability** - High probability of regime change")
    else:
        insights.append("🔄 **Moderate regime stability** - Some chance of state transition")

    # Future heat anticipation
    market_future = future_signals[
        (future_signals['market'] == market_data['market']) &
        (future_signals['trailer_type'] == trailer_type)
        ]

    future_heating = market_future[market_future['deviation_sigma'].abs() >= 2.0]
    if not future_heating.empty:
        heat_dates = future_heating['forecast_date'].dt.strftime("%b %Y").tolist()[:3]
        insights.append(f"📅 **Future heating anticipated**: {', '.join(heat_dates)}")

    # Timeframe context
    timeframe_map = {"W": "weekly", "M": "monthly", "Q": "quarterly"}
    insights.append(f"📊 **Analysis timeframe**: {timeframe_map.get(timeframe, timeframe)}")

    return insights


# -------------------------
# Enhanced Main Application with New Features
# -------------------------

st.sidebar.header("Configuration")
lt_file = st.sidebar.file_uploader("L/T Data", type=["xls", "xlsx"])
map_file = st.sidebar.file_uploader("Market Map", type=["xls", "xlsx"])

# ENHANCEMENT 3: Configurable market count
st.sidebar.subheader("📈 Display Settings")
market_count = st.sidebar.slider("Number of Markets to Display", min_value=1, max_value=20, value=5)

# ENHANCEMENT 4: Future Period Analysis
st.sidebar.subheader("🔮 Future Period Analysis")
use_future_period = st.sidebar.checkbox("Analyze Specific Future Period")

if use_future_period:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        future_start = st.date_input("Start Date",
                                     value=datetime.now() + timedelta(days=180),
                                     min_value=datetime.now() + timedelta(days=30))
    with col2:
        future_end = st.date_input("End Date",
                                   value=datetime.now() + timedelta(days=270),
                                   min_value=future_start)

    future_analysis_type = st.sidebar.selectbox("Analysis Type",
                                                ["heated", "volatile", "stable"])

# Existing parameters (keeping everything else the same)
timeframe = st.sidebar.selectbox("Select Timeframe", ["W", "M", "Q"], index=1)
timeframe_labels = {"W": "Weekly", "M": "Monthly", "Q": "Quarterly"}

min_significance = st.sidebar.slider("Significance Threshold", 0.1, 1.0, 0.3)
heat_threshold = st.sidebar.slider("Heat Threshold (σ)", 1.0, 3.0, 2.0, step=0.1)
lookahead_periods = st.sidebar.slider("Lookahead Periods", 1, 12, 6)

if lt_file and map_file:
    lt_raw = read_excel_safely(lt_file)
    map_raw = read_excel_safely(map_file)

    if lt_raw is not None and map_raw is not None:
        lt_df = load_lt(lt_raw)
        map_df = load_map(map_raw)

        if lt_df is not None and map_df is not None:
            with st.spinner(f"Processing {timeframe_labels[timeframe]} data..."):
                lt_full = preprocess_and_reindex(lt_df)
                lt_agg = aggregate_df(lt_full, timeframe)
                lt_agg = lt_agg.rename(columns={
                    "date": "date",
                    "ratio": "ratio",
                    "loads": "loads",
                    "trucks": "trucks",
                    "market": "market",
                    "type_of_trailer": "type_of_trailer"
                })

            # Get unique trailer types
            trailer_types = lt_agg['type_of_trailer'].unique()
            st.sidebar.subheader("Trailer Type Selection")
            selected_type = st.sidebar.selectbox("Analyze Type", trailer_types)

            # Enhanced analysis with dynamic equilibrium
            st.header(f"📊 {selected_type} - {timeframe_labels[timeframe]} Analysis")

            with st.spinner("Computing dynamic equilibriums with Markov regime switching..."):
                analysis_df, future_signals = analyze_market_by_type_dynamic(
                    lt_agg, selected_type, timeframe, min_significance
                )

            if not analysis_df.empty:
                # ENHANCEMENT: Create market selection with state information
                market_state_data = create_market_selection_with_states(analysis_df, map_df)

                if market_state_data is not None:
                    # ENHANCEMENT: Single market selection in filter pane with state information
                    st.sidebar.subheader("📍 Market Selection")

                    # Create selection options with state information
                    market_options = market_state_data['display_name'].unique().tolist()

                    # Add "All Markets" option for overview
                    all_options = ["All Markets (Overview)"] + market_options

                    selected_market_display = st.sidebar.selectbox(
                        "Select Market for Detailed View",
                        all_options,
                        help="Select a specific market for detailed analysis or 'All Markets' for overview"
                    )

                    # Extract market name from display name if a specific market is selected
                    if selected_market_display == "All Markets (Overview)":
                        selected_market = None  # Show overview
                    else:
                        selected_market = selected_market_display.split(" (")[0]  # Extract market name

                # ENHANCEMENT: Use configurable market count
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader(f"🔥 Top {market_count} Heated Markets")
                    heated_markets = get_top_markets_by_type(analysis_df, selected_type, 'deviation_sigma',
                                                             market_count)

                    for _, market in heated_markets.iterrows():
                        # ENHANCEMENT: Add state information to market display
                        market_state_info = ""
                        if market_state_data is not None:
                            state_info = market_state_data[market_state_data['market'] == market['market']]
                            if not state_info.empty and pd.notna(state_info['State'].iloc[0]):
                                market_state_info = f" | {state_info['State'].iloc[0]}"

                        color = "red" if market['deviation_sigma'] > 2 else "orange"
                        st.markdown(
                            f"<div style='border-left: 4px solid {color}; padding-left: 10px; margin: 5px 0;'>"
                            f"<strong>{market['market']}{market_state_info}</strong><br>"
                            f"Ratio: {market['current_ratio']:.1f} | Eq: {market['dynamic_equilibrium']:.1f}<br>"
                            f"Deviation: {market['deviation_sigma']:.1f}σ | Regime: {market['current_regime']}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                with col2:
                    st.subheader(f"⚡ Top {market_count} Volatile Markets")
                    volatile_markets = get_top_markets_by_type(analysis_df, selected_type, 'volatility', market_count)

                    for _, market in volatile_markets.iterrows():
                        # ENHANCEMENT: Add state information to market display
                        market_state_info = ""
                        if market_state_data is not None:
                            state_info = market_state_data[market_state_data['market'] == market['market']]
                            if not state_info.empty and pd.notna(state_info['State'].iloc[0]):
                                market_state_info = f" | {state_info['State'].iloc[0]}"

                        volatility_pct = (market['volatility'] / market['dynamic_equilibrium']) * 100
                        st.markdown(
                            f"<div style='border-left: 4px solid purple; padding-left: 10px; margin: 5px 0;'>"
                            f"<strong>{market['market']}{market_state_info}</strong><br>"
                            f"Volatility: {market['volatility']:.2f} ({volatility_pct:.0f}%)<br>"
                            f"Stability: {market['regime_stability']:.0%}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                with col3:
                    # ENHANCEMENT: Future period analysis or regular future heated
                    if use_future_period:
                        st.subheader(f"🎯 {future_analysis_type.title()} Markets ({future_start} to {future_end})")
                        future_period_results = analyze_future_period(
                            future_signals, future_start, future_end, future_analysis_type, market_count
                        )

                        if not future_period_results.empty:
                            for _, market in future_period_results.iterrows():
                                # ENHANCEMENT: Add state information to market display
                                market_state_info = ""
                                if market_state_data is not None:
                                    state_info = market_state_data[market_state_data['market'] == market['market']]
                                    if not state_info.empty and pd.notna(state_info['State'].iloc[0]):
                                        market_state_info = f" | {state_info['State'].iloc[0]}"

                                st.markdown(
                                    f"<div style='border-left: 4px solid #FF6B6B; padding-left: 10px; margin: 5px 0;'>"
                                    f"<strong>{market['market']}{market_state_info}</strong><br>"
                                    f"Avg Ratio: {market['forecast_ratio']:.1f}<br>"
                                    f"Deviation: {market['deviation_sigma']:.1f}σ | Vol: {market['volatility']:.2f}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info(f"No {future_analysis_type} markets found in selected period")
                    else:
                        st.subheader(f"🎯 Next {market_count} Future Heated Markets")
                        future_heated = identify_future_heated_markets(
                            future_signals, lookahead_periods, heat_threshold
                        )

                        if not future_heated.empty:
                            for _, market in future_heated.head(market_count).iterrows():
                                # ENHANCEMENT: Add state information to market display
                                market_state_info = ""
                                if market_state_data is not None:
                                    state_info = market_state_data[market_state_data['market'] == market['market']]
                                    if not state_info.empty and pd.notna(state_info['State'].iloc[0]):
                                        market_state_info = f" | {state_info['State'].iloc[0]}"

                                st.markdown(
                                    f"<div style='border-left: 4px solid #FF6B6B; padding-left: 10px; margin: 5px 0;'>"
                                    f"<strong>{market['market']}{market_state_info}</strong><br>"
                                    f"First Heat: {market['first_heat_date'].strftime('%b %Y')}<br>"
                                    f"Periods: {market['heat_periods']} | Severity: {market['avg_severity']:.1f}σ"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No significant future heating anticipated")

                # REGIONAL HEAT MAP SECTION
                st.subheader("🗺️ Regional Heat Map")
                heat_map_fig = create_regional_heat_map(analysis_df, map_df)

                if heat_map_fig:
                    st.plotly_chart(heat_map_fig, use_container_width=True)

                    # Heat map interpretation
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Markets Analyzed", len(analysis_df))
                    with col2:
                        avg_deviation = analysis_df['deviation_sigma'].abs().mean()
                        st.metric("Average Absolute Deviation", f"{avg_deviation:.2f}σ")
                    with col3:
                        heated_count = len(analysis_df[analysis_df['deviation_sigma'].abs() > 2.0])
                        st.metric("Markets in Extreme State", heated_count)
                else:
                    st.warning("Could not create heat map. Check if map data contains State/Abb information.")

                # MARKOV REGIME SWITCHING ANALYTICAL TOOL
                st.subheader("🔮 Markov Regime Switching Analysis")

                # ENHANCEMENT: Use the selected market from sidebar or provide selection
                if selected_market is not None:
                    # Use the market selected in the sidebar
                    selected_market_regime = selected_market
                else:
                    # Fallback to dropdown if no specific market selected
                    selected_market_regime = st.selectbox("Select Market for Detailed Regime Analysis",
                                                          analysis_df['market'].unique())

                market_regime_data = analysis_df[analysis_df['market'] == selected_market_regime].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Current Regime Analysis**")
                    st.metric("Current Regime", market_regime_data['current_regime'].replace('_', ' ').title())
                    st.metric("Regime Stability", f"{market_regime_data['regime_stability']:.0%}")

                    if market_regime_data['regime_means']:
                        st.write("**Regime Characteristics**")
                        for regime, mean in market_regime_data['regime_means'].items():
                            st.write(f"• {regime.replace('_', ' ').title()}: {mean:.2f} typical ratio")

                with col2:
                    if market_regime_data['transition_matrix'] is not None:
                        st.write("**Regime Transition Probabilities**")
                        st.dataframe(market_regime_data['transition_matrix'].style.format("{:.1%}"))

                        # Get current regime and stability for this specific market
                        current_regime = market_regime_data['current_regime']
                        stability = market_regime_data['regime_stability']
                        market_name = selected_market_regime

                        st.write(f"**{market_name} - Regime Analysis**:")

                        # Dynamic explanation based on current regime and stability
                        if current_regime == "low_volatility":
                            if stability > 0.7:
                                st.success(f"""
                                **Stable Calm Period**: {market_name} is in a predictable low-volatility state with **{stability:.0%} stability**.

                                - **High probability ({stability:.0%})** of remaining calm
                                - Minimal expected fluctuations
                                - Ideal for consistent operations and planning
                                - Low risk environment for carriers and shippers
                                """)
                            else:
                                st.warning(f"""
                                **Unstable Calm Period**: {market_name} is currently calm but with **only {stability:.0%} stability**.

                                - **High chance of regime change** ({1 - stability:.0%} probability)
                                - Monitor closely for emerging volatility
                                - Prepare for potential market shifts
                                - Consider flexible capacity planning
                                """)

                        elif current_regime == "normal_volatility":
                            if stability > 0.7:
                                st.info(f"""
                                **Stable Normal Conditions**: {market_name} is in typical market behavior with **{stability:.0%} stability**.

                                - **Good predictability** for near-term planning
                                - Moderate, expected fluctuations
                                - Balanced risk-reward environment
                                - Standard operational patterns apply
                                """)
                            else:
                                st.warning(f"""
                                **Transitional Normal Conditions**: {market_name} is in normal volatility but **unstable ({stability:.0%})**.

                                - **Likely to shift** to either high or low volatility
                                - Increased uncertainty in near-term forecasts
                                - Monitor transition probabilities closely
                                - Prepare for potential market extremes
                                """)

                        elif current_regime == "high_volatility":
                            if stability > 0.7:
                                st.error(f"""
                                **Persistent High Volatility**: {market_name} is in extreme conditions with **{stability:.0%} stability**.

                                - **High likelihood** of continued turbulence
                                - Expect rapid, unpredictable changes
                                - Premium pricing and risk management critical
                                - Consider alternative routing options
                                """)
                            else:
                                st.warning(f"""
                                **Unstable High Volatility**: {market_name} is volatile but **likely to change ({stability:.0%} stability)**.

                                - **Good chance** of transitioning to calmer conditions
                                - Current turbulence may be temporary
                                - Monitor for stabilization signals
                                - Opportunity for strategic positioning
                                """)

                        # Add matrix reading guide
                        st.info("""
                        **Reading the Transition Matrix:**
                        - **Rows**: Current regime state
                        - **Columns**: Next period possibilities  
                        - **Bold diagonal**: Probability of staying in current regime
                        - **Off-diagonal**: Probabilities of transitioning to other states
                        """)

                # Detailed Market Analysis (existing functionality)
                st.subheader("🔍 Detailed Market Analysis")

                # ENHANCEMENT: Use the selected market from sidebar or provide selection
                if selected_market is not None:
                    # Use the market selected in the sidebar
                    selected_market_detail = selected_market
                else:
                    # Fallback to dropdown if no specific market selected
                    selected_market_detail = st.selectbox("Select Market for Detailed View",
                                                          analysis_df['market'].unique())

                market_details = analysis_df[analysis_df['market'] == selected_market_detail].iloc[0]
                market_data = lt_agg[
                    (lt_agg['market'] == selected_market_detail) &
                    (lt_agg['type_of_trailer'] == selected_type)
                    ].set_index('date').sort_index()

                # Dynamic equilibrium visualization
                ts_data = market_data['ratio'].dropna()
                dynamic_eq_series, current_eq, regime_shifts = compute_dynamic_equilibrium(ts_data, window=12,
                                                                                           method='adaptive')
                # Use Markov regime analysis
                regime_analysis = markov_regime_switching_analysis(ts_data)
                regimes = regime_analysis['regimes']
                forecast = fit_seasonal_forecast(ts_data, steps=lookahead_periods)

                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=ts_data.index, y=ts_data.values,
                    mode='lines+markers', name='Historical Ratio',
                    line=dict(color='blue', width=2)
                ))

                # Dynamic equilibrium
                fig.add_trace(go.Scatter(
                    x=dynamic_eq_series.index, y=dynamic_eq_series.values,
                    mode='lines', name='Dynamic Equilibrium',
                    line=dict(color='green', dash='dash', width=2)
                ))

                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast.index, y=forecast.values,
                    mode='lines+markers', name='Seasonal Forecast',
                    line=dict(color='orange', width=2)
                ))

                # Current equilibrium level
                fig.add_hline(
                    y=market_details['dynamic_equilibrium'],
                    line=dict(color='red', dash='dot'),
                    annotation_text=f"Current Equilibrium: {market_details['dynamic_equilibrium']:.2f}"
                )

                fig.update_layout(
                    height=500,
                    title=f"{selected_market_detail} - Dynamic Equilibrium & Markov Regime Analysis",
                    xaxis_title="Date",
                    yaxis_title="Load-to-Truck Ratio"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Market metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Ratio", f"{market_details['current_ratio']:.2f}")
                with col2:
                    st.metric("Dynamic Equilibrium", f"{market_details['dynamic_equilibrium']:.2f}")
                with col3:
                    st.metric("Deviation", f"{market_details['deviation_sigma']:.1f}σ")
                with col4:
                    st.metric("Regime Stability", f"{market_details['regime_stability']:.0%}")

                # Enhanced Insights with Markov explanations
                st.subheader("💡 Dynamic Market Intelligence")
                insights = generate_dynamic_insights(market_details, future_signals, selected_type, timeframe)
                for insight in insights:
                    st.write(insight)

            else:
                st.warning(f"No significant markets found for {selected_type}")

else:
    st.info("Please upload both L/T data and Market Map files to begin analysis")

st.markdown("---")
st.write("""
**Enhanced Methodology:**
- **Markov Regime Switching**: Statistical detection of market states with transition probabilities
- **Configurable Rankings**: Display 1-20 top markets based on your preference
- **Future Period Analysis**: Analyze specific date ranges (e.g., Nov 1 2025 - Dec 12 2025)
- **Dynamic Equilibrium**: Adaptive equilibrium with regime shift detection
- **Regional Heat Maps**: Visualize market conditions across states
- **Multiple Timeframes**: Weekly, Monthly, Quarterly analysis
- **Statistical Significance**: Volume-based filtering for reliable signals
- **ENHANCED MARKET SELECTION**: Single selection in filter pane with state information
""")

# Download functionality
if 'analysis_df' in locals() and not analysis_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Export")

    csv_data = analysis_df.to_csv(index=False)
    st.sidebar.download_button(
        "Download Market Analysis",
        csv_data,
        file_name=f"{selected_type}_{timeframe}_market_analysis.csv",
        mime="text/csv"
    )