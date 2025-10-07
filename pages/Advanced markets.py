# market1.py - Advanced Market Analysis Laboratory
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(layout="wide", page_title="Advanced Market Analysis Lab")
st.title("🔬 Advanced Market Analysis Laboratory")
st.markdown("**Structural Stability + Bayesian Analysis + Fibonacci + ML Forecasting**")


# Data Loading Functions
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


def create_market_selection_with_states(analysis_df, map_df):
    if analysis_df.empty or map_df.empty:
        return None
    market_state_data = analysis_df.merge(
        map_df[['Market', 'State', 'Abb']].drop_duplicates(),
        left_on='market', right_on='Market', how='left'
    )
    market_state_data['display_name'] = market_state_data.apply(
        lambda x: f"{x['market']} ({x['State']})" if pd.notna(x['State']) else x['market'],
        axis=1
    )
    market_state_data = market_state_data.sort_values(['State', 'market'])
    return market_state_data


# Advanced Analytical Functions
def chow_test_statistical(ts_data, break_point=None, min_obs=20):
    if len(ts_data) < 2 * min_obs:
        return {'structural_break': False, 'confidence': 0.0, 'chow_statistic': None, 'p_value': None}
    if break_point is None:
        break_point = len(ts_data) // 2
    if break_point < min_obs or (len(ts_data) - break_point) < min_obs:
        return {'structural_break': False, 'confidence': 0.0, 'chow_statistic': None, 'p_value': None}
    try:
        from scipy import stats
        y = ts_data.values
        X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
        y1, y2 = y[:break_point], y[break_point:]
        X1, X2 = X[:break_point], X[break_point:]
        beta1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
        beta2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
        beta_pooled, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid1 = y1 - X1 @ beta1
        resid2 = y2 - X2 @ beta2
        resid_pooled = y - X @ beta_pooled
        k = X.shape[1]
        n1, n2 = len(y1), len(y2)
        RSS_pooled = np.sum(resid_pooled ** 2)
        RSS_individual = np.sum(resid1 ** 2) + np.sum(resid2 ** 2)
        numerator = (RSS_pooled - RSS_individual) / k
        denominator = RSS_individual / (n1 + n2 - 2 * k) if (n1 + n2 - 2 * k) > 0 else 1
        chow_stat = numerator / denominator if denominator > 0 else 0
        p_value = 1 - stats.f.cdf(chow_stat, k, n1 + n2 - 2 * k) if denominator > 0 else 1.0
        structural_break = p_value < 0.05
        return {
            'chow_statistic': chow_stat,
            'p_value': p_value,
            'structural_break': structural_break,
            'break_point': ts_data.index[break_point],
            'confidence': 1 - p_value if p_value is not None else 0.0
        }
    except Exception:
        return {'structural_break': False, 'confidence': 0.0, 'chow_statistic': None, 'p_value': None}


def detect_structural_breaks_rolling(ts_data, window_size=24):
    if len(ts_data) < 2 * window_size:
        return []
    breaks = []
    for i in range(window_size, len(ts_data) - window_size, 3):
        chow_result = chow_test_statistical(ts_data, break_point=i)
        if chow_result['structural_break'] and chow_result['p_value'] < 0.05:
            breaks.append({
                'break_date': ts_data.index[i],
                'confidence': chow_result['confidence'],
                'chow_statistic': chow_result['chow_statistic'],
                'p_value': chow_result['p_value']
            })
    return breaks


def fibonacci_retracement_levels(ts_data):
    if len(ts_data) < 20:
        return None
    high = ts_data.max()
    low = ts_data.min()
    current = ts_data.iloc[-1]
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618]
    retracement_levels = {}
    extension_levels = {}
    for ratio in ratios:
        level = high - (high - low) * ratio
        retracement_levels[f'FIB_{int(ratio * 1000)}'] = level
    for ratio in [1.272, 1.414, 1.618, 2.0, 2.618]:
        level = high + (high - low) * (ratio - 1.0)
        extension_levels[f'EXT_{int(ratio * 1000)}'] = level
    fib_position = "Above 61.8%" if current > retracement_levels['FIB_618'] else \
        "Between 38.2%-61.8%" if current > retracement_levels['FIB_382'] else \
            "Between 23.6%-38.2%" if current > retracement_levels['FIB_236'] else \
                "Below 23.6%"
    return {
        'high': high,
        'low': low,
        'current': current,
        'retracement_levels': retracement_levels,
        'extension_levels': extension_levels,
        'current_position': fib_position,
        'range': high - low
    }


def bayesian_structural_timeseries(ts_data, horizon=12):
    try:
        if len(ts_data) < 24:
            return None
        y = ts_data.values
        X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
        mu_prior = np.array([y.mean(), 0])
        Sigma_prior = np.diag([100, 10])
        Sigma_posterior = np.linalg.inv(np.linalg.inv(Sigma_prior) + X.T @ X)
        mu_posterior = Sigma_posterior @ (np.linalg.inv(Sigma_prior) @ mu_prior + X.T @ y)
        future_dates = pd.date_range(
            start=ts_data.index[-1] + pd.Timedelta(days=30),
            periods=horizon, freq='M'
        )
        X_future = np.column_stack([np.ones(horizon), np.arange(len(ts_data), len(ts_data) + horizon)])
        forecasts = X_future @ mu_posterior
        forecast_variance = 1 + np.diag(X_future @ Sigma_posterior @ X_future.T)
        lower_bound = forecasts - 1.96 * np.sqrt(forecast_variance)
        upper_bound = forecasts + 1.96 * np.sqrt(forecast_variance)
        return {
            'forecast': pd.Series(forecasts, index=future_dates),
            'lower_bound': pd.Series(lower_bound, index=future_dates),
            'upper_bound': pd.Series(upper_bound, index=future_dates),
            'trend_slope': mu_posterior[1],
            'trend_significant': abs(mu_posterior[1]) > 2 * np.sqrt(Sigma_posterior[1, 1])
        }
    except Exception as e:
        st.warning(f"Bayesian analysis simplified: {e}")
        return None


def create_features_for_ml(dates):
    return pd.DataFrame({
        'dayofyear': dates.dayofyear,
        'weekofyear': dates.isocalendar().week,
        'month': dates.month,
        'quarter': dates.quarter,
        'year': dates.year,
        'dayofweek': dates.dayofweek,
    }, index=dates)


def xgboost_forecast(ts_data, horizon=12):
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        features = create_features_for_ml(ts_data.index)
        target = ts_data.values
        for lag in [1, 2, 3, 6, 12]:
            if len(ts_data) > lag:
                features[f'lag_{lag}'] = ts_data.shift(lag).values
        features = features.dropna()
        target = target[len(target) - len(features):]
        if len(features) < 24:
            return None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, subsample=0.8, colsample_bytree=0.8
        )
        model.fit(X_scaled, target)
        future_dates = pd.date_range(
            start=ts_data.index[-1] + pd.Timedelta(days=30),
            periods=horizon, freq='M'
        )
        predictions = []
        current_features = features.iloc[-1:].copy()
        for i in range(horizon):
            future_features = create_features_for_ml([future_dates[i]]).iloc[0]
            for j, lag in enumerate([1, 2, 3, 6, 12]):
                if j < len(predictions):
                    future_features[f'lag_{lag}'] = predictions[-(j + 1)] if (j + 1) <= len(predictions) else \
                    current_features[f'lag_{lag}'].iloc[0]
                else:
                    future_features[f'lag_{lag}'] = current_features[f'lag_{lag}'].iloc[
                        0] if f'lag_{lag}' in current_features else ts_data.iloc[-1]
            future_scaled = scaler.transform([future_features])
            pred = model.predict(future_scaled)[0]
            predictions.append(pred)
        return pd.Series(predictions, index=future_dates)
    except ImportError:
        st.warning("XGBoost not installed. Using fallback method.")
        return fallback_forecast(ts_data, horizon)
    except Exception as e:
        st.warning(f"XGBoost failed: {e}")
        return fallback_forecast(ts_data, horizon)


def lstm_forecast(ts_data, horizon=12, lookback=12):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])
        if len(X) < 10:
            return None
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, batch_size=32, epochs=50, verbose=0, validation_split=0.2)
        last_sequence = scaled_data[-lookback:]
        predictions = []
        for _ in range(horizon):
            next_pred = model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
            predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(
            start=ts_data.index[-1] + pd.Timedelta(days=30),
            periods=horizon, freq='M'
        )
        return pd.Series(predictions.flatten(), index=future_dates)
    except ImportError:
        st.warning("TensorFlow not installed. Using fallback method.")
        return fallback_forecast(ts_data, horizon)
    except Exception as e:
        st.warning(f"LSTM failed: {e}")
        return fallback_forecast(ts_data, horizon)


def random_forest_forecast(ts_data, horizon=12):
    try:
        from sklearn.ensemble import RandomForestRegressor
        features = create_features_for_ml(ts_data.index)
        target = ts_data.values
        for lag in [1, 2, 3, 6, 12]:
            if len(ts_data) > lag:
                features[f'lag_{lag}'] = ts_data.shift(lag).values
        features = features.dropna()
        target = target[len(target) - len(features):]
        if len(features) < 24:
            return None
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42,
            min_samples_split=5, min_samples_leaf=2
        )
        model.fit(features, target)
        future_dates = pd.date_range(
            start=ts_data.index[-1] + pd.Timedelta(days=30),
            periods=horizon, freq='M'
        )
        predictions = []
        current_features = features.iloc[-1:].copy()
        for i in range(horizon):
            future_features = create_features_for_ml([future_dates[i]]).iloc[0]
            for j, lag in enumerate([1, 2, 3, 6, 12]):
                if j < len(predictions):
                    future_features[f'lag_{lag}'] = predictions[-(j + 1)] if (j + 1) <= len(predictions) else \
                    current_features[f'lag_{lag}'].iloc[0]
                else:
                    future_features[f'lag_{lag}'] = current_features[f'lag_{lag}'].iloc[
                        0] if f'lag_{lag}' in current_features else ts_data.iloc[-1]
            pred = model.predict([future_features])[0]
            predictions.append(pred)
        return pd.Series(predictions, index=future_dates)
    except ImportError:
        st.warning("scikit-learn not installed. Using fallback method.")
        return fallback_forecast(ts_data, horizon)
    except Exception as e:
        st.warning(f"Random Forest failed: {e}")
        return fallback_forecast(ts_data, horizon)


def fallback_forecast(ts_data, horizon=12):
    try:
        if len(ts_data) >= 12:
            seasonal_pattern = ts_data.tail(12).values
        else:
            seasonal_pattern = ts_data.values
        if len(ts_data) >= 6:
            x = np.arange(len(ts_data))
            trend_coef = np.polyfit(x, ts_data.values, 1)[0]
        else:
            trend_coef = 0
        forecast_values = []
        for i in range(horizon):
            seasonal_component = seasonal_pattern[i % len(seasonal_pattern)]
            trend_component = trend_coef * (i + 1)
            forecast_values.append(seasonal_component + trend_component * 0.5)
        future_dates = pd.date_range(
            start=ts_data.index[-1] + pd.Timedelta(days=30),
            periods=horizon, freq='M'
        )
        return pd.Series(forecast_values, index=future_dates)
    except Exception:
        return None


def ensemble_forecast(ts_data, horizon=12):
    models = {
        'XGBoost': lambda x: xgboost_forecast(x, horizon),
        'LSTM': lambda x: lstm_forecast(x, horizon),
        'Random Forest': lambda x: random_forest_forecast(x, horizon),
        'Bayesian': lambda x: bayesian_structural_timeseries(x, horizon)['forecast'] if bayesian_structural_timeseries(
            x, horizon) else None,
    }
    split_point = int(len(ts_data) * 0.8)
    if split_point < 24:
        return fallback_forecast(ts_data, horizon), 'Fallback (insufficient data)'
    train_data = ts_data[:split_point]
    test_data = ts_data[split_point:]
    best_model = None
    best_mae = float('inf')
    forecasts = {}
    for name, model_func in models.items():
        try:
            forecast = model_func(train_data)
            if forecast is not None and len(forecast) > 0:
                common_dates = forecast.index.intersection(test_data.index)
                if len(common_dates) > 0:
                    mae = np.mean(np.abs(forecast[common_dates] - test_data[common_dates]))
                    forecasts[name] = forecast
                    if mae < best_mae:
                        best_mae = mae
                        best_model = name
        except Exception:
            continue
    if best_model and best_model in forecasts:
        return forecasts[best_model], best_model
    else:
        fallback = fallback_forecast(ts_data, horizon)
        return fallback, 'Fallback (no ML models available)' if fallback is not None else (None, 'No models converged')


def calculate_statistical_tests(ts_data):
    from scipy import stats
    tests = {}
    if len(ts_data) >= 8:
        shapiro_stat, shapiro_p = stats.shapiro(ts_data)
        tests['Normality (Shapiro-Wilk)'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(ts_data.dropna())
    tests['Stationarity (ADF)'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'is_stationary': adf_result[1] < 0.05
    }
    from statsmodels.tsa.stattools import acf
    autocorr = acf(ts_data, nlags=min(20, len(ts_data) // 4))
    tests['Autocorrelation'] = {
        'lags': autocorr,
        'significant_lags': np.where(np.abs(autocorr) > 1.96 / np.sqrt(len(ts_data)))[0]
    }
    return tests


def generate_market_interpretation(market_name, ts_data, structural_breaks, fib_results, bayesian_results,
                                   statistical_tests, forecast_result, best_model):
    """Generate comprehensive market interpretation"""

    current_ratio = ts_data.iloc[-1]
    historical_mean = ts_data.mean()
    position_relative_to_mean = "significantly above" if current_ratio > historical_mean * 1.2 else "above" if current_ratio > historical_mean else "below" if current_ratio < historical_mean * 0.8 else "near"

    # Structural stability interpretation
    if structural_breaks:
        stability_interpretation = f"🚨 **Market Instability Detected**: {len(structural_breaks)} structural breaks identified. The market has experienced fundamental changes in behavior, indicating shifting supply/demand dynamics."
    else:
        stability_interpretation = "✅ **Market Stability**: No significant structural breaks detected. The market shows consistent behavior patterns, suggesting stable supply/demand fundamentals."

    # Fibonacci interpretation
    if fib_results:
        if fib_results['current_position'] == "Above 61.8%":
            fib_interpretation = f"📈 **Overbought Territory**: Current ratio ({current_ratio:.2f}) is in the top 38.2% of its historical range. This suggests potential for a pullback toward the 61.8% Fibonacci level at {fib_results['retracement_levels']['FIB_618']:.2f}."
        elif fib_results['current_position'] == "Below 23.6%":
            fib_interpretation = f"📉 **Oversold Opportunity**: Current ratio ({current_ratio:.2f}) is in the bottom 23.6% of its range. This may present a buying opportunity with potential resistance at the 38.2% level ({fib_results['retracement_levels']['FIB_382']:.2f})."
        else:
            fib_interpretation = f"⚖️ **Neutral Zone**: Current position suggests balanced market conditions between key Fibonacci levels."
    else:
        fib_interpretation = "Fibonacci analysis not available due to insufficient data."

    # Statistical interpretation
    stat_interpretation = ""
    if statistical_tests:
        normality = statistical_tests['Normality (Shapiro-Wilk)'][
            'is_normal'] if 'Normality (Shapiro-Wilk)' in statistical_tests else False
        stationarity = statistical_tests['Stationarity (ADF)'][
            'is_stationary'] if 'Stationarity (ADF)' in statistical_tests else False

        if normality:
            stat_interpretation += "✅ Data follows normal distribution patterns. "
        else:
            stat_interpretation += "⚠️ Data shows non-normal distribution. "

        if stationarity:
            stat_interpretation += "✅ Time series is stationary. "
        else:
            stat_interpretation += "⚠️ Time series has trends/seasonality. "

    # Forecast interpretation with timing
    if forecast_result is not None:
        forecast_avg = forecast_result.mean()
        change_pct = ((forecast_avg - current_ratio) / current_ratio) * 100

        # Calculate when the shift becomes viable (crosses historical mean)
        viable_shift_period = None
        for i, (date, value) in enumerate(forecast_result.items()):
            if (change_pct > 0 and value > historical_mean) or (change_pct < 0 and value < historical_mean):
                viable_shift_period = i + 1  # Month number
                break

        if abs(change_pct) > 10:
            direction = "increase" if change_pct > 0 else "decrease"
            forecast_interpretation = f"🎯 **Strong {direction.upper()} Expected**: Forecast shows {abs(change_pct):.1f}% {direction} to {forecast_avg:.2f}"
            if viable_shift_period:
                forecast_interpretation += f". Shift becomes viable in **{viable_shift_period} month{'s' if viable_shift_period > 1 else ''}**"
        else:
            forecast_interpretation = f"⚖️ **Stable Outlook**: Market expected to remain relatively stable (±{abs(change_pct):.1f}%)"
    else:
        forecast_interpretation = "Forecast not available."

    # Recommendation
    if forecast_result is not None:
        change_pct = ((forecast_result.mean() - current_ratio) / current_ratio) * 100
        if change_pct > 15:
            recommendation = "🟢 **STRONG BUY**: Significant upside potential expected"
        elif change_pct > 5:
            recommendation = "🟡 **MODERATE BUY**: Moderate growth expected"
        elif change_pct < -15:
            recommendation = "🔴 **STRONG SELL**: Significant downside risk"
        elif change_pct < -5:
            recommendation = "🟠 **MODERATE SELL**: Moderate decline expected"
        else:
            recommendation = "⚪ **HOLD**: Market expected to remain range-bound"
    else:
        recommendation = "📊 **ANALYZE FURTHER**: Insufficient data for clear recommendation"

    interpretation = f"""
    ## 📋 {market_name} - Executive Summary

    ### 📊 Current Market State
    The {market_name} market is currently at **{current_ratio:.2f}**, which is **{position_relative_to_mean}** its historical average of {historical_mean:.2f}. 
    Volatility of {ts_data.std():.2f} indicates {'high' if ts_data.std() > historical_mean * 0.3 else 'moderate' if ts_data.std() > historical_mean * 0.15 else 'low'} price swings.

    ### 🏗️ Structural Outlook
    {stability_interpretation}

    ### 📐 Technical Positioning  
    {fib_interpretation}

    ### 📈 Statistical Foundation
    {stat_interpretation}

    ### 🤖 Forecast Outlook ({best_model})
    {forecast_interpretation}

    ### 💡 Recommended Action
    {recommendation}

    ### ⏰ Expected Timing
    - **Immediate** (1-2 months): Monitor for confirmation of trend
    - **Short-term** (3-6 months): Expected period for significant moves
    - **Medium-term** (7-12 months): Full forecast realization period
    """

    return interpretation


# Main Application
def main():
    st.sidebar.header("Configuration")
    lt_file = st.sidebar.file_uploader("L/T Data", type=["xls", "xlsx"])
    map_file = st.sidebar.file_uploader("Market Map", type=["xls", "xlsx"])

    st.sidebar.subheader("📈 Analysis Settings")
    timeframe = st.sidebar.selectbox("Select Timeframe", ["W", "M", "Q"], index=1)
    timeframe_labels = {"W": "Weekly", "M": "Monthly", "Q": "Quarterly"}
    min_significance = st.sidebar.slider("Significance Threshold", 0.1, 1.0, 0.3)
    market_count = st.sidebar.slider("Number of Markets to Display", min_value=1, max_value=20, value=5)

    st.sidebar.subheader("🔮 Advanced Settings")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)
    detect_breaks = st.sidebar.checkbox("Detect Structural Breaks", value=True)
    run_statistical_tests = st.sidebar.checkbox("Run Statistical Tests", value=True)
    include_fibonacci = st.sidebar.checkbox("Include Fibonacci Analysis", value=True)
    include_bayesian = st.sidebar.checkbox("Include Bayesian Analysis", value=True)

    # Explanation Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📖 Understanding Your Analysis")

    with st.sidebar.expander("🔍 What Do These Results Mean?"):
        st.markdown("""
        ### 🏗️ Structural Stability Analysis
        **Chow Test** checks if your market has experienced significant structural breaks - sudden changes in behavior that could indicate new market conditions. 
        - ✅ **Stable**: Market behavior is consistent over time
        - 🚨 **Breaks Detected**: Market fundamentals have changed

        ### 📐 Fibonacci Technical Analysis  
        **Fibonacci levels** identify potential support/resistance zones based on mathematical ratios:
        - **23.6%**: Minor support/resistance
        - **38.2%**: Significant level  
        - **61.8%**: "Golden ratio" - strongest level
        - **Current Position** tells you if the market is overbought or oversold

        ### 🎯 Bayesian Time Series
        **Bayesian analysis** provides probabilistic forecasts with uncertainty ranges:
        - **Trend Slope**: Direction and strength of market trend
        - **Significant Trend**: High confidence the trend is real (not random)
        - **Credible Intervals**: 95% probability range for future values

        ### 📊 Statistical Tests
        - **Normality**: Whether data follows a bell curve (ideal for many models)
        - **Stationarity**: Whether statistical properties are constant over time
        - **Autocorrelation**: Whether current values depend on past values

        ### 🤖 Machine Learning Forecasting
        **Ensemble approach** automatically selects the best model:
        - **XGBoost**: Excellent for complex relationships
        - **LSTM**: Neural network for time series patterns
        - **Random Forest**: Robust against outliers
        - **Bayesian**: Provides uncertainty estimates

        ### ⏰ Expected Timing
        - **Viable Shift**: When forecast crosses historical average
        - **Immediate**: 1-2 months for trend confirmation
        - **Short-term**: 3-6 months for significant moves
        - **Medium-term**: 7-12 months for full realization
        """)

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
                        "date": "date", "ratio": "ratio", "loads": "loads",
                        "trucks": "trucks", "market": "market", "type_of_trailer": "type_of_trailer"
                    })
                trailer_types = lt_agg['type_of_trailer'].unique()
                st.sidebar.subheader("Trailer Type Selection")
                selected_type = st.sidebar.selectbox("Analyze Type", trailer_types)
                type_data = lt_agg[lt_agg['type_of_trailer'] == selected_type]
                markets = type_data['market'].unique()

                if len(markets) > 0:
                    market_state_data = create_market_selection_with_states(pd.DataFrame({'market': markets}), map_df)

                    st.sidebar.subheader("📍 Market Selection")
                    if market_state_data is not None:
                        selected_markets = st.sidebar.multiselect(
                            "Select Markets to Analyze",
                            options=market_state_data['display_name'].unique(),
                            default=list(market_state_data['display_name'].unique()[:market_count])
                        )
                        # Convert display names back to market names
                        market_display_map = dict(zip(market_state_data['display_name'], market_state_data['market']))
                        selected_market_names = [market_display_map[m] for m in selected_markets]
                    else:
                        selected_market_names = st.sidebar.multiselect(
                            "Select Markets to Analyze",
                            options=markets,
                            default=list(markets[:market_count])
                        )

                    # Main Analysis Section
                    st.header("📊 Advanced Market Analysis Dashboard")

                    if selected_market_names:
                        # Create tabs for different analysis views
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "📈 Market Overview",
                            "🔍 Deep Analysis",
                            "📋 Executive Summary",
                            "📊 Statistical Details"
                        ])

                        with tab1:
                            st.subheader("Market Performance Overview")

                            # Create comparison charts
                            col1, col2 = st.columns(2)

                            with col1:
                                # Current ratios comparison
                                current_ratios = []
                                for market in selected_market_names:
                                    market_data = type_data[type_data['market'] == market]
                                    if not market_data.empty and 'ratio' in market_data.columns:
                                        current_ratio = market_data['ratio'].iloc[-1] if len(
                                            market_data) > 0 else np.nan
                                        current_ratios.append({'Market': market, 'Current Ratio': current_ratio})

                                if current_ratios:
                                    current_df = pd.DataFrame(current_ratios)
                                    fig_current = px.bar(current_df, x='Market', y='Current Ratio',
                                                         title="Current Market Ratios")
                                    st.plotly_chart(fig_current, use_container_width=True)

                            with col2:
                                # Volatility comparison
                                volatility_data = []
                                for market in selected_market_names:
                                    market_data = type_data[type_data['market'] == market]
                                    if not market_data.empty and 'ratio' in market_data.columns:
                                        volatility = market_data['ratio'].std()
                                        volatility_data.append({'Market': market, 'Volatility': volatility})

                                if volatility_data:
                                    vol_df = pd.DataFrame(volatility_data)
                                    fig_vol = px.bar(vol_df, x='Market', y='Volatility',
                                                     title="Market Volatility (Standard Deviation)")
                                    st.plotly_chart(fig_vol, use_container_width=True)

                        with tab2:
                            st.subheader("Deep Market Analysis")

                            # Let user select a specific market for deep analysis
                            deep_market = st.selectbox("Select Market for Detailed Analysis", selected_market_names)

                            if deep_market:
                                market_data = type_data[type_data['market'] == deep_market].set_index('date')
                                if 'ratio' in market_data.columns:
                                    ratio_series = market_data['ratio'].dropna()

                                    if len(ratio_series) > 0:
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            # Time series plot
                                            fig_ts = px.line(x=ratio_series.index, y=ratio_series.values,
                                                             title=f"{deep_market} - Ratio Over Time")
                                            fig_ts.update_layout(xaxis_title="Date", yaxis_title="Ratio")
                                            st.plotly_chart(fig_ts, use_container_width=True)

                                        with col2:
                                            # Distribution plot
                                            fig_dist = px.histogram(x=ratio_series.values,
                                                                    title=f"{deep_market} - Ratio Distribution")
                                            fig_dist.update_layout(xaxis_title="Ratio", yaxis_title="Frequency")
                                            st.plotly_chart(fig_dist, use_container_width=True)

                                        # Advanced Analysis Section
                                        st.subheader("🔬 Advanced Analytical Results")

                                        analysis_col1, analysis_col2 = st.columns(2)

                                        with analysis_col1:
                                            # Structural Breaks Analysis
                                            if detect_breaks and len(ratio_series) >= 24:
                                                with st.spinner("Detecting structural breaks..."):
                                                    structural_breaks = detect_structural_breaks_rolling(ratio_series)

                                                st.metric("Structural Breaks Detected", len(structural_breaks))
                                                if structural_breaks:
                                                    breaks_df = pd.DataFrame(structural_breaks)
                                                    st.dataframe(
                                                        breaks_df[['break_date', 'confidence', 'p_value']].round(4))

                                            # Fibonacci Analysis
                                            if include_fibonacci and len(ratio_series) >= 20:
                                                fib_results = fibonacci_retracement_levels(ratio_series)
                                                if fib_results:
                                                    st.write("**Fibonacci Levels:**")
                                                    st.write(f"Current Position: {fib_results['current_position']}")
                                                    st.write(
                                                        f"Range: {fib_results['low']:.2f} - {fib_results['high']:.2f}")

                                        with analysis_col2:
                                            # Statistical Tests
                                            if run_statistical_tests:
                                                with st.spinner("Running statistical tests..."):
                                                    statistical_tests = calculate_statistical_tests(ratio_series)

                                                if statistical_tests:
                                                    if 'Normality (Shapiro-Wilk)' in statistical_tests:
                                                        norm_test = statistical_tests['Normality (Shapiro-Wilk)']
                                                        st.metric("Normal Distribution",
                                                                  "Yes" if norm_test['is_normal'] else "No",
                                                                  delta=f"p-value: {norm_test['p_value']:.4f}")

                                                    if 'Stationarity (ADF)' in statistical_tests:
                                                        stat_test = statistical_tests['Stationarity (ADF)']
                                                        st.metric("Stationary Series",
                                                                  "Yes" if stat_test['is_stationary'] else "No",
                                                                  delta=f"p-value: {stat_test['p_value']:.4f}")

                                        # Forecasting Section
                                        st.subheader("📈 Market Forecasting")

                                        if len(ratio_series) >= 24:
                                            with st.spinner("Generating forecasts..."):
                                                forecast_result, best_model = ensemble_forecast(ratio_series,
                                                                                                forecast_horizon)

                                            if forecast_result is not None:
                                                st.write(f"**Best Model: {best_model}**")

                                                # Create forecast plot
                                                fig_forecast = go.Figure()
                                                fig_forecast.add_trace(go.Scatter(
                                                    x=ratio_series.index, y=ratio_series.values,
                                                    name='Historical', line=dict(color='blue')
                                                ))
                                                fig_forecast.add_trace(go.Scatter(
                                                    x=forecast_result.index, y=forecast_result.values,
                                                    name='Forecast', line=dict(color='red', dash='dash')
                                                ))
                                                fig_forecast.update_layout(title=f"{deep_market} - Ratio Forecast")
                                                st.plotly_chart(fig_forecast, use_container_width=True)

                                                # Forecast statistics
                                                current_val = ratio_series.iloc[-1]
                                                forecast_avg = forecast_result.mean()
                                                change_pct = ((forecast_avg - current_val) / current_val) * 100

                                                col1, col2, col3 = st.columns(3)
                                                col1.metric("Current Ratio", f"{current_val:.2f}")
                                                col2.metric("Forecast Average", f"{forecast_avg:.2f}")
                                                col3.metric("Expected Change", f"{change_pct:+.1f}%")

                        with tab3:
                            st.subheader("Executive Summary")

                            for market in selected_market_names[:5]:  # Limit to first 5 for performance
                                market_data = type_data[type_data['market'] == market].set_index('date')
                                if 'ratio' in market_data.columns:
                                    ratio_series = market_data['ratio'].dropna()

                                    if len(ratio_series) >= 12:
                                        # Run analyses
                                        structural_breaks = detect_structural_breaks_rolling(
                                            ratio_series) if detect_breaks else []
                                        fib_results = fibonacci_retracement_levels(
                                            ratio_series) if include_fibonacci else None
                                        statistical_tests = calculate_statistical_tests(
                                            ratio_series) if run_statistical_tests else {}
                                        forecast_result, best_model = ensemble_forecast(ratio_series,
                                                                                        6)  # Shorter horizon for summary

                                        # Generate interpretation
                                        interpretation = generate_market_interpretation(
                                            market, ratio_series, structural_breaks, fib_results,
                                            None, statistical_tests, forecast_result, best_model
                                        )

                                        with st.expander(f"📋 {market} - Analysis Summary"):
                                            st.markdown(interpretation)

                        with tab4:
                            st.subheader("Statistical Details")

                            selected_stat_market = st.selectbox("Select Market for Statistical Details",
                                                                selected_market_names)

                            if selected_stat_market:
                                market_data = type_data[type_data['market'] == selected_stat_market].set_index('date')
                                if 'ratio' in market_data.columns:
                                    ratio_series = market_data['ratio'].dropna()

                                    if run_statistical_tests:
                                        statistical_tests = calculate_statistical_tests(ratio_series)

                                        for test_name, test_results in statistical_tests.items():
                                            st.write(f"**{test_name}**")
                                            st.json(test_results)

                    else:
                        st.warning("Please select at least one market to analyze.")
                else:
                    st.warning("No market data available for the selected trailer type.")
            else:
                st.error("Error processing data files.")
        else:
            st.error("Error loading data files. Please check the file formats.")
    else:
        st.info("👈 Please upload both L/T Data and Market Map files to begin analysis.")

    # Add footer
    st.markdown("---")
    st.markdown("**Advanced Market Analysis Laboratory** • Built with Streamlit")


if __name__ == "__main__":
    main()