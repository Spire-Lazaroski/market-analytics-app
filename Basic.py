# basic.py (Main Dashboard) - FIXED WITH NATIVE STREAMLIT COMPONENTS
"""
Main Dashboard for Market Analysis
- Professional dark theme interface
- US heat map with L/T ratios
- Multi-level filtering including date range
- Consistent aggregation for all metrics
- Professional business insights
- Loads vs Trucks visualization
- Navigation to advanced pages
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration with dark theme
st.set_page_config(
    layout="wide",
    page_title="Market Intelligence Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark professional styling with ORANGE TEXT for metrics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00ffff;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #00ffff;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #00ffff;
        padding-bottom: 0.5rem;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.2);
    }
    .market-status-high {
        background: linear-gradient(135deg, #ff4444, #ff6666);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ff4444;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
    }
    .market-status-low {
        background: linear-gradient(135deg, #4444ff, #6666ff);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4444ff;
        color: white;
        box-shadow: 0 4px 15px rgba(68, 68, 255, 0.3);
    }
    .market-status-balanced {
        background: linear-gradient(135deg, #44ff44, #66ff66);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #44ff44;
        color: white;
        box-shadow: 0 4px 15px rgba(68, 255, 68, 0.3);
    }
    .insight-section {
        background: linear-gradient(135deg, #2a2a2a, #3a3a3a);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #555555;
        color: #ffffff;
    }
    .filter-section {
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #444444;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00ffff, #0099ff);
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 255, 0.4);
    }

    /* ORANGE TEXT for metrics - numbers and labels */
    [data-testid="stMetricValue"] {
        color: #ffa500 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #ffa500 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #ffa500 !important;
        font-weight: 600 !important;
    }

    /* Improve metric visibility */
    .stMetric {
        background-color: #1a1a1a !important;
        border: 1px solid #00ffff !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Apply dark theme to the entire app
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stSelectbox, .stSlider, .stDateInput {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚛 Market Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Professional Load-to-Truck Ratio Analysis & Market Insights**")


# [ALL YOUR EXISTING DATA LOADING FUNCTIONS REMAIN EXACTLY THE SAME]
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
        elif "city" in lc:
            col_map[c] = "City"
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
    """Aggregate data by selected frequency - CONSISTENT aggregation for all metrics"""
    out = []
    for (market, typ), g in df.groupby(["market", "type_of_trailer"]):
        g = g.set_index("date").sort_index()

        if freq == "Daily":
            res = g.resample("D").agg({"loads": "mean", "trucks": "mean", "ratio": "mean"}).reset_index()
        elif freq == "Weekly":
            res = g.resample("W-MON").agg({"loads": "mean", "trucks": "mean", "ratio": "mean"}).reset_index()
        elif freq == "Monthly":
            res = g.resample("M").agg({"loads": "mean", "trucks": "mean", "ratio": "mean"}).reset_index()
        elif freq == "Quarterly":
            res = g.resample("Q").agg({"loads": "mean", "trucks": "mean", "ratio": "mean"}).reset_index()
        else:
            res = g.resample("M").agg({"loads": "mean", "trucks": "mean", "ratio": "mean"}).reset_index()

        res["market"] = market
        res["type_of_trailer"] = typ
        out.append(res)

    if out:
        return pd.concat(out, ignore_index=True)
    else:
        return pd.DataFrame(columns=["date", "loads", "trucks", "ratio", "market", "type_of_trailer"])


def create_market_city_mapping(map_df):
    """Create mapping between cities, states, and markets"""
    if map_df is None or map_df.empty:
        return None

    mapping_data = []
    for _, row in map_df.iterrows():
        market = row.get('Market', '')
        state = row.get('State', '')
        city = row.get('City', '')
        abb = row.get('Abb', '')

        if market and state:
            mapping_data.append({
                'market': market,
                'state': state,
                'city': city if city else market,  # Use market name if city not available
                'state_abb': abb
            })

    return pd.DataFrame(mapping_data)


def filter_by_date_range(df, start_date, end_date):
    """Filter data based on custom date range"""
    if start_date and end_date:
        return df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
    return df


def filter_by_timeframe(df, timeframe):
    """Filter data based on selected timeframe"""
    if timeframe == "Last 7 Days":
        cutoff = datetime.now() - timedelta(days=7)
    elif timeframe == "Last 30 Days":
        cutoff = datetime.now() - timedelta(days=30)
    elif timeframe == "Last 90 Days":
        cutoff = datetime.now() - timedelta(days=90)
    elif timeframe == "Last 6 Months":
        cutoff = datetime.now() - timedelta(days=180)
    elif timeframe == "Last 12 Months":
        cutoff = datetime.now() - timedelta(days=365)
    else:  # All Available Data
        return df

    return df[df['date'] >= cutoff]


def create_us_heatmap(analysis_data, map_df, trailer_type, timeframe):
    """Create US heat map with L/T ratios"""
    if analysis_data.empty or map_df.empty:
        return None

    # Merge with map data to get state information
    regional_data = analysis_data.merge(
        map_df[['Market', 'State', 'Abb']].drop_duplicates(),
        left_on='market', right_on='Market', how='left'
    )

    if regional_data.empty or 'Abb' not in regional_data.columns:
        return None

    # Aggregate by state for the heat map
    state_heat = regional_data.groupby(['State', 'Abb']).agg({
        'ratio': 'mean',
        'loads': 'mean',
        'trucks': 'mean',
        'market': 'count'
    }).reset_index()

    state_heat = state_heat.rename(columns={'market': 'market_count'})

    # Create the heat map with dark theme
    fig = px.choropleth(
        state_heat,
        locations='Abb',
        locationmode='USA-states',
        color='ratio',
        hover_name='State',
        hover_data={
            'ratio': ':.2f',
            'loads': ':,.0f',
            'trucks': ':,.0f',
            'market_count': True,
            'Abb': False
        },
        scope='usa',
        color_continuous_scale='reds',
        title=f'US Market Heat Map - {trailer_type} ({timeframe})',
        labels={'ratio': 'Load-to-Truck Ratio'}
    )

    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        geo=dict(
            lakecolor='rgb(0,0,0)',
            landcolor='rgb(30,30,30)',
            bgcolor='rgba(0,0,0,0)'
        )
    )

    return fig


def display_market_insights(market_data, market_name, trailer_type, timeframe):
    """Display professional business insights using ONLY Streamlit native components"""
    if market_data.empty:
        st.warning("Insufficient data for analysis.")
        return

    current_ratio = market_data['ratio'].iloc[-1] if not market_data.empty else 0
    avg_ratio = market_data['ratio'].mean()
    volatility = market_data['ratio'].std()

    # Use AVERAGE loads and trucks (consistent aggregation)
    avg_loads = market_data['loads'].mean()
    avg_trucks = market_data['trucks'].mean()

    # FIXED: Correct market condition logic
    deviation_pct = ((current_ratio - avg_ratio) / avg_ratio) * 100

    if deviation_pct > 20:
        condition = "High Demand"
        css_class = "market-status-high"
        implication = "Carrier advantage - potential for rate increases"
        trend = "Above Average"
    elif deviation_pct < -20:
        condition = "Excess Capacity"
        css_class = "market-status-low"
        implication = "Shipper advantage - potential for cost savings"
        trend = "Below Average"
    else:
        condition = "Market Equilibrium"
        css_class = "market-status-balanced"
        implication = "Balanced negotiation power"
        trend = "Near Average"

    # Display Market Status
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    st.subheader(f"📈 Market Condition: {condition}")
    st.write(f"**{implication}**")
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Metrics in columns - USING ONLY STREAMLIT NATIVE METRICS
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Current L/T Ratio",
            f"{current_ratio:.2f}",
            f"{deviation_pct:+.1f}%"
        )
        st.caption("vs Historical Average")
    with col2:
        st.metric(
            "Historical Average",
            f"{avg_ratio:.2f}"
        )
        st.caption("Period Average")
    with col3:
        st.metric(
            "Market Position",
            trend
        )
        st.caption("Market Condition")
    with col4:
        st.metric(
            "Data Periods",
            f"{len(market_data)}"
        )
        st.caption("Analysis Points")

    # Volume Analysis
    st.markdown('<div class="insight-section">', unsafe_allow_html=True)
    st.subheader("📈 Volume Analysis")

    vol_col1, vol_col2, vol_col3 = st.columns(3)
    with vol_col1:
        st.metric(
            "Average Loads",
            f"{avg_loads:,.0f}"
        )
        st.caption("Per Period")
    with vol_col2:
        st.metric(
            "Average Trucks",
            f"{avg_trucks:,.0f}"
        )
        st.caption("Per Period")
    with vol_col3:
        utilization = (avg_loads / avg_trucks) if avg_trucks > 0 else 0
        st.metric(
            "Load-to-Truck Ratio",
            f"{utilization:.2f}"
        )
        st.caption("Utilization")
    st.markdown('</div>', unsafe_allow_html=True)

    # Market Context
    st.markdown('<div class="insight-section">', unsafe_allow_html=True)
    st.subheader("⏰ Market Context")

    ctx_col1, ctx_col2 = st.columns(2)
    with ctx_col1:
        st.write(f"**Analysis Period:** {timeframe}")
        st.write(
            f"**Market Stability:** {'High volatility' if volatility > avg_ratio * 0.3 else 'Moderate stability' if volatility > avg_ratio * 0.15 else 'Low volatility'}")
    with ctx_col2:
        st.write(f"**Equipment Type:** {trailer_type}")
        st.write(f"**Market:** {market_name}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Strategic Implications
    st.markdown('<div class="insight-section">', unsafe_allow_html=True)
    st.subheader("💡 Strategic Implications")

    if condition == "High Demand":
        st.success(
            f"**🚛 Carrier Opportunity:** Consider rate increases of 5-10% to capitalize on strong market conditions")
        st.info("**📦 Shipper Consideration:** Secure capacity early and explore alternative routing options")
    elif condition == "Excess Capacity":
        st.info(
            f"**📦 Shipper Opportunity:** Negotiate rates 5-15% below current levels due to favorable market conditions")
        st.warning("**🚛 Carrier Consideration:** Focus on operational efficiency and explore adjacent markets")
    else:
        st.success("**⚖️ Balanced Market:** Maintain current pricing strategy with focus on service differentiation")
        st.info("**🤝 Both Parties:** Focus on building long-term relationships and consistent service levels")
    st.markdown('</div>', unsafe_allow_html=True)


# [THE REST OF YOUR MAIN FUNCTION REMAINS EXACTLY THE SAME - NO CHANGES NEEDED]
def main():
    # Sidebar - File Upload with dark theme
    st.sidebar.markdown("""
    <style>
    .sidebar-header {
        color: #00ffff;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-header">📁 Data Configuration</div>', unsafe_allow_html=True)
    lt_file = st.sidebar.file_uploader("Upload L/T Data File", type=["xls", "xlsx"], key="lt_upload")
    map_file = st.sidebar.file_uploader("Upload Market Map File", type=["xls", "xlsx"], key="map_upload")

    # Navigation to advanced pages
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-header">🔬 Advanced Analytics</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='color: #cccccc; font-size: 0.9rem;'>
    Access specialized analytical tools:
    - **Market Analysis Lab**: Structural stability & ML forecasting
    - **Dynamic Analysis**: Regime switching & equilibrium analysis
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📊 Market Lab", use_container_width=True):
            st.switch_page("pages/markets.py")
    with col2:
        if st.button("🔄 Dynamic Analysis", use_container_width=True):
            st.switch_page("pages/advanced_markets.py")

    if lt_file and map_file:
        # Load data
        with st.spinner("Loading and processing data..."):
            lt_raw = read_excel_safely(lt_file)
            map_raw = read_excel_safely(map_file)

            if lt_raw is not None and map_raw is not None:
                lt_df = load_lt(lt_raw)
                map_df = load_map(map_raw)

                if lt_df is not None and map_df is not None:
                    # Create city-market mapping
                    city_mapping = create_market_city_mapping(map_df)

                    # Sidebar Filters
                    st.sidebar.markdown("---")
                    st.sidebar.markdown('<div class="sidebar-header">🎛️ Analysis Filters</div>', unsafe_allow_html=True)

                    # Date Range Filter
                    st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
                    st.sidebar.subheader("📅 Date Range")
                    use_custom_date = st.sidebar.checkbox("Use Custom Date Range", value=False)

                    if use_custom_date:
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
                        with col2:
                            end_date = st.date_input("End Date", value=datetime.now())
                        timeframe_display = f"{start_date} to {end_date}"
                    else:
                        # Standard timeframe filter
                        timeframe = st.sidebar.selectbox(
                            "Time Period",
                            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last 12 Months",
                             "All Available Data"],
                            index=2
                        )
                        start_date = None
                        end_date = None
                        timeframe_display = timeframe
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)

                    # Aggregation level
                    st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
                    st.sidebar.subheader("📊 Aggregation")
                    aggregation = st.sidebar.selectbox(
                        "Aggregation Level",
                        ["Daily", "Weekly", "Monthly", "Quarterly"],
                        index=2
                    )
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)

                    # Trailer type filter
                    st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
                    st.sidebar.subheader("🚛 Equipment Type")
                    trailer_types = lt_df['type_of_trailer'].unique()
                    selected_trailer = st.sidebar.selectbox(
                        "Select Equipment Type",
                        trailer_types,
                        index=0 if len(trailer_types) > 0 else 0
                    )
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)

                    # Location filters
                    st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
                    st.sidebar.subheader("📍 Location Selection")

                    # State filter
                    if city_mapping is not None:
                        states = sorted(city_mapping['state'].unique())
                        selected_state = st.sidebar.selectbox("Select State", ["All States"] + states)

                        # City filter based on state
                        if selected_state != "All States":
                            cities = sorted(city_mapping[city_mapping['state'] == selected_state]['city'].unique())
                        else:
                            cities = sorted(city_mapping['city'].unique())

                        selected_city = st.sidebar.selectbox("Select City", ["All Cities"] + cities)

                        # Market filter
                        if selected_city != "All Cities":
                            markets = city_mapping[city_mapping['city'] == selected_city]['market'].unique()
                        elif selected_state != "All States":
                            markets = city_mapping[city_mapping['state'] == selected_state]['market'].unique()
                        else:
                            markets = city_mapping['market'].unique()
                    else:
                        markets = lt_df['market'].unique()
                        selected_state = "All States"
                        selected_city = "All Cities"

                    selected_market = st.sidebar.selectbox("Select Market", ["All Markets"] + sorted(markets))
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)

                    # Process data based on filters
                    with st.spinner("Applying filters and aggregating data..."):
                        # Filter by trailer type
                        filtered_data = lt_df[lt_df['type_of_trailer'] == selected_trailer]

                        # Preprocess and aggregate
                        lt_full = preprocess_and_reindex(filtered_data)
                        lt_agg = aggregate_df(lt_full, aggregation)

                        # Apply date filtering
                        if use_custom_date and start_date and end_date:
                            lt_agg = filter_by_date_range(lt_agg, start_date, end_date)
                        elif not use_custom_date:
                            lt_agg = filter_by_timeframe(lt_agg, timeframe)

                        # Filter by market if specific market selected
                        if selected_market != "All Markets":
                            market_data = lt_agg[lt_agg['market'] == selected_market].copy()
                        elif selected_city != "All Cities" and city_mapping is not None:
                            # Filter by city
                            city_markets = city_mapping[city_mapping['city'] == selected_city]['market'].unique()
                            market_data = lt_agg[lt_agg['market'].isin(city_markets)].copy()
                        elif selected_state != "All States" and city_mapping is not None:
                            # Filter by state
                            state_markets = city_mapping[city_mapping['state'] == selected_state]['market'].unique()
                            market_data = lt_agg[lt_agg['market'].isin(state_markets)].copy()
                        else:
                            market_data = lt_agg.copy()

                    # Main Dashboard Content
                    st.markdown('<div class="sub-header">🌎 US Market Overview</div>', unsafe_allow_html=True)

                    # US Heat Map
                    heatmap_fig = create_us_heatmap(lt_agg, map_df, selected_trailer, timeframe_display)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    else:
                        st.warning("Could not generate heat map. Please check if map data contains state information.")

                    # Market Insights and Detailed Analysis
                    if selected_market != "All Markets":
                        st.markdown('<div class="sub-header">📋 Market Detailed Analysis</div>', unsafe_allow_html=True)

                        # Display insights using native Streamlit components
                        display_market_insights(market_data, selected_market, selected_trailer, timeframe_display)

                        # Charts
                        if not market_data.empty:
                            col1, col2 = st.columns(2)

                            with col1:
                                # L/T Ratio Over Time
                                avg_ratio = market_data['ratio'].mean()
                                fig_ratio = px.line(
                                    market_data, x='date', y='ratio',
                                    title=f"{selected_market} - Load-to-Truck Ratio Trend",
                                    labels={'ratio': 'L/T Ratio', 'date': 'Date'}
                                )
                                fig_ratio.update_layout(
                                    height=400,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white',
                                    xaxis=dict(gridcolor='#444444'),
                                    yaxis=dict(gridcolor='#444444')
                                )
                                # Add average line
                                fig_ratio.add_hline(y=avg_ratio, line_dash="dash", line_color="#00ffff",
                                                    annotation_text=f"Average: {avg_ratio:.2f}")
                                st.plotly_chart(fig_ratio, use_container_width=True)

                            with col2:
                                # Loads vs Trucks - AVERAGE values
                                fig_volume = go.Figure()
                                fig_volume.add_trace(go.Scatter(
                                    x=market_data['date'], y=market_data['loads'],
                                    name='Average Loads', line=dict(color='#00ffff', width=3),
                                    fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)'
                                ))
                                fig_volume.add_trace(go.Scatter(
                                    x=market_data['date'], y=market_data['trucks'],
                                    name='Average Trucks', line=dict(color='#ff4444', width=3),
                                    fill='tozeroy', fillcolor='rgba(255, 68, 68, 0.1)'
                                ))
                                fig_volume.update_layout(
                                    title=f"{selected_market} - Loads vs Trucks Volume",
                                    xaxis_title="Date",
                                    yaxis_title="Average Count",
                                    height=400,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white',
                                    xaxis=dict(gridcolor='#444444'),
                                    yaxis=dict(gridcolor='#444444'),
                                    legend=dict(
                                        bgcolor='rgba(0,0,0,0.7)',
                                        bordercolor='#444444',
                                        font=dict(color='white')
                                    )
                                )
                                st.plotly_chart(fig_volume, use_container_width=True)

                    else:
                        # Overview for multiple markets
                        st.markdown('<div class="sub-header">📊 Multi-Market Overview</div>', unsafe_allow_html=True)

                        # Top markets summary - Using AVERAGE values
                        market_summary = lt_agg.groupby('market').agg({
                            'ratio': ['mean', 'std'],
                            'loads': 'mean',
                            'trucks': 'mean'
                        }).round(2)

                        market_summary.columns = ['Avg Ratio', 'Volatility', 'Avg Loads', 'Avg Trucks']
                        market_summary = market_summary.sort_values('Avg Ratio', ascending=False)

                        st.dataframe(
                            market_summary,
                            use_container_width=True,
                            height=400
                        )

                        st.info(
                            f"Showing data for {len(market_summary)} markets. Select a specific market for detailed analysis.")

                else:
                    st.error("Error processing data files. Please check the file formats.")
            else:
                st.error("Error loading data files. Please check the file formats.")
    else:
        # Welcome screen when no files uploaded
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #1a1a1a, #2a2a2a); border-radius: 15px; border: 1px solid #333;'>
            <h2 style='color: #00ffff; margin-bottom: 2rem; text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);'>Welcome to Market Intelligence Dashboard</h2>
            <p style='font-size: 1.2rem; margin-bottom: 3rem; color: #cccccc;'>
                Upload your L/T Data and Market Map files to begin analyzing market conditions across the United States.
            </p>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; text-align: left;'>
                <div>
                    <h4 style='color: #00ffff;'>📁 Required Files:</h4>
                    <ul style='color: #cccccc;'>
                        <li><strong>L/T Data File</strong>: Contains load, truck, and ratio data</li>
                        <li><strong>Market Map File</strong>: Maps markets to states and cities</li>
                    </ul>
                </div>
                <div>
                    <h4 style='color: #00ffff;'>🔍 Key Features:</h4>
                    <ul style='color: #cccccc;'>
                        <li>US Market Heat Maps</li>
                        <li>Multi-level filtering</li>
                        <li>Professional analytics</li>
                        <li>Advanced analysis tools</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    