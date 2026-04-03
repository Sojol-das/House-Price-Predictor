import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import datetime

# ── Color Palette
PRIMARY   = "#1B4F72"
SECONDARY = "#2E86C1"
ACCENT    = "#27AE60"
NEGATIVE  = "#C0392B"
GRAY      = "#566573"
WARN      = "#F39C12"
PURPLE    = "#8E44AD"
COLORS    = [PRIMARY, SECONDARY, ACCENT, WARN, PURPLE]

# ── Page Config
st.set_page_config(
    page_title="House Price Predictor | King County",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load Model & Scaler
@st.cache_resource
def load_model():
    with open("model.pkl",  "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Load Feature Importance
@st.cache_data
def load_importance():
    return pd.read_csv("feature_importance.csv")

importance_df = load_importance()

# ── Load Raw Data
@st.cache_data
def load_data():
    df = pd.read_csv("kc_house_data.csv")
    df['date']           = pd.to_datetime(df['date'], format='%Y%m%dT000000')
    df['month']          = df['date'].dt.to_period('M').astype(str)
    df['house_age']      = 2024 - df['yr_built']
    df['was_renovated']  = (df['yr_renovated'] != 0).astype(int)
    df['total_sqft']     = df['sqft_living'] + df['sqft_basement']
    df['rooms_total']    = df['bedrooms'] + df['bathrooms']
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    return df

df = load_data()

# ── Prediction helper (reusable)
def predict_price(bedrooms, bathrooms, sqft_living, sqft_lot,
                  floors, waterfront, view, condition, grade,
                  sqft_above, sqft_basement, house_age,
                  was_renovated, total_sqft, rooms_total, lat, long):
    inp = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                     floors, waterfront, view, condition, grade,
                     sqft_above, sqft_basement, house_age,
                     was_renovated, total_sqft, rooms_total, lat, long]])
    return model.predict(scaler.transform(inp))[0]

# ── Custom CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .main {{ background-color: #f0f4f8; }}

    /* Metric cards */
    [data-testid="stMetricValue"] {{
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: {PRIMARY} !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: {GRAY} !important;
    }}

    /* Predict Button */
    div.stButton > button {{
        background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 18px 0px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(27,79,114,0.4);
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(90deg, {SECONDARY}, {PRIMARY});
        box-shadow: 0 6px 20px rgba(27,79,114,0.6);
        transform: translateY(-2px);
    }}

    /* Slider labels */
    .stSlider label {{
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: {PRIMARY} !important;
    }}

    /* Section headers */
    h1 {{ font-family: 'Outfit', sans-serif !important; }}
    h2, h3, h4 {{
        font-family: 'Outfit', sans-serif !important;
        color: {PRIMARY} !important;
    }}

    /* Tabs styling */
    button[data-baseweb="tab"] {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}

    /* Expander */
    details summary {{
        font-weight: 600;
        color: {PRIMARY};
    }}

    /* Download button */
    div.stDownloadButton > button {{
        background: {ACCENT} !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
    }}
    div.stDownloadButton > button:hover {{
        background: #219653 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# HEADER
# ════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
     padding: 32px 36px; border-radius: 16px; margin-bottom: 20px;
     box-shadow: 0 8px 32px rgba(27,79,114,0.25);'>
  <h1 style='color:white; font-family:Outfit,sans-serif; font-size:2.5rem;
             font-weight:800; margin:0; letter-spacing:-0.5px;'>
    🏠 House Price Predictor
  </h1>
  <p style='color:rgba(255,255,255,0.82); font-size:1.05rem; margin:8px 0 16px 0;'>
    AI-powered valuation tool · King County, USA · 21,613 real transactions
  </p>
  <div style='display:flex; gap:28px; flex-wrap:wrap;'>
    <span style='background:rgba(255,255,255,0.15); color:white; padding:6px 16px;
                 border-radius:20px; font-size:0.9rem; font-weight:600;'>
      🤖 Gradient Boosting
    </span>
    <span style='background:rgba(255,255,255,0.15); color:white; padding:6px 16px;
                 border-radius:20px; font-size:0.9rem; font-weight:600;'>
      📊 R² Score: 85.3%
    </span>
    <span style='background:rgba(255,255,255,0.15); color:white; padding:6px 16px;
                 border-radius:20px; font-size:0.9rem; font-weight:600;'>
      📉 Avg Error: $70,717
    </span>
    <span style='background:rgba(255,255,255,0.15); color:white; padding:6px 16px;
                 border-radius:20px; font-size:0.9rem; font-weight:600;'>
      🏘️ 21,613 houses trained
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# TABS
# ════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Predict Price",
    "🗺️  Map Explorer",
    "📊  Market Insights",
    "🤖  How It Works"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🏗️ Configure Your House")
    st.markdown("Adjust the parameters below and click **Predict** to get an instant AI price estimate.")

    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            bedrooms  = st.slider("🛏️ Bedrooms",        1, 10, 3)
            bathrooms = st.slider("🚿 Bathrooms",        1, 8,  2)
            floors    = st.slider("🏢 Floors",           1, 4,  1)
            condition = st.slider("🔧 Condition (1–5)",  1, 5,  3,
                help="1=Poor  3=Average  5=Excellent")
        with c2:
            grade      = st.slider("⭐ Grade (1–13)",       1, 13, 7,
                help="Construction & design quality. 7=Average, 10+=Luxury")
            view       = st.slider("👁️ View Quality (0–4)", 0, 4,  0,
                help="0=No view  4=Excellent view")
            waterfront = st.selectbox("🌊 Waterfront",
                options=[0, 1], format_func=lambda x: "Yes 🌊" if x == 1 else "No")
            house_age  = st.slider("📅 House Age (years)", 0, 120, 20)

        st.markdown("##### 📐 Size & Location")
        c3, c4 = st.columns(2)
        with c3:
            sqft_living   = st.slider("🏠 Living Area (sqft)",  200,  14000, 2000)
            sqft_lot      = st.slider("🌿 Lot Size (sqft)",     500, 150000, 8000)
            sqft_above    = st.slider("⬆️ Above Ground (sqft)", 200,  10000, 1500)
        with c4:
            sqft_basement = st.slider("⬇️ Basement (sqft)",       0,   5000,  500)
            lat           = st.slider("📍 Latitude",  47.15, 47.78, 47.50,
                help="North = higher value areas")
            long          = st.slider("📍 Longitude", -122.51, -121.31, -122.00,
                help="West Seattle = more expensive")

        was_renovated = 1 if house_age > 20 else 0
        total_sqft    = sqft_living + sqft_basement
        rooms_total   = bedrooms + bathrooms

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  PREDICT HOUSE PRICE  🔮", use_container_width=True)

    # ── Result Panel
    with col_result:
        st.markdown("### 📊 Prediction Result")

        if predict_btn:
            predicted_price = predict_price(
                bedrooms, bathrooms, sqft_living, sqft_lot,
                floors, waterfront, view, condition, grade,
                sqft_above, sqft_basement, house_age,
                was_renovated, total_sqft, rooms_total, lat, long
            )

            low_est  = predicted_price * 0.92
            high_est = predicted_price * 1.08

            # ── Main price card
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
                 padding:32px; border-radius:16px; text-align:center;
                 box-shadow:0 6px 24px rgba(27,79,114,0.3); margin-top:8px;'>
                <p style='color:rgba(255,255,255,0.8); font-size:1rem; margin:0;
                          letter-spacing:2px; font-family:Outfit,sans-serif;'>
                    ESTIMATED PRICE
                </p>
                <h1 style='font-size:3rem; margin:10px 0; color:white;
                           font-weight:800; font-family:Outfit,sans-serif;
                           letter-spacing:-1px;'>
                    ${predicted_price:,.0f}
                </h1>
                <p style='color:rgba(255,255,255,0.75); font-size:0.9rem; margin:0;'>
                    Range &nbsp;·&nbsp; ${low_est:,.0f} – ${high_est:,.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Category badge
            if predicted_price < 300_000:
                label, color = "🟢 Budget Home",  ACCENT
            elif predicted_price < 600_000:
                label, color = "🔵 Mid-Range Home", SECONDARY
            elif predicted_price < 1_000_000:
                label, color = "🟡 Premium Home",  WARN
            else:
                label, color = "💎 Luxury Home",   PURPLE

            st.markdown(f"""
            <div style='background:{color}; padding:12px; border-radius:10px;
                 text-align:center; margin-top:12px;'>
                <span style='color:white; font-size:1.2rem; font-weight:700;
                             font-family:Outfit,sans-serif;'>{label}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Market Percentile
            percentile = (df['price'] < predicted_price).mean() * 100
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:white; padding:18px 22px; border-radius:12px;
                 border-left:5px solid {SECONDARY}; box-shadow:0 2px 10px rgba(0,0,0,0.07);'>
                <p style='margin:0; color:{GRAY}; font-size:0.85rem; font-weight:600;
                          letter-spacing:1px;'>MARKET POSITION</p>
                <p style='margin:4px 0 0 0; color:{PRIMARY}; font-size:1.3rem; font-weight:800;
                          font-family:Outfit,sans-serif;'>
                    More expensive than {percentile:.1f}% of homes
                </p>
                <p style='margin:2px 0 0 0; color:{GRAY}; font-size:0.88rem;'>
                    Top {100-percentile:.1f}% of King County market
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Percentile gauge bar
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentile,
                number={'suffix': '%', 'font': {'size': 28, 'color': PRIMARY}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'size': 11}},
                    'bar': {'color': SECONDARY, 'thickness': 0.3},
                    'bgcolor': '#f0f4f8',
                    'bordercolor': '#dde',
                    'steps': [
                        {'range': [0,   33],  'color': '#d5f5e3'},
                        {'range': [33,  66],  'color': '#d6eaf8'},
                        {'range': [66,  100], 'color': '#d2b4de'},
                    ],
                    'threshold': {
                        'line': {'color': NEGATIVE, 'width': 3},
                        'thickness': 0.8,
                        'value': percentile
                    }
                },
                title={'text': "Market Percentile", 'font': {'size': 14, 'color': GRAY}}
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Input Summary
            st.markdown("#### 📋 Input Summary")
            summary = pd.DataFrame({
                "Feature": ["Bedrooms","Bathrooms","Living Area",
                            "Grade","Condition","House Age",
                            "Waterfront","View Quality"],
                "Value":   [bedrooms, bathrooms, f"{sqft_living:,} sqft",
                            f"{grade}/13", f"{condition}/5", f"{house_age} yrs",
                            "Yes 🌊" if waterfront else "No", f"{view}/4"]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # ── Download Report
            st.markdown("<br>", unsafe_allow_html=True)
            report_lines = [
                "HOUSE PRICE PREDICTION REPORT",
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "=" * 40,
                f"Predicted Price : ${predicted_price:,.0f}",
                f"Price Range     : ${low_est:,.0f} – ${high_est:,.0f}",
                f"Category        : {label}",
                f"Market Position : Top {100-percentile:.1f}% (beats {percentile:.1f}% of homes)",
                "",
                "INPUT PARAMETERS",
                "-" * 40,
                f"Bedrooms        : {bedrooms}",
                f"Bathrooms       : {bathrooms}",
                f"Living Area     : {sqft_living:,} sqft",
                f"Lot Size        : {sqft_lot:,} sqft",
                f"Grade           : {grade}/13",
                f"Condition       : {condition}/5",
                f"House Age       : {house_age} years",
                f"Waterfront      : {'Yes' if waterfront else 'No'}",
                f"View Quality    : {view}/4",
                f"Floors          : {floors}",
                f"Basement        : {sqft_basement:,} sqft",
                f"Latitude        : {lat}",
                f"Longitude       : {long}",
                "",
                "Model: Gradient Boosting Regressor | R² = 85.3%",
                "Data: King County, USA — 21,613 transactions"
            ]
            report_text = "\n".join(report_lines)
            st.download_button(
                label="📥 Download Prediction Report",
                data=report_text,
                file_name="house_price_prediction.txt",
                mime="text/plain",
                use_container_width=True
            )

        else:
            st.markdown(f"""
            <div style='background:white; padding:50px 30px; border-radius:16px;
                 text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.07);
                 margin-top:20px; border:2px dashed {SECONDARY};'>
                <p style='font-size:4rem; margin:0;'>🏠</p>
                <p style='color:{GRAY}; font-size:1.1rem; margin-top:16px; line-height:1.6;'>
                    Adjust the sliders on the left<br>and click
                    <b style='color:{PRIMARY};'>Predict</b> to get your estimate!
                </p>
                <p style='color:{SECONDARY}; font-size:0.9rem; margin-top:12px;'>
                    ✅ Instant AI valuation &nbsp;·&nbsp;
                    📊 Market rank &nbsp;·&nbsp;
                    📥 Downloadable report
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════
    # WHAT-IF SIMULATOR
    # ═══════════════════════════════════
    if predict_btn:
        st.markdown("---")
        st.markdown("### 💡 What-If Simulator")
        st.markdown("See how upgrades could change your home's value:")

        base_price = predicted_price

        scenarios = {
            "➕ Add 1 Bedroom":            dict(bedrooms=bedrooms+1),
            "⬆️ Upgrade Grade by 1":        dict(grade=min(grade+1, 13)),
            "🔧 Improve Condition by 1":    dict(condition=min(condition+1, 5)),
            "📐 Add 500 sqft Living Area":  dict(sqft_living=sqft_living+500,
                                                  sqft_above=sqft_above+500,
                                                  total_sqft=total_sqft+500),
            "🌊 Add Waterfront":            dict(waterfront=1),
            "👁️ Improve View by 1":         dict(view=min(view+1, 4)),
        }

        whatif_cols = st.columns(3)
        for i, (scenario_name, overrides) in enumerate(scenarios.items()):
            params = dict(
                bedrooms=bedrooms, bathrooms=bathrooms, sqft_living=sqft_living,
                sqft_lot=sqft_lot, floors=floors, waterfront=waterfront,
                view=view, condition=condition, grade=grade,
                sqft_above=sqft_above, sqft_basement=sqft_basement,
                house_age=house_age, was_renovated=was_renovated,
                total_sqft=total_sqft, rooms_total=rooms_total,
                lat=lat, long=long
            )
            params.update(overrides)
            new_price = predict_price(**params)
            delta     = new_price - base_price
            delta_pct = (delta / base_price) * 100
            arrow     = "▲" if delta > 0 else "▼"
            clr       = ACCENT if delta > 0 else NEGATIVE

            with whatif_cols[i % 3]:
                st.markdown(f"""
                <div style='background:white; padding:18px; border-radius:12px;
                     box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-bottom:12px;
                     border-top:4px solid {clr};'>
                    <p style='margin:0; font-size:0.9rem; font-weight:600; color:{GRAY};'>
                        {scenario_name}
                    </p>
                    <p style='margin:8px 0 2px 0; font-size:1.5rem; font-weight:800;
                              color:{PRIMARY}; font-family:Outfit,sans-serif;'>
                        ${new_price:,.0f}
                    </p>
                    <p style='margin:0; font-size:1rem; font-weight:700; color:{clr};'>
                        {arrow} ${abs(delta):,.0f} ({delta_pct:+.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # ── Similar Homes Table
        st.markdown("---")
        st.markdown("### 🏘️ Similar Homes in the Dataset")
        st.markdown(f"Showing homes priced within **±15%** of your predicted price (${base_price:,.0f})")

        similar = df[
            (df['price'] >= base_price * 0.85) &
            (df['price'] <= base_price * 1.15)
        ][['price', 'bedrooms', 'bathrooms', 'sqft_living', 'grade',
           'condition', 'house_age', 'waterfront', 'price_per_sqft']].copy()

        similar = similar.sort_values('price').head(8).reset_index(drop=True)
        similar.columns = ['Price ($)', 'Beds', 'Baths', 'Sqft Living',
                           'Grade', 'Condition', 'Age (yrs)',
                           'Waterfront', 'Price/sqft']
        similar['Price ($)']   = similar['Price ($)'].apply(lambda x: f"${x:,.0f}")
        similar['Price/sqft']  = similar['Price/sqft'].apply(lambda x: f"${x:,.0f}")
        similar['Waterfront']  = similar['Waterfront'].apply(lambda x: "Yes 🌊" if x else "No")
        st.dataframe(similar, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — MAP EXPLORER
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ King County Interactive Map")
    st.markdown("Explore real house sale prices across the region. Each dot is an actual transaction.")

    col_map_ctrl, col_map_main = st.columns([1, 3])

    with col_map_ctrl:
        st.markdown("#### 🎛️ Filters")
        price_range = st.slider(
            "Price Range ($)",
            int(df['price'].min()), int(df['price'].max()),
            (100_000, 1_500_000), step=50_000,
            format="$%d"
        )
        min_grade = st.slider("Minimum Grade", 1, 13, 1)
        wf_filter = st.selectbox("Waterfront Only?",
            options=["All", "Waterfront Only", "Non-Waterfront"])
        map_sample = st.slider("Map Sample Size", 500, 5000, 2000, step=500,
            help="More points = slower render")
        color_by = st.selectbox("Color By",
            options=["price", "grade", "condition", "price_per_sqft", "sqft_living", "house_age"])

    with col_map_main:
        map_df = df[
            (df['price'] >= price_range[0]) &
            (df['price'] <= price_range[1]) &
            (df['grade']  >= min_grade)
        ].copy()
        if wf_filter == "Waterfront Only":
            map_df = map_df[map_df['waterfront'] == 1]
        elif wf_filter == "Non-Waterfront":
            map_df = map_df[map_df['waterfront'] == 0]

        map_df = map_df.sample(min(map_sample, len(map_df)), random_state=42)

        color_labels = {
            "price":          "Price ($)",
            "grade":          "Grade",
            "condition":      "Condition",
            "price_per_sqft": "Price/sqft ($)",
            "sqft_living":    "Living Area (sqft)",
            "house_age":      "House Age (yrs)"
        }

        fig_map = px.scatter_mapbox(
            map_df, lat='lat', lon='long',
            color=color_by,
            size='sqft_living',
            size_max=14,
            color_continuous_scale='RdYlGn_r' if color_by == 'house_age' else 'RdYlGn',
            hover_data={
                'price': ':,.0f',
                'bedrooms': True,
                'bathrooms': True,
                'sqft_living': ':,',
                'grade': True,
                'condition': True,
                'lat': False, 'long': False
            },
            labels={color_by: color_labels.get(color_by, color_by)},
            mapbox_style='carto-positron',
            zoom=9.5,
            center={"lat": 47.50, "lon": -122.00},
            opacity=0.75,
            height=580
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(
                title=color_labels.get(color_by, color_by),
                thickness=14, len=0.6
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown(f"""
        <p style='color:{GRAY}; font-size:0.85rem; text-align:center;'>
            Showing <b>{len(map_df):,}</b> of <b>{len(df[
                (df['price'] >= price_range[0]) &
                (df['price'] <= price_range[1]) &
                (df['grade']  >= min_grade)
            ]):,}</b> filtered homes
        </p>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — MARKET INSIGHTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 King County Market Insights")

    # ── KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("🏘️ Total Houses",    f"{len(df):,}")
    with k2: st.metric("💰 Avg Price",        f"${df['price'].mean():,.0f}")
    with k3: st.metric("📈 Median Price",     f"${df['price'].median():,.0f}")
    with k4: st.metric("💎 Highest Price",    f"${df['price'].max():,.0f}")
    with k5: st.metric("📉 Lowest Price",     f"${df['price'].min():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💰 Price Distribution")
        fig1 = px.histogram(df, x='price', nbins=60,
            color_discrete_sequence=[SECONDARY],
            labels={'price': 'Price ($)', 'count': 'Number of Houses'})
        fig1.add_vline(x=df['price'].mean(), line_dash="dash", line_color=NEGATIVE,
            annotation_text=f"Mean ${df['price'].mean():,.0f}",
            annotation_position="top right")
        fig1.add_vline(x=df['price'].median(), line_dash="dot", line_color=ACCENT,
            annotation_text=f"Median ${df['price'].median():,.0f}",
            annotation_position="top left")
        fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(tickfont=dict(size=12), gridcolor='#f0f0f0'),
            yaxis=dict(tickfont=dict(size=12), gridcolor='#f0f0f0'))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("#### 📈 Price Trend Over Time")
        monthly = df.groupby('month')['price'].agg(['mean', 'median']).reset_index()
        monthly.columns = ['month', 'mean_price', 'median_price']
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly['month'], y=monthly['mean_price'],
            name='Avg Price', line=dict(color=SECONDARY, width=2.5),
            fill='tozeroy', fillcolor='rgba(46,134,193,0.08)'
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly['month'], y=monthly['median_price'],
            name='Median Price', line=dict(color=ACCENT, width=2, dash='dash')
        ))
        fig_trend.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Month", yaxis_title="Price ($)",
            xaxis=dict(tickangle=45, tickfont=dict(size=11), gridcolor='#f0f0f0'),
            yaxis=dict(tickfont=dict(size=12), gridcolor='#f0f0f0'),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Row 2
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⭐ Average Price by Grade")
        grade_price = df.groupby('grade')['price'].mean().reset_index()
        fig2 = px.bar(grade_price, x='grade', y='price',
            color='price', color_continuous_scale='Blues', text_auto='.2s')
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            coloraxis_showscale=False,
            xaxis_title="Grade", yaxis_title="Avg Price ($)",
            xaxis=dict(tickfont=dict(size=12)), yaxis=dict(gridcolor='#f0f0f0'))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("#### 📐 Living Area vs Price")
        sample = df.sample(2000, random_state=42)
        fig3 = px.scatter(sample, x='sqft_living', y='price',
            color='grade', size='bedrooms',
            hover_data=['bedrooms', 'bathrooms', 'condition'],
            color_continuous_scale='Blues', opacity=0.6)
        fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Living Area (sqft)", yaxis_title="Price ($)",
            xaxis=dict(tickfont=dict(size=12), gridcolor='#f0f0f0'),
            yaxis=dict(tickfont=dict(size=12), gridcolor='#f0f0f0'))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Row 3
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🛏️ Avg Price by Bedrooms")
        bed_price = df[df['bedrooms'] <= 8].groupby('bedrooms')['price'].mean().reset_index()
        fig5 = px.bar(bed_price, x='bedrooms', y='price',
            color='price', color_continuous_scale='Blues', text_auto='.2s')
        fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            coloraxis_showscale=False,
            xaxis_title="Bedrooms", yaxis_title="Avg Price ($)",
            yaxis=dict(gridcolor='#f0f0f0'))
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        st.markdown("#### 🤖 Top 10 Feature Importances")
        top_imp = importance_df.head(10)
        fig4 = px.bar(top_imp, x='Importance', y='Feature',
            orientation='h', color='Importance', color_continuous_scale='Blues')
        fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            coloraxis_showscale=False,
            xaxis_title="Importance Score",
            yaxis=dict(showgrid=False))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 4 — Price/Sqft heatmap by grade & condition
    st.markdown("#### 🔥 Price per Sqft — Grade vs Condition Heatmap")
    heat_data = df.groupby(['grade', 'condition'])['price_per_sqft'].mean().reset_index()
    heat_pivot = heat_data.pivot(index='grade', columns='condition', values='price_per_sqft')
    fig_heat = px.imshow(
        heat_pivot, color_continuous_scale='Blues',
        labels=dict(x="Condition", y="Grade", color="$/sqft"),
        aspect='auto'
    )
    fig_heat.update_layout(paper_bgcolor='white', height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Waterfront premium
    st.markdown("#### 🌊 Waterfront Premium Analysis")
    wf_comp = df.groupby('waterfront')['price'].agg(['mean', 'median', 'count']).reset_index()
    wf_comp['waterfront'] = wf_comp['waterfront'].map({0: 'Non-Waterfront', 1: 'Waterfront'})
    wf_comp.columns = ['Type', 'Avg Price', 'Median Price', 'Count']
    premium_pct = (wf_comp.iloc[1]['Avg Price'] / wf_comp.iloc[0]['Avg Price'] - 1) * 100
    col_wf1, col_wf2 = st.columns([1, 2])
    with col_wf1:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
             padding:28px; border-radius:14px; text-align:center; color:white;'>
            <p style='margin:0; font-size:0.9rem; opacity:0.8;'>WATERFRONT PREMIUM</p>
            <p style='margin:8px 0; font-size:2.8rem; font-weight:800;
                      font-family:Outfit,sans-serif;'>
                +{premium_pct:.1f}%
            </p>
            <p style='margin:0; font-size:0.85rem; opacity:0.75;'>
                Average price uplift for waterfront homes
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_wf2:
        fig_wf = px.bar(
            wf_comp.melt(id_vars='Type', value_vars=['Avg Price', 'Median Price']),
            x='Type', y='value', color='variable', barmode='group',
            color_discrete_sequence=[PRIMARY, SECONDARY],
            labels={'value': 'Price ($)', 'variable': ''}
        )
        fig_wf.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(gridcolor='#f0f0f0'))
        st.plotly_chart(fig_wf, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 How the Model Works")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div style='background:white; padding:24px; border-radius:14px;
             box-shadow:0 2px 12px rgba(0,0,0,0.07); margin-bottom:16px;'>
            <h4 style='color:{PRIMARY}; margin-top:0;'>📦 About the Dataset</h4>
            <p style='color:{GRAY}; line-height:1.7;'>
                The model was trained on <b>21,613 real house sales</b> from
                King County, USA (which includes Seattle). The data includes sales
                between <b>May 2014 – May 2015</b> and covers properties with a wide
                range of sizes, locations, and features.
            </p>
            <ul style='color:{GRAY}; line-height:2;'>
                <li>21,613 unique house sales</li>
                <li>19 original features</li>
                <li>5 engineered features added</li>
                <li>Geographic coordinates included</li>
                <li>Prices range: $75K – $7.7M</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:white; padding:24px; border-radius:14px;
             box-shadow:0 2px 12px rgba(0,0,0,0.07);'>
            <h4 style='color:{PRIMARY}; margin-top:0;'>⚙️ Feature Engineering</h4>
            <p style='color:{GRAY}; line-height:1.7;'>
                In addition to raw features, the following were engineered:
            </p>
            <ul style='color:{GRAY}; line-height:2;'>
                <li><b>house_age</b> — 2024 minus year built</li>
                <li><b>was_renovated</b> — binary flag for renovated homes</li>
                <li><b>total_sqft</b> — living + basement area</li>
                <li><b>rooms_total</b> — bedrooms + bathrooms combined</li>
                <li><b>price_per_sqft</b> — for analysis only (not model input)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style='background:white; padding:24px; border-radius:14px;
             box-shadow:0 2px 12px rgba(0,0,0,0.07); margin-bottom:16px;'>
            <h4 style='color:{PRIMARY}; margin-top:0;'>🤖 The ML Model</h4>
            <p style='color:{GRAY}; line-height:1.7;'>
                A <b>Gradient Boosting Regressor</b> was selected after comparing
                multiple algorithms including Linear Regression, Random Forest,
                and XGBoost. Gradient Boosting performed best overall.
            </p>
            <ul style='color:{GRAY}; line-height:2;'>
                <li>Algorithm: Gradient Boosting Regressor</li>
                <li>R² Score: <b>85.3%</b> (test set)</li>
                <li>Mean Absolute Error: <b>$70,717</b></li>
                <li>Train/Test Split: 80% / 20%</li>
                <li>Features scaled with StandardScaler</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:white; padding:24px; border-radius:14px;
             box-shadow:0 2px 12px rgba(0,0,0,0.07);'>
            <h4 style='color:{PRIMARY}; margin-top:0;'>📊 Model Performance</h4>
            <p style='color:{GRAY}; line-height:1.7; margin-bottom:10px;'>
                The model explains <b>85.3%</b> of the variance in house prices.
                The ±8% price range shown in predictions accounts for
                real-world variability not captured by features alone
                (e.g. interior finish, negotiation, market timing).
            </p>
            <p style='color:{GRAY}; line-height:1.7;'>
                <b>Limitations:</b> The model is trained on 2014–15 data.
                Prices in King County have changed significantly since then.
                This tool is best used for <i>relative comparisons</i>
                and feature analysis rather than absolute current valuations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline diagram
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:linear-gradient(90deg,{PRIMARY},{SECONDARY});
         padding:20px 28px; border-radius:14px; text-align:center;'>
        <p style='color:white; font-family:Outfit,sans-serif; font-size:1rem;
                  font-weight:700; margin:0 0 12px 0; letter-spacing:1px;'>
            ML PIPELINE
        </p>
        <div style='display:flex; justify-content:center; align-items:center;
                    flex-wrap:wrap; gap:8px; color:white; font-size:0.95rem; font-weight:600;'>
            <span style='background:rgba(255,255,255,0.15); padding:10px 18px;
                         border-radius:8px;'>📂 Raw Data</span>
            <span>→</span>
            <span style='background:rgba(255,255,255,0.15); padding:10px 18px;
                         border-radius:8px;'>🔧 Feature Engineering</span>
            <span>→</span>
            <span style='background:rgba(255,255,255,0.15); padding:10px 18px;
                         border-radius:8px;'>⚖️ StandardScaler</span>
            <span>→</span>
            <span style='background:rgba(255,255,255,0.15); padding:10px 18px;
                         border-radius:8px;'>🤖 Gradient Boosting</span>
            <span>→</span>
            <span style='background:rgba(255,255,255,0.2); padding:10px 18px;
                         border-radius:8px; border:2px solid rgba(255,255,255,0.5);'>
                💰 Price Prediction
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<hr style='border:1px solid #dde;'>
<div style='text-align:center; padding:12px 0;'>
    <p style='color:{GRAY}; font-size:0.92rem; margin:0;'>
        🏠 House Price Predictor &nbsp;·&nbsp;
        Built by <strong style='color:{PRIMARY};'>Sojol Das</strong> &nbsp;·&nbsp;
        Data Analysis & AI/ML &nbsp;·&nbsp;
        Model Accuracy: 85.3% R²
        <br>
    
        

</div>
""", unsafe_allow_html=True)
