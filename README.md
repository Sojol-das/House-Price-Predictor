# 🏠 House Price Predictor — Machine Learning Web App

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-85.3%25-blue)

---

## 🌐 Live Demo
👉 **https://real-estate-ml-dashboard.streamlit.app** 

---

## 📌 Project Overview

A fully interactive **Machine Learning web application** that predicts house
prices in King County, USA (Seattle area) using a **Gradient Boosting
Regressor** trained on 21,613 real house sales.

A user simply enters house details — bedrooms, size, grade, location — and
the app instantly returns a **predicted price, price range, home category,
comparable properties, and upgrade simulations.**

This project directly mirrors what real estate companies, property platforms,
and investment firms need from data scientists and ML engineers.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🔮 **Price Predictor** | Instant ML-powered price estimate with confidence range |
| 🗺️ **Interactive Map** | Scatter mapbox of all King County sales, filterable by price, grade & waterfront |
| 📊 **Market Insights** | 6 charts covering price distribution, grade impact, bedroom trends, and more |
| 📈 **Price Percentile** | "Your predicted house beats X% of all homes in the dataset" |
| 🔧 **What-If Simulator** | 6 upgrade scenarios showing instant price delta (e.g. add a bedroom → +$X) |
| 🏘️ **Similar Homes** | 8 real comparable homes within ±15% of predicted price |
| 📉 **Price Trend** | Monthly avg vs median price over time |
| 🌊 **Waterfront Premium** | Exact % premium waterfront adds to price |
| 📥 **Download Report** | Full `.txt` prediction report to save or share |
| 💡 **How It Works Tab** | Explains dataset, model, and ML pipeline — transparent and client-friendly |

---

## 🤖 Machine Learning Pipeline
```
Raw Data (21,613 rows)
        ↓
Feature Engineering
(house_age, was_renovated, total_sqft, rooms_total, price_per_sqft)
        ↓
Train/Test Split (80/20)
        ↓
StandardScaler (normalization)
        ↓
Gradient Boosting Regressor
(n_estimators=300, learning_rate=0.1, max_depth=5)
        ↓
Evaluation → R²: 85.3% | MAE: $70,717
        ↓
Saved as model.pkl + scaler.pkl
        ↓
Streamlit Web App (live predictions)
```

---

## 📊 Model Performance

| Metric | Value | Meaning |
|---|---|---|
| **R² Score** | **85.3%** | Model explains 85% of price variation |
| **MAE** | **$70,717** | Avg prediction error (~13% of mean price) |
| **Training Size** | 17,290 houses | 80% of dataset |
| **Test Size** | 4,323 houses | 20% of dataset |
| **Algorithm** | Gradient Boosting | Best for tabular regression tasks |

### 🏆 Top 5 Features by Importance

| Feature | Importance | Why It Matters |
|---|---|---|
| `grade` | 30.4% | Construction & design quality |
| `sqft_living` | 29.1% | Living space size |
| `lat` | 16.9% | Location (north = more expensive) |
| `long` | 7.3% | East-West location |
| `house_age` | 3.5% | Newer homes cost more |

---

## 🗂️ Dataset

| Detail | Info |
|---|---|
| **Source** | [Kaggle — House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) |
| **Rows** | 21,613 house sales |
| **Columns** | 21 features |
| **Location** | King County, Seattle, USA |
| **Missing Values** | None — clean dataset |
| **Target Variable** | `price` (house sale price) |

---

## 🛠️ Tools & Libraries
```python
Python 3.8+
Scikit-learn    — ML model training & evaluation
Pandas          — Data manipulation
NumPy           — Numerical operations
Streamlit       — Live web app deployment
Plotly          — Interactive charts & maps
Pickle          — Model serialization
```

---

## 📁 Project Structure
```
House-Price-Predictor/
├── kc_house_data.csv          ← Raw dataset
├── model.py                   ← Model training script
├── app.py                     ← Streamlit web application
├── model.pkl                  ← Saved trained model
├── scaler.pkl                 ← Saved feature scaler
├── feature_importance.csv     ← Feature importance scores
└── README.md
```

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Sojol-das/House-Price-Predictor.git
cd House-Price-Predictor
```

**2. Install dependencies**
```bash
pip install streamlit scikit-learn pandas numpy plotly
```

**3. Train the model first**
```bash
python model.py
```

**4. Launch the web app**
```bash
streamlit run app.py
```

**5. Open in browser**
```
http://localhost:8501
```

---

## 💼 Business Use Cases

This type of ML application is directly useful for:

- 🏢 **Real Estate Agencies** — Instant property valuations for agents
- 🏦 **Banks & Lenders** — Automated collateral assessment
- 📱 **Property Platforms** — Zillow/MagicBricks-style price estimates
- 💰 **Property Investors** — Quick ROI analysis on potential purchases
- 🏗️ **Developers** — Land & construction value estimation

---

## 💡 Key Business Insights Discovered

1. **Grade is the #1 price driver** — upgrading from grade 7 to 9 adds ~40% to price
2. **Location matters more than size** — latitude explains 16.9% of price variation
3. **Waterfront adds a significant premium** — waterfront homes cost substantially more
4. **Renovated homes command higher prices** — even older renovated homes outperform newer unrenovated ones
5. **Sweet spot is 3–4 bedrooms** — diminishing returns beyond 5 bedrooms

---

