import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# ── Load Data
df = pd.read_csv("kc_house_data.csv")

print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# ── Feature Engineering
df['house_age']     = 2024 - df['yr_built']
df['was_renovated'] = (df['yr_renovated'] != 0).astype(int)
df['total_sqft']    = df['sqft_living'] + df['sqft_basement']
df['rooms_total']   = df['bedrooms'] + df['bathrooms']
df['price_per_sqft']= df['price'] / df['sqft_living']

# ── Select Features
features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'house_age',
    'was_renovated', 'total_sqft', 'rooms_total', 'lat', 'long'
]

X = df[features]
y = df['price']

# ── Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train Model (Gradient Boosting — best for tabular data)
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ── Evaluate
y_pred = model.predict(X_test_scaled)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n✅ Model Training Complete!")
print(f"   R² Score  : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"   MAE       : ${mae:,.0f}  (avg prediction error)")

# ── Save Model & Scaler
with open("model.pkl",  "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ── Save feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

importance_df.to_csv("feature_importance.csv", index=False)

print("\n✅ model.pkl and scaler.pkl saved successfully!")
print("\nTop 5 Important Features:")
print(importance_df.head())