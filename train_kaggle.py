import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import json

print("Loading data...")
# In Kaggle, the path will likely be '../input/your-dataset-name/pgcb_data.csv'
# Upload your dataset to Kaggle to get the correct path.
data_path = '/kaggle/input/datasets/mahmudakon/pgcb-data/pgcb_data.csv'
df = pd.read_csv(data_path)

print(f"Original shape: {df.shape}")

# 1. PREPROCESSING
df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M')
df = df.sort_values('datetime').reset_index(drop=True)

# Impute missing demand_mw with generation_mw + load_shedding (assuming load_shedding is 0 if NaN)
df['load_shedding'] = df['load_shedding'].fillna(0)
# If demand is NaN, use generation + load_shedding
df['demand_mw'] = df['demand_mw'].fillna(df['generation_mw'] + df['load_shedding'])

# Fill remaining NaNs with 0
df = df.fillna(0)

# 2. FEATURE ENGINEERING
print("Creating features...")
def create_features(df):
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for hour (makes 23:00 close to 00:00)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Simple lag features
    df['lag_24h'] = df['demand_mw'].shift(24)
    df['lag_48h'] = df['demand_mw'].shift(48)
    df['lag_168h'] = df['demand_mw'].shift(168) # 1 week ago
    
    return df

df_features = create_features(df)

# Drop rows with NaN from shifting
df_features = df_features.dropna()

features = ['hour', 'dayofweek', 'month', 'day', 'is_weekend', 
            'hour_sin', 'hour_cos', 'lag_24h', 'lag_48h', 'lag_168h']
target_demand = 'demand_mw'
targets_generation = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 
                      'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']

# Split data (using the last 30 days as validation)
train = df_features[:-720]
valid = df_features[-720:]

X_train = train[features]
y_train_demand = train[target_demand]
y_train_gen = train[targets_generation]

X_valid = valid[features]
y_valid_demand = valid[target_demand]
y_valid_gen = valid[targets_generation]

# 3. TRAIN DEMAND FORECAST MODEL (XGBoost)
print("Training Demand Forecasting Model (XGBoost)...")
demand_model = xgb.XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42, 
    early_stopping_rounds=50
)
demand_model.fit(X_train, y_train_demand, 
                 eval_set=[(X_valid, y_valid_demand)],
                 verbose=50)

# 4. TRAIN GENERATION MIX MODEL (Random Forest Multi-Output)
print("Training Generation Mix Model...")
# We use the predicted demand as an additional feature for the generation model!
X_train_gen = X_train.copy()
X_train_gen['predicted_demand'] = demand_model.predict(X_train)

X_valid_gen = X_valid.copy()
X_valid_gen['predicted_demand'] = demand_model.predict(X_valid)

gen_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
gen_model.fit(X_train_gen, y_train_gen)

# 5. EVALUATE
print("\n--- Evaluation ---")
demand_preds = demand_model.predict(X_valid)
print(f"Demand MAE: {mean_absolute_error(y_valid_demand, demand_preds):.2f} MW")

gen_preds = gen_model.predict(X_valid_gen)
print(f"Generation Mix MAE: {mean_absolute_error(y_valid_gen, gen_preds):.2f} MW")

# 6. SAVE MODELS FOR PRODUCTION
print("\nSaving models for GitHub Actions deployment...")
demand_model.save_model('demand_model.json')
joblib.dump(gen_model, 'gen_model.pkl')

print("Training Complete! Download 'demand_model.json' and 'gen_model.pkl' from Kaggle.")
