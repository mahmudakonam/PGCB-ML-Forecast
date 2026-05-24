import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import urllib.request
import os
from datetime import timedelta

# URL to your MAIN repository's pgcb_data.csv
# UPDATE THIS URL once your main repo is public on GitHub!
DATA_URL = "https://raw.githubusercontent.com/mahmudakonam/PGCB-Power-Data-Visualizer/main/pgcb_data.csv"

def download_data():
    print("Downloading latest data from main repository...")
    urllib.request.urlretrieve(DATA_URL, "pgcb_data.csv")

def create_future_features(last_df, hours=24):
    """Generate future rows based on the last known timestamp."""
    last_time = last_df['datetime'].max()
    
    future_dates = [last_time + timedelta(hours=i) for i in range(1, hours + 1)]
    future_df = pd.DataFrame({'datetime': future_dates})
    
    future_df['hour'] = future_df['datetime'].dt.hour
    future_df['dayofweek'] = future_df['datetime'].dt.dayofweek
    future_df['month'] = future_df['datetime'].dt.month
    future_df['day'] = future_df['datetime'].dt.day
    future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical hour encoding
    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
    
    # Create lag columns
    future_df['lag_24h'] = np.nan
    future_df['lag_48h'] = np.nan
    future_df['lag_168h'] = np.nan
    
    for i, row in future_df.iterrows():
        target_time_24 = row['datetime'] - timedelta(hours=24)
        target_time_48 = row['datetime'] - timedelta(hours=48)
        target_time_168 = row['datetime'] - timedelta(hours=168)
        
        hist_24 = last_df[last_df['datetime'] == target_time_24]
        hist_48 = last_df[last_df['datetime'] == target_time_48]
        hist_168 = last_df[last_df['datetime'] == target_time_168]
        
        if not hist_24.empty:
            future_df.at[i, 'lag_24h'] = hist_24.iloc[0]['demand_mw']
        if not hist_48.empty:
            future_df.at[i, 'lag_48h'] = hist_48.iloc[0]['demand_mw']
        if not hist_168.empty:
            future_df.at[i, 'lag_168h'] = hist_168.iloc[0]['demand_mw']
            
    # Fill remaining NaNs with the last known demand
    last_known_demand = last_df['demand_mw'].iloc[-1]
    future_df['lag_24h'] = future_df['lag_24h'].ffill().bfill().fillna(last_known_demand)
    future_df['lag_48h'] = future_df['lag_48h'].ffill().bfill().fillna(last_known_demand)
    future_df['lag_168h'] = future_df['lag_168h'].ffill().bfill().fillna(last_known_demand)
    
    return future_df

def main():
    download_data()
    
    print("Loading data...")
    df = pd.read_csv('pgcb_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Impute missing demand
    df['load_shedding'] = df['load_shedding'].fillna(0)
    df['demand_mw'] = df['demand_mw'].fillna(df['generation_mw'] + df['load_shedding'])
    
    print("Generating future timeline...")
    future_df = create_future_features(df, hours=24)
    features = ['hour', 'dayofweek', 'month', 'day', 'is_weekend', 
                'hour_sin', 'hour_cos', 'lag_24h', 'lag_48h', 'lag_168h']
    
    if not os.path.exists('demand_model.json') or not os.path.exists('demand_rf_model.pkl') or not os.path.exists('gen_model.pkl'):
        print("Models not found! Please run the Kaggle script and place 'demand_model.json', 'demand_rf_model.pkl', and 'gen_model.pkl' in this folder.")
        # Create a dummy forecast.json so the Github Action doesn't completely crash the first time
        with open('forecast.json', 'w') as f:
            json.dump({"status": "models missing", "forecast": []}, f)
        return

    print("Loading Models...")
    demand_model = xgb.XGBRegressor()
    demand_model.load_model('demand_model.json')
    demand_rf_model = joblib.load('demand_rf_model.pkl')
    gen_model = joblib.load('gen_model.pkl')
    
    print("Predicting Demand...")
    X_future = future_df[features]
    predicted_xgb = demand_model.predict(X_future)
    predicted_rf = demand_rf_model.predict(X_future)
    predicted_ensemble = (predicted_xgb + predicted_rf) / 2
    
    future_df['predicted_demand'] = predicted_ensemble
    
    print("Predicting Generation Mix...")
    # Add predicted demand to features for the generation model
    X_future_gen = X_future.copy()
    X_future_gen['predicted_demand'] = predicted_ensemble
    
    gen_preds = gen_model.predict(X_future_gen)
    
    targets_generation = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 
                          'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']
    
    # Combine into a final dictionary
    results = []
    for i, row in future_df.iterrows():
        entry = {
            "datetime": row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
            "predicted_demand_xgb": round(float(predicted_xgb[i]), 2),
            "predicted_demand_rf": round(float(predicted_rf[i]), 2),
            "predicted_demand_ensemble": round(float(predicted_ensemble[i]), 2),
            "predicted_demand_baseline": round(float(row['lag_24h']), 2),
            "generation_mix_forecast": {
                gen_type: round(float(gen_preds[i][j]), 2) for j, gen_type in enumerate(targets_generation)
            }
        }
        results.append(entry)
        
    forecast_data = {
        "generated_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "forecast": results
    }
    
    print("Saving forecast.json...")
    with open('forecast.json', 'w') as f:
        json.dump(forecast_data, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    main()
