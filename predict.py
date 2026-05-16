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
    
    # We need lags. We pull the actual demand from 24h and 48h ago in our historical dataset.
    # For a perfect forecast we'd iteratively predict, but for simplicity we pull historical lags.
    
    # Create an empty lag column
    future_df['lag_24h'] = np.nan
    future_df['lag_48h'] = np.nan
    
    for i, row in future_df.iterrows():
        # 24h ago from this future date
        target_time_24 = row['datetime'] - timedelta(hours=24)
        target_time_48 = row['datetime'] - timedelta(hours=48)
        
        # Try to find it in the historical data
        hist_24 = last_df[last_df['datetime'] == target_time_24]
        hist_48 = last_df[last_df['datetime'] == target_time_48]
        
        if not hist_24.empty:
            future_df.at[i, 'lag_24h'] = hist_24.iloc[0]['demand_mw']
        if not hist_48.empty:
            future_df.at[i, 'lag_48h'] = hist_48.iloc[0]['demand_mw']
            
    # Fill remaining NaNs with the last known demand just in case
    last_known_demand = last_df['demand_mw'].iloc[-1]
    future_df['lag_24h'].fillna(last_known_demand, inplace=True)
    future_df['lag_48h'].fillna(last_known_demand, inplace=True)
    
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
    features = ['hour', 'dayofweek', 'month', 'day', 'is_weekend', 'lag_24h', 'lag_48h']
    
    if not os.path.exists('demand_model.json') or not os.path.exists('gen_model.pkl'):
        print("Models not found! Please run the Kaggle script and place 'demand_model.json' and 'gen_model.pkl' in this folder.")
        # Create a dummy forecast.json so the Github Action doesn't completely crash the first time
        with open('forecast.json', 'w') as f:
            json.dump({"status": "models missing", "forecast": []}, f)
        return

    print("Loading Models...")
    demand_model = xgb.XGBRegressor()
    demand_model.load_model('demand_model.json')
    gen_model = joblib.load('gen_model.pkl')
    
    print("Predicting Demand...")
    X_future = future_df[features]
    predicted_demand = demand_model.predict(X_future)
    future_df['predicted_demand'] = predicted_demand
    
    print("Predicting Generation Mix...")
    # Add predicted demand to features for the generation model
    X_future_gen = X_future.copy()
    X_future_gen['predicted_demand'] = predicted_demand
    
    gen_preds = gen_model.predict(X_future_gen)
    
    targets_generation = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 
                          'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']
    
    # Combine into a final dictionary
    results = []
    for i, row in future_df.iterrows():
        entry = {
            "datetime": row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
            "demand_mw_forecast": round(float(row['predicted_demand']), 2),
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
