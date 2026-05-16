import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import urllib.request
import os
from datetime import timedelta
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="PGCB ML Forecast API")

# Enable CORS so your GitHub Pages frontend can fetch data from this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you'd restrict this to your github.io domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_URL = "https://raw.githubusercontent.com/mahmudakonam/PGCB-Power-Data-Visualizer/main/pgcb_data.csv"
DATA_FILE = "pgcb_data.csv"

# Global model cache to avoid reloading on every request
MODELS = {}

def load_models():
    if "demand" not in MODELS:
        print("Loading Demand Model...")
        if not os.path.exists('demand_model.json'):
            raise Exception("demand_model.json not found!")
        dm = xgb.XGBRegressor()
        dm.load_model('demand_model.json')
        MODELS["demand"] = dm
        
    if "gen" not in MODELS:
        print("Loading Generation Mix Model...")
        if not os.path.exists('gen_model.pkl'):
            raise Exception("gen_model.pkl not found!")
        MODELS["gen"] = joblib.load('gen_model.pkl')

def download_data():
    # Only download if we don't have it or it's old
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

def prep_historical_data():
    download_data()
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df['load_shedding'] = df['load_shedding'].fillna(0)
    df['demand_mw'] = df['demand_mw'].fillna(df['generation_mw'] + df['load_shedding'])
    return df

@app.get("/")
def health_check():
    return {"status": "online", "message": "PGCB ML API is running"}

@app.get("/predict")
def get_prediction(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    try:
        target_date = pd.to_datetime(date).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    df = prep_historical_data()
    
    # Generate the 24 hours for the target date
    target_datetimes = [pd.to_datetime(f"{target_date} {str(h).zfill(2)}:00:00") for h in range(24)]
    target_df = pd.DataFrame({'datetime': target_datetimes})
    
    target_df['hour'] = target_df['datetime'].dt.hour
    target_df['dayofweek'] = target_df['datetime'].dt.dayofweek
    target_df['month'] = target_df['datetime'].dt.month
    target_df['day'] = target_df['datetime'].dt.day
    target_df['is_weekend'] = target_df['dayofweek'].isin([5, 6]).astype(int)
    
    target_df['hour_sin'] = np.sin(2 * np.pi * target_df['hour'] / 24)
    target_df['hour_cos'] = np.cos(2 * np.pi * target_df['hour'] / 24)
    
    # Retrieve lags from historical df
    target_df['lag_24h'] = np.nan
    target_df['lag_48h'] = np.nan
    target_df['lag_168h'] = np.nan
    target_df['actual_demand'] = np.nan
    
    for i, row in target_df.iterrows():
        t = row['datetime']
        
        hist_actual = df[df['datetime'] == t]
        hist_24 = df[df['datetime'] == t - timedelta(hours=24)]
        hist_48 = df[df['datetime'] == t - timedelta(hours=48)]
        hist_168 = df[df['datetime'] == t - timedelta(hours=168)]
        
        if not hist_actual.empty:
            target_df.at[i, 'actual_demand'] = hist_actual.iloc[0]['demand_mw']
        if not hist_24.empty:
            target_df.at[i, 'lag_24h'] = hist_24.iloc[0]['demand_mw']
        if not hist_48.empty:
            target_df.at[i, 'lag_48h'] = hist_48.iloc[0]['demand_mw']
        if not hist_168.empty:
            target_df.at[i, 'lag_168h'] = hist_168.iloc[0]['demand_mw']
            
    # Forward fill missing lags just in case
    target_df['lag_24h'] = target_df['lag_24h'].ffill().bfill().fillna(0)
    target_df['lag_48h'] = target_df['lag_48h'].ffill().bfill().fillna(0)
    target_df['lag_168h'] = target_df['lag_168h'].ffill().bfill().fillna(0)
    
    features = ['hour', 'dayofweek', 'month', 'day', 'is_weekend', 
                'hour_sin', 'hour_cos', 'lag_24h', 'lag_48h', 'lag_168h']
                
    X = target_df[features]
    
    # Predict Demand
    predicted_demand = MODELS["demand"].predict(X)
    target_df['predicted_demand'] = predicted_demand
    
    # Predict Gen Mix
    X_gen = X.copy()
    X_gen['predicted_demand'] = predicted_demand
    gen_preds = MODELS["gen"].predict(X_gen)
    
    targets_generation = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 
                          'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']
                          
    results = []
    for i, row in target_df.iterrows():
        act_d = row['actual_demand']
        pred_d = round(float(row['predicted_demand']), 2)
        
        entry = {
            "datetime": row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
            "actual_demand_mw": float(act_d) if not pd.isna(act_d) else None,
            "predicted_demand_mw": pred_d,
            "generation_mix_forecast": {
                gen_type: round(float(gen_preds[i][j]), 2) for j, gen_type in enumerate(targets_generation)
            }
        }
        results.append(entry)
        
    return {
        "target_date": str(target_date),
        "generated_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "forecast": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
