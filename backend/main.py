from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np, pandas as pd, joblib, json, logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from .pipeline import make_demo_data, build_grid, FEATURES, BOUNDS
from .exports import write_geotiff, write_geojson

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"; MODELS = BASE / "trained_models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
app = FastAPI(title="Satellite Air Quality Intelligence API", version="1.0.0",
              description="Transparent demo downscaling prototype. Demo values are synthetic.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
state = {"model": None, "metrics": None, "predictions": None, "status": "not_trained", "mode":"DEMO"}
logging.basicConfig(level=logging.INFO)

class RunRequest(BaseModel):
    region: str = "Bhubaneswar"
    pollutant: str = Field("pm25", pattern="^(pm25|pm10)$")
    resolution_km: float = Field(1.0, ge=0.5, le=10)
    date: str = "2026-11-25"

@app.get("/api/health")
def health():
    return {"status":"ok","mode":"DEMO","message":"Synthetic demonstration mode is active."}

@app.get("/api/regions")
def regions():
    return [{"name":"Bhubaneswar","state":"Odisha","country":"India","bounds":BOUNDS}]

@app.get("/api/data/sources")
def sources():
    return [{"name":"Demo environmental dataset","status":"available","type":"SYNTHETIC DEMO"},
            {"name":"Sentinel-5P / TROPOMI","status":"not_configured","type":"satellite"},
            {"name":"CPCB / OpenAQ","status":"not_configured","type":"ground observations"},
            {"name":"ERA5 / CAMS","status":"not_configured","type":"meteorology"}]

@app.post("/api/data/acquire")
def acquire():
    df = make_demo_data()
    path = OUT / "demo_training_data.csv"; df.to_csv(path,index=False)
    return {"status":"ok","mode":"DEMO","rows":len(df),"file":str(path.name)}

@app.post("/api/data/preprocess")
def preprocess():
    df = make_demo_data()
    df = df.replace([np.inf,-np.inf],np.nan).dropna()
    return {"status":"ok","rows_after_cleaning":len(df),"features":FEATURES,"mode":"DEMO"}

@app.post("/api/model/train")
def train(payload: RunRequest = RunRequest()):
    try:
        df = make_demo_data()
        target = payload.pollutant
        X, y = df[FEATURES], df[target]
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=42)
        model = RandomForestRegressor(n_estimators=120, min_samples_leaf=2,
                                      max_features=.9, random_state=42, n_jobs=-1)
        model.fit(Xtr,ytr)
        pred = model.predict(Xte)
        metrics = {"r2":float(r2_score(yte,pred)),
                   "rmse":float(np.sqrt(mean_squared_error(yte,pred))),
                   "mae":float(mean_absolute_error(yte,pred)),
                   "mbe":float(np.mean(pred-yte)),
                   "validation":"random holdout of synthetic demo samples; not independent real-world validation"}
        joblib.dump(model, MODELS / f"{target}_demo.joblib")
        state.update(model=model,metrics=metrics,status="trained",pollutant=target)
        return {"status":"trained","pollutant":target,"metrics":metrics,"mode":"DEMO"}
    except Exception as e:
        logging.exception("Training failed")
        raise HTTPException(500,str(e))

@app.get("/api/model/status")
def model_status():
    return {"status":state["status"],"mode":state["mode"],"metrics":state["metrics"]}

@app.post("/api/downscale")
def downscale(payload: RunRequest):
    if state["model"] is None or state.get("pollutant") != payload.pollutant:
        train(payload)
    try:
        grid = build_grid(payload.resolution_km)
        grid[payload.pollutant] = state["model"].predict(grid[FEATURES])
        grid["uncertainty"] = np.std([tree.predict(grid[FEATURES]) for tree in state["model"].estimators_[:40]],axis=0)
        grid["pollutant"] = payload.pollutant
        grid["mode"] = "DEMO"
        path = OUT / "latest_predictions.geojson"
        write_geojson(grid,path,payload.pollutant)
        state["predictions"] = grid
        return {"status":"ok","mode":"DEMO","pollutant":payload.pollutant,
                "resolution_km":payload.resolution_km,"count":len(grid),
                "summary":{"mean":float(grid[payload.pollutant].mean()),
                           "min":float(grid[payload.pollutant].min()),
                           "max":float(grid[payload.pollutant].max()),
                           "mean_uncertainty":float(grid.uncertainty.mean())},
                "features":grid.to_dict(orient="records")}
    except Exception as e:
        logging.exception("Downscaling failed")
        raise HTTPException(500,str(e))

@app.get("/api/map/layers")
def layers():
    return [{"id":"pm25","label":"PM2.5","unit":"µg/m³"},{"id":"pm10","label":"PM10","unit":"µg/m³"}]

@app.get("/api/predictions")
def predictions():
    if state["predictions"] is None: raise HTTPException(404,"Run downscaling first")
    return {"mode":"DEMO","features":state["predictions"].to_dict(orient="records")}

@app.get("/api/metrics")
def metrics():
    if state["metrics"] is None: raise HTTPException(404,"Train a model first")
    return state["metrics"]

@app.get("/api/history")
def history():
    # Synthetic illustrative daily series, explicitly labelled.
    dates=pd.date_range("2026-11-01",periods=25)
    vals=38+9*np.sin(np.arange(25)/4)+np.linspace(0,5,25)
    return {"mode":"DEMO","series":[{"date":d.strftime("%Y-%m-%d"),"pm25":round(float(v),2)} for d,v in zip(dates,vals)]}

@app.get("/api/export/geojson")
def export_geojson():
    p=OUT/"latest_predictions.geojson"
    if not p.exists(): raise HTTPException(404,"Run downscaling first")
    return FileResponse(p,media_type="application/geo+json",filename=p.name)

@app.get("/api/export/geotiff")
def export_tif():
    if state["predictions"] is None: raise HTTPException(404,"Run downscaling first")
    p=OUT/"latest_predictions.tif"
    try: write_geotiff(state["predictions"],p,state.get("pollutant","pm25"))
    except ImportError: raise HTTPException(501,"Install rasterio to enable GeoTIFF export")
    return FileResponse(p,media_type="image/tiff",filename=p.name)
