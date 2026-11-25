# Satellite Air Quality Intelligence System

A runnable B.Tech prototype with a FastAPI backend, Random Forest regression, synthetic Bhubaneswar demo data, a React/Leaflet map, evaluation metrics, and GeoJSON/GeoTIFF export.

> **Scientific transparency:** This package runs in `DEMO` mode. Its environmental predictors, target PM2.5/PM10 labels, historical series, and map predictions are synthetic. They are not CPCB readings, satellite observations, or validated pollution estimates. The model scores only describe a holdout split of synthetic data. No AQI is calculated.

## Requirements
- Python 3.10+ (3.11 recommended)
- Node.js 18+
- Internet connection for OpenStreetMap tiles and frontend package installation

## Run locally

### 1. Backend
From the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

### 2. Frontend
Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### 3. Demo workflow
1. Select PM2.5 or PM10.
2. Choose a target grid spacing.
3. Click **Run downscaling**.
4. Inspect the colored map markers and click a marker for its value and ensemble spread.
5. Download GeoJSON. GeoTIFF export is enabled when `rasterio` is installed.

## API
- `GET /api/health`
- `GET /api/regions`
- `GET /api/data/sources`
- `POST /api/data/acquire`
- `POST /api/data/preprocess`
- `POST /api/model/train`
- `GET /api/model/status`
- `POST /api/downscale`
- `GET /api/map/layers`
- `GET /api/predictions`
- `GET /api/metrics`
- `GET /api/history`
- `GET /api/export/geojson`
- `GET /api/export/geotiff`

Example request:

```json
{"region":"Bhubaneswar","pollutant":"pm25","resolution_km":1,"date":"2026-11-25"}
```

## Architecture

```text
React + TypeScript + Leaflet
          |
       REST API
          |
      FastAPI app
       /   |    \
synthetic  ML   export
data      Random Forest   GeoJSON/GeoTIFF
          |
   fine target grid + feature engineering
```

## Connecting real data: important next steps
The current code intentionally does not pretend to retrieve real satellite or station data. To make scientifically defensible real-world predictions:

1. Implement authenticated Sentinel-5P/TROPOMI or CAMS acquisition for the requested time and bounding box, including product-specific QA filtering and units.
2. Obtain co-located CPCB/OpenAQ ground PM2.5/PM10 measurements with station metadata and quality controls.
3. Add ERA5 meteorology and real DEM/NDVI/land-cover/population/road layers, recording source, timestamp, CRS, units, and native resolution.
4. Build spatially and temporally blocked train/test splits. Avoid using the same station, nearby pixels, or time window in both train and test.
5. Train against measured ground PM concentrations. Satellite NO2/AOD are predictors, not direct PM measurements.
6. Validate against independent stations and dates; report uncertainty/calibration and limitations.
7. Configure secrets only in backend environment variables. Never place API credentials in frontend code.

A finer output grid is a model-estimated surface, not a new satellite measurement. The `resolution_km` setting controls target grid spacing only.
