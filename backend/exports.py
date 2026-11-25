import json
def write_geojson(df,path,pollutant):
    features=[]
    for r in df.to_dict(orient="records"):
        features.append({"type":"Feature","geometry":{"type":"Point","coordinates":[r["lon"],r["lat"]]},
                         "properties":{k:(float(v) if hasattr(v,"item") else v) for k,v in r.items() if k not in ("lon","lat")}})
    with open(path,"w") as f: json.dump({"type":"FeatureCollection","name":"DEMO synthetic predictions","features":features},f)
def write_geotiff(df,path,pollutant):
    import rasterio
    from rasterio.transform import from_origin
    import numpy as np
    lats=sorted(df.lat.unique(),reverse=True); lons=sorted(df.lon.unique())
    arr=np.full((len(lats),len(lons)),np.nan,dtype="float32")
    li={v:i for i,v in enumerate(lats)}; lo={v:i for i,v in enumerate(lons)}
    for _,r in df.iterrows(): arr[li[r.lat],lo[r.lon]]=r[pollutant]
    dx=(lons[-1]-lons[0])/(len(lons)-1) if len(lons)>1 else .01
    dy=(lats[0]-lats[-1])/(len(lats)-1) if len(lats)>1 else .01
    with rasterio.open(path,"w",driver="GTiff",height=arr.shape[0],width=arr.shape[1],count=1,dtype="float32",crs="EPSG:4326",transform=from_origin(lons[0]-dx/2,lats[0]+dy/2,dx,dy),nodata=np.nan) as dst:
        dst.write(arr,1); dst.set_band_description(1,f"{pollutant} synthetic demo prediction")
