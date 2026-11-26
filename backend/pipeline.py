import numpy as np, pandas as pd
FEATURES=["lat","lon","temperature","humidity","wind_speed","rainfall","elevation","ndvi","road_density","population_density","no2_proxy","aod_proxy","day_sin","day_cos"]
# Approximate Bhubaneswar city demonstration bounds; not a monitoring boundary.
BOUNDS={"south":20.15,"west":85.72,"north":20.38,"east":85.92}
def _features(lat,lon,day):
    # Deterministic synthetic spatial fields for a reproducible teaching demo.
    x=(lon-85.82); y=(lat-20.265)
    temperature=30-5*y+1.2*np.sin(x*35)
    humidity=65+12*np.cos(y*28)
    wind=2.0+0.8*np.sin(x*20+y*10)
    rainfall=np.maximum(0,2+3*np.sin(day/8+x*25))
    elevation=25+30*np.sin(x*18)*np.cos(y*15)
    ndvi=np.clip(.45+.25*np.sin(x*30)*np.cos(y*22),.05,.9)
    roads=np.clip(.5+2*np.exp(-((x/.035)**2+(y/.035)**2)),.1,3)
    population=np.clip(5000+18000*np.exp(-((x/.04)**2+(y/.04)**2)),500,30000)
    no2=1.5+1.8*roads+.4*np.sin(day/6)
    aod=.25+.08*roads+.05*np.cos(day/7)
    #new line
    n = len(lat)
    return np.column_stack([
    lat, lon, temperature, humidity, wind, rainfall,
    elevation, ndvi, roads, population, no2, aod,
    np.full(n, np.sin(2*np.pi*day/365)),
    np.full(n, np.cos(2*np.pi*day/365)),])
    #return np.column_stack([lat,lon,temperature,humidity,wind,rainfall,elevation,ndvi,roads,population,no2,aod,np.sin(2*np.pi*day/365),np.cos(2*np.pi*day/365)])
def make_demo_data(n=1800,seed=7):
    rng=np.random.default_rng(seed)
    lat=rng.uniform(BOUNDS["south"],BOUNDS["north"],n); lon=rng.uniform(BOUNDS["west"],BOUNDS["east"],n)
    day=rng.integers(1,366,n); X=_features(lat,lon,day)
    df=pd.DataFrame(X,columns=FEATURES)
    urban=np.exp(-(((lon-85.82)/.04)**2+((lat-20.265)/.04)**2))
    df["pm25"]=np.maximum(3,18+9*urban+2*df.no2_proxy+5*df.aod_proxy-.09*df.wind_speed-.035*df.humidity+rng.normal(0,2,n))
    df["pm10"]=np.maximum(5,32+16*urban+3*df.no2_proxy+8*df.aod_proxy-.12*df.wind_speed+rng.normal(0,3,n))
    return df
def build_grid(resolution_km=1):
    # Approximate degrees-per-km conversion, suitable only for this local demo.
    step=resolution_km/111
    lats=np.arange(BOUNDS["south"],BOUNDS["north"]+step/2,step)
    lons=np.arange(BOUNDS["west"],BOUNDS["east"]+step/2,step)
    lon,lat=np.meshgrid(lons,lats); day=329
    return pd.DataFrame(_features(lat.ravel(),lon.ravel(),day),columns=FEATURES)
