import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
OUT = Path(__file__).resolve().parent / "improved_data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
n = 30 * 24 * 60
ts = pd.date_range("2026-01-01", periods=n, freq="min")
t = np.arange(n)
hour, dow = ts.hour.to_numpy(), ts.dayofweek.to_numpy()
daily = 1 + .32*np.sin(2*np.pi*(hour-8)/24) + .10*np.sin(4*np.pi*(hour-13)/24)
weekly = np.where(dow < 5, 1., .78)
traffic = np.maximum(80, 1150*daily*weekly*(1+.012*t/1440)*rng.lognormal(0,.10,n))
event = np.full(n, "normal", object)
traffic_events = [
    (3, 10, 45, 2.3, "traffic_spike"),
    (5, 8, 60, 2.0, "traffic_spike"),
    (7, 18, 90, 2.0, "traffic_spike"),
    (9, 15, 70, 2.2, "traffic_spike"),
    (12, 9, 180, 1.7, "gradual_overload"),
    (14, 13, 60, 1.8, "gradual_overload"),
    (16, 19, 45, 2.4, "traffic_spike"),
    (18, 14, 35, 3.1, "traffic_spike"),
    (20, 9, 90, 1.9, "gradual_overload"),
    (22, 17, 50, 2.5, "traffic_spike"),
    (24, 19, 120, 1.8, "gradual_overload"),
    (27, 12, 75, 2.0, "gradual_overload"),
]
for d, h, dur, mult, lab in traffic_events:
    s=d*1440+h*60; e=min(n,s+dur)
    traffic[s:e] *= np.linspace(1,mult,e-s) if lab=="gradual_overload" else mult
    event[s:e]=lab
s=15*1440+11*60; traffic[s:s+50]*=.22; event[s:s+50]="traffic_drop"
capacity=2600.; load=traffic/capacity
cpu=np.clip(24+61*np.minimum(load,1.35)+rng.normal(0,4,n),5,100)
mem=np.clip(36+28*np.minimum(load,1.4)+.12*np.maximum(0,t/1440-10)+rng.normal(0,3,n),15,100)
net=np.maximum(2,traffic*.018+rng.normal(0,4,n))
db=np.clip(18+72*np.minimum(load,1.2)+rng.normal(0,5,n),2,250)
lat=np.clip(180+210*np.maximum(0,load-.45)**1.5+.8*np.maximum(cpu-70,0)+rng.normal(0,28,n),60,None)
err=np.clip(.25+.055*np.maximum(cpu-65,0)+.012*np.maximum(lat-500,0)+rng.normal(0,.25,n),0,35)
incidents = [
    (3*1440 + 10*60 + 25, 3*1440 + 10*60 + 38, "outage"),
    (5*1440 + 8*60 + 30, 5*1440 + 8*60 + 42, "degraded"),
    (7*1440 + 18*60 + 55, 7*1440 + 19*60 + 10, "outage"),
    (9*1440 + 16*60 + 25, 9*1440 + 16*60 + 35, "outage"),
    (12*1440 + 11*60, 12*1440 + 11*60 + 22, "degraded"),
    (14*1440 + 13*60 + 30, 14*1440 + 13*60 + 44, "degraded"),
    (16*1440 + 19*60 + 25, 16*1440 + 19*60 + 36, "outage"),
    (18*1440 + 14*60 + 20, 18*1440 + 14*60 + 32, "outage"),
    (20*1440 + 9*60 + 30, 20*1440 + 9*60 + 42, "degraded"),
    (22*1440 + 17*60 + 20, 22*1440 + 17*60 + 32, "outage"),
    (24*1440 + 20*60, 24*1440 + 20*60 + 18, "degraded"),
    (27*1440 + 12*60 + 30, 27*1440 + 12*60 + 45, "degraded"),
]
status=np.full(n,"healthy",object)
for s,e,lab in incidents:
    cpu[s:e]=np.clip(cpu[s:e]+rng.uniform(8,17,e-s),0,100); mem[s:e]=np.clip(mem[s:e]+rng.uniform(4,10,e-s),0,100)
    lat[s:e]+=rng.uniform(600,1800,e-s); err[s:e]=np.clip(err[s:e]+rng.uniform(8,25,e-s),0,100)
    db[s:e]=np.clip(db[s:e]+rng.uniform(15,45,e-s),0,250); status[s:e]=lab; event[s:e]=lab
for s,e,kind in [(5*1440+180,5*1440+188,"latency_anomaly"),(10*1440+960,10*1440+972,"error_spike"),(21*1440+420,21*1440+430,"cpu_anomaly")]:
    if kind=="latency_anomaly": lat[s:e]+=rng.uniform(900,1600,e-s)
    elif kind=="error_spike": err[s:e]=np.clip(err[s:e]+rng.uniform(5,18,e-s),0,100)
    else: cpu[s:e]=np.clip(cpu[s:e]+rng.uniform(20,35,e-s),0,100)
    event[s:e]=kind
degraded=((cpu>=88)&(lat>=700))|((err>=5)&(cpu>=75))
status[(status=="healthy")&degraded]="degraded"
anom=np.isin(event,["traffic_spike","traffic_drop","latency_anomaly","error_spike","cpu_anomaly","outage","degraded"]).astype(int)
anom[(cpu>96)|(lat>1800)|(err>15)]=1
# Merge incident gaps of at most 3 minutes.
incident = np.isin(status, ["degraded", "outage"])
merged = incident.copy()
i = 0
while i < n:
    if incident[i]:
        i += 1
        continue
    gap_start = i
    while i < n and not incident[i]:
        i += 1
    gap_end = i
    if (
        gap_end - gap_start <= 3
        and gap_start > 0
        and gap_end < n
        and incident[gap_start - 1]
        and incident[gap_end]
    ):
        merged[gap_start:gap_end] = True

# Predict an incident START in the next 30 minutes,
# excluding the current minute.
incident_start = merged & ~np.r_[False, merged[:-1]]
risk = np.zeros(n, dtype=int)
for i in range(n - 30):
    risk[i] = int(incident_start[i + 1:i + 31].any())

valid = np.ones(n, dtype=int)
valid[-30:] = 0
risk[-30:] = 0
df=pd.DataFrame({"timestamp":ts,"requests_per_minute":np.round(traffic).astype(int),"active_users":np.maximum(1,np.round(traffic*rng.uniform(.35,.65,n))).astype(int),"cpu_usage_pct":np.round(cpu,2),"memory_usage_pct":np.round(mem,2),"latency_ms":np.round(lat,2),"error_rate_pct":np.round(err,3),"http_5xx_count":rng.poisson(np.maximum(.2,traffic*err/100)).astype(int),"network_usage_mbps":np.round(net,2),"database_connections":np.round(db).astype(int),"event_type":event,"website_status":status,"anomaly_label":anom,"outage_within_30m":risk,"risk_label_valid":valid})
for lag in [1,5,15,30,60]: df[f"traffic_lag_{lag}m"]=df.requests_per_minute.shift(lag).fillna(df.requests_per_minute)
df["traffic_rolling_mean_5m"]=df.requests_per_minute.shift(1).rolling(5,min_periods=1).mean().fillna(df.requests_per_minute)
df["traffic_rolling_mean_15m"]=df.requests_per_minute.shift(1).rolling(15,min_periods=1).mean().fillna(df.requests_per_minute)
df["traffic_rolling_std_15m"]=df.requests_per_minute.shift(1).rolling(15,min_periods=2).std().fillna(0)
df["traffic_growth_pct_5m"]=(df.requests_per_minute/df.requests_per_minute.shift(5).replace(0,np.nan)-1).mul(100).replace([np.inf,-np.inf],np.nan).fillna(0)
df["cpu_change_5m"]=(df.cpu_usage_pct-df.cpu_usage_pct.shift(5)).fillna(0)
df["latency_change_5m"]=(df.latency_ms-df.latency_ms.shift(5)).fillna(0)
df["error_rate_change_5m"]=(df.error_rate_pct-df.error_rate_pct.shift(5)).fillna(0)
df["hour"]=df.timestamp.dt.hour; df["day_of_week"]=df.timestamp.dt.dayofweek; df["is_weekend"]=(df.day_of_week>=5).astype(int); df["minute_of_day"]=df.hour*60+df.timestamp.dt.minute
df.to_csv(OUT/"website_traffic_outage_synthetic_full.csv",index=False)
fc=["timestamp","requests_per_minute","active_users","hour","day_of_week","is_weekend"]+[f"traffic_lag_{x}m" for x in [1,5,15,30,60]]+["traffic_rolling_mean_5m","traffic_rolling_mean_15m","traffic_rolling_std_15m"]
df[fc].to_csv(OUT/"traffic_forecasting_dataset.csv",index=False)
rc=["timestamp","requests_per_minute","active_users","cpu_usage_pct","memory_usage_pct","latency_ms","error_rate_pct","http_5xx_count","network_usage_mbps","database_connections","traffic_growth_pct_5m","cpu_change_5m","latency_change_5m","error_rate_change_5m","traffic_rolling_mean_5m","traffic_rolling_mean_15m","anomaly_label","website_status","outage_within_30m","risk_label_valid"]
df[rc].to_csv(OUT/"outage_risk_dataset.csv",index=False)
print(f"Generated {len(df):,} rows in {OUT.resolve()}")
