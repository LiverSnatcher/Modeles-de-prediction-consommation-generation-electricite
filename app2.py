import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"

from unittest.mock import MagicMock
_fake_nvml = MagicMock()
_fake_nvml.nvmlDeviceGetCount.return_value = 0
sys.modules["pynvml"]     = _fake_nvml
sys.modules["nvidia_smi"] = _fake_nvml

import torch
torch.cuda.is_available = lambda: False

if not getattr(torch.load, "_cpu_patched", False):
    _orig_torch_load = torch.load
    def _cpu_load(f, map_location=None, **kwargs):
        return _orig_torch_load(f, map_location=torch.device("cpu"), **kwargs)
    _cpu_load._cpu_patched = True
    torch.load = _cpu_load

import torchmetrics

if not getattr(torchmetrics.Metric, "_cpu_patched", False):
    def _make_patch():
        _real = torchmetrics.Metric._apply
        def _safe(self, fn, exclude_state="", **kwargs):
            object.__setattr__(self, "_device", torch.device("cpu"))
            return _real(self, fn, exclude_state=exclude_state, **kwargs)
        torchmetrics.Metric._apply = _safe
        torchmetrics.Metric._cpu_patched = True
    _make_patch()
    
_orig_metric_apply = torchmetrics.Metric._apply
def _safe_metric_apply(self, fn, *args, **kwargs):
    self._device = torch.device("cpu")
    if not args and "exclude_state" not in kwargs:
        kwargs["exclude_state"] = frozenset()
    elif args and isinstance(args[0], bool):
        args = (frozenset(),) + args[1:]
    return _orig_metric_apply(self, fn, *args, **kwargs)
torchmetrics.Metric._apply = _safe_metric_apply

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import requests
import holidays
import math
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

from supabase import create_client, Client
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings('ignore')


class ForecastUncertaintyNoise(pl.Callback):
    def __init__(self, max_noise_std=0.05):
        self.max_noise_std = max_noise_std
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Réseau Électrique · France",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌿"
)
load_dotenv()

PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

ZONES_SOLAIRE = {
    "NouvAquitaine": {"lat": 44.83, "lon": -0.57},
    "Occitanie":     {"lat": 43.60, "lon": 1.44},
    "PACA":          {"lat": 43.29, "lon": 5.36},
    "AuvergneRA":    {"lat": 45.76, "lon": 4.83}
}

CAPACITE_EOLIEN  = 25500.0
CAPACITE_SOLAIRE = 21000.0

ZONES_TFT = {
    "HDF_Somme":     {"lat": 49.90, "lon": 2.30},
    "HDF_Aisne":     {"lat": 49.50, "lon": 3.50},
    "GE_Marne":      {"lat": 48.95, "lon": 4.35},
    "GE_Aube":       {"lat": 48.25, "lon": 4.05},
    "Occ_Aude":      {"lat": 43.20, "lon": 2.35},
    "Off_StNazaire": {"lat": 47.15, "lon": -2.65},
    "Off_Fecamp":    {"lat": 49.89, "lon":  0.35},
}

ZONE_WEIGHTS = {
    "HDF_Somme": 0.22, "HDF_Aisne": 0.13, "GE_Marne": 0.15,
    "GE_Aube": 0.15, "Occ_Aude": 0.15, "Off_StNazaire": 0.10, "Off_Fecamp": 0.10
}

CURRENT_CAPACITY = 25500.0


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

.stApp { background-color: #2d4a35; color: #e0edd8; }

.main-title {
    font-family: 'Nunito', sans-serif; font-weight: 800; font-size: 1.6rem;
    color: #e0edd8; margin: 0 0 0.3rem 0;
}
.main-sub {
    font-family: 'Nunito', sans-serif; font-size: 0.78rem; color: #a8c8a0;
    letter-spacing: 0.04em; margin-bottom: 1.5rem;
    border-bottom: 1px solid #4a6e52; padding-bottom: 1rem;
}
.tag {
    display: inline-block; font-family: 'Nunito', sans-serif; font-size: 0.65rem;
    font-weight: 700; background: #3a5c42; border: 1px solid #5a8060;
    color: #8ae8a0; padding: 0.15rem 0.5rem; border-radius: 20px;
    letter-spacing: 0.04em; text-transform: uppercase; margin-right: 0.3rem;
}

[data-testid="stSidebar"] {
    background-color: #223828 !important;
    border-right: 1px solid #3a5c42 !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Nunito', sans-serif !important; font-size: 0.7rem !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #a8c8a0 !important;
}
[data-testid="stSidebar"] .stSlider label {
    font-family: 'Nunito', sans-serif !important; font-size: 0.78rem !important;
    color: #c8e4c0 !important; font-weight: 600 !important;
}

.stButton button {
    font-family: 'Nunito', sans-serif !important; font-size: 0.78rem !important;
    font-weight: 700 !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important; border-radius: 20px !important;
    transition: all 0.2s !important;
}
.stButton button[kind="primary"] {
    background: #4db86a !important; border: none !important; color: #152018 !important;
}
.stButton button[kind="primary"]:hover { background: #70d488 !important; color: #152018 !important; }
.stButton button[kind="secondary"] {
    background: transparent !important; border: 1px solid #5a8060 !important; color: #a8c8a0 !important;
}
.stButton button[kind="secondary"]:hover { border-color: #a8c8a0 !important; color: #c8e4c0 !important; }

[data-testid="stMetric"] {
    background: #3a5c42 !important; border: 1px solid #4a6e52 !important;
    border-left: 4px solid #6ed98a !important; border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Nunito', sans-serif !important; font-size: 0.7rem !important;
    font-weight: 700 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: #a8c8a0 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Nunito', sans-serif !important; font-size: 1.4rem !important;
    font-weight: 800 !important; color: #6ed98a !important;
}
[data-testid="column"]:nth-child(1) [data-testid="stMetric"] { border-left-color: #d4785a !important; }
[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { border-left-color: #6ed98a !important; }
[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { border-left-color: #e8c84a !important; }
[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { border-left-color: #90b8d8 !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 2px solid #4a6e52 !important; gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Nunito', sans-serif !important; font-size: 0.75rem !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important; color: #7aaa80 !important;
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; padding: 0.6rem 1.2rem !important;
    margin-bottom: -2px !important; transition: all 0.15s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: #c8e4c0 !important; }
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e0edd8 !important; border-bottom-color: #6ed98a !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    background: transparent !important; padding-top: 1rem !important;
}

hr { border-color: #4a6e52 !important; margin: 1.2rem 0 !important; }

[data-testid="stSelectbox"] label {
    font-family: 'Nunito', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 600 !important; color: #a8c8a0 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}

.stAlert {
    background: #3a5c42 !important; border: 1px solid #5a8060 !important;
    border-radius: 8px !important; font-family: 'Nunito', sans-serif !important;
    font-size: 0.8rem !important; color: #c8e4c0 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #223828; }
::-webkit-scrollbar-thumb { background: #4a6e52; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #6ed98a; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SUPABASE — ROLLING FORECAST
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()


def load_rolling_forecast():
    try:
        response = supabase.table("rolling_forecast").select("*").eq("id", 1).execute()
        if not response.data or response.data[0].get("last_updated") is None:
            return None
        data   = response.data[0]
        run_dt = datetime.fromisoformat(data["last_updated"])
        res    = {"run_dt": run_dt}
        for key in ["conso", "eolien", "solaire", "residuelle"]:
            s = pd.Series(data[key]) if data.get(key) else pd.Series()
            if not s.empty:
                s.index = pd.to_datetime(s.index)
            res[key] = s.sort_index()
        return res
    except Exception as e:
        print(f"Erreur Supabase Load: {e}")
        return None


def save_rolling_forecast(run_dt, df_conso, df_eol, df_sol, df_res):
    try:
        def series_to_dict(s):
            return {k.isoformat(): v for k, v in s.round(0).items()}
        payload = {
            "id": 1,
            "last_updated": run_dt.isoformat(),
            "conso":      series_to_dict(df_conso["Prédiction (MW)"]),
            "eolien":     series_to_dict(df_eol["Prédiction (MW)"]),
            "solaire":    series_to_dict(df_sol["Prédiction (MW)"]),
            "residuelle": series_to_dict(df_res["Prédiction (MW)"]),
        }
        supabase.table("rolling_forecast").upsert(payload).execute()
    except Exception as e:
        print(f"Erreur Supabase Save: {e}")


def stitch_forecasts(old_series, new_series, split_dt):
    if old_series is None or old_series.empty:
        return new_series
    past   = old_series[old_series.index <= split_dt]
    future = new_series[new_series.index > split_dt]
    return pd.concat([past, future])


# ══════════════════════════════════════════════════════════════════════════════
# 3. DONNÉES & MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    model_conso   = joblib.load("model_consommation.pkl")
    model_solaire = joblib.load("model_solaire_v2.pkl")
    model_eolien_tft = TemporalFusionTransformer.load_from_checkpoint(
        "model_eolien_tft.ckpt",
        map_location=torch.device("cpu"),
        strict=False,
    )
    model_eolien_tft.eval()
    return model_conso, model_solaire, model_eolien_tft


@st.cache_data(ttl=900)
def get_eco2mix_realtime():
    all_records = []
    offset = 0
    limit  = 100
    total  = None
    while total is None or offset < total:
        r = requests.get(
            "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/records",
            params={"limit": limit, "offset": offset, "order_by": "date_heure desc"}
        )
        if not r.ok:
            raise RuntimeError("Erreur ODRE")
        data = r.json()
        if total is None:
            total = min(data["total_count"], 13 * 24 * 4)
        all_records.extend(data["results"])
        offset += limit
        if offset >= total:
            break
    df = pd.DataFrame(all_records)
    df["date_heure"] = (
        pd.to_datetime(df["date_heure"], utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None)
    )
    return df.set_index("date_heure").sort_index()


@st.cache_data(ttl=3600)
def get_weather_all(start_dt, horizon_days):
    fetch_days   = min(horizon_days + 1, 16)
    end_dt       = start_dt + timedelta(days=fetch_days)
    api_start_dt = start_dt - timedelta(days=3)
    all_series   = {}

    # Température nationale (conso)
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": 48.85, "longitude": 2.35,
        "start_date": api_start_dt.strftime("%Y-%m-%d"),
        "end_date":   end_dt.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m", "timezone": "Europe/Paris",
    })
    times = pd.to_datetime(r.json()["hourly"]["time"])
    all_series["temp_national"] = pd.Series(r.json()["hourly"]["temperature_2m"], index=times)

    # Solaire — 4 zones
    for region, coords in ZONES_SOLAIRE.items():
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": api_start_dt.strftime("%Y-%m-%d"),
            "end_date":   end_dt.strftime("%Y-%m-%d"),
            "hourly": "shortwave_radiation,direct_radiation,cloud_cover,temperature_2m",
            "timezone": "Europe/Paris",
        })
        data = r.json()["hourly"]
        all_series[f"rad_short_{region}"] = pd.Series(data["shortwave_radiation"], index=times)
        all_series[f"rad_dir_{region}"]   = pd.Series(data["direct_radiation"],    index=times)
        all_series[f"cloud_{region}"]     = pd.Series(data["cloud_cover"],          index=times)
        all_series[f"temp_{region}"]      = pd.Series(data["temperature_2m"],       index=times)

    # Éolien TFT — 7 zones
    for region, coords in ZONES_TFT.items():
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": api_start_dt.strftime("%Y-%m-%d"),
            "end_date":   end_dt.strftime("%Y-%m-%d"),
            "hourly": "wind_speed_100m,wind_direction_100m,temperature_2m,surface_pressure",
            "timezone": "Europe/Paris",
        })
        data = r.json()["hourly"]
        all_series[f"wind_speed_{region}"] = pd.Series(data["wind_speed_100m"],      index=times)
        all_series[f"wind_dir_{region}"]   = pd.Series(data["wind_direction_100m"],  index=times)
        all_series[f"temp_{region}"]       = pd.Series(data["temperature_2m"],       index=times)
        all_series[f"pres_{region}"]       = pd.Series(data["surface_pressure"],     index=times)

    df_weather = pd.DataFrame(all_series)
    return df_weather.resample("30min").ffill().bfill()


# ══════════════════════════════════════════════════════════════════════════════
# 4. UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════
def safe_last(series):
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else 0.0


def trim_flat_tail(series, threshold=20):
    s = series.dropna()
    if len(s) < 3:
        return s
    for i in range(len(s) - 1, 1, -1):
        if abs(s.iloc[i] - s.iloc[i - 1]) > threshold:
            return s.iloc[: i + 1]
    return s


def cyclic(val, period):
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def bridge_pred(real: pd.Series, pred: pd.Series) -> pd.Series:
    if real.empty or pred.empty:
        return pred
    bridge = pd.Series([real.iloc[-1]], index=[real.index[-1]])
    return pd.concat([bridge, pred])


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTLY
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_BASE = dict(
    height=430,
    hovermode="x unified",
    margin=dict(l=0, r=0, t=36, b=0),
    plot_bgcolor="rgba(42,75,52,0.5)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Nunito, sans-serif", size=11, color="#a8c8a0"),
    xaxis=dict(
        gridcolor="#3a5c42", linecolor="#4a6e52",
        tickfont=dict(size=10, color="#a8c8a0"),
        showspikes=True, spikecolor="#6ed98a", spikethickness=1, spikedash="dot",
    ),
    yaxis=dict(
        gridcolor="#3a5c42", linecolor="#4a6e52",
        tickfont=dict(size=10, color="#a8c8a0"),
        ticksuffix=" MW",
    ),
    legend=dict(
        bgcolor="rgba(34,56,40,0.95)", bordercolor="#4a6e52", borderwidth=1,
        font=dict(size=10, color="#e0edd8"),
    ),
    hoverlabel=dict(
        bgcolor="#223828", bordercolor="#4a6e52",
        font=dict(family="Nunito, sans-serif", size=11, color="#e0edd8"),
    ),
)


def make_fig(real, pred, color_real, color_pred, title, split_dt, compare_series=None):
    fig = go.Figure()

    fig.add_vline(x=split_dt, line=dict(color="#a8c8a0", width=1, dash="dot"))
    fig.add_vrect(
        x0=split_dt, x1=pred.index.max(),
        fillcolor="rgba(255,255,255,0.02)", layer="below", line_width=0,
    )
    fig.add_trace(go.Scatter(
        x=real.index, y=real.values,
        name="Historique RTE",
        line=dict(color=color_real, width=2.2),
        fill="tozeroy", fillcolor=hex_to_rgba(color_real, 0.12),
        hovertemplate="%{y:,.0f} MW<extra>Historique</extra>",
    ))
    if compare_series is not None:
        fig.add_trace(go.Scatter(
            x=compare_series.index, y=compare_series.values,
            name="Run précédent",
            line=dict(color="#f0e8a0", width=1.8, dash="dot"),
            opacity=0.85,
            hovertemplate="%{y:,.0f} MW<extra>Run précédent</extra>",
        ))
    pred_bridged = bridge_pred(real, pred)
    fig.add_trace(go.Scatter(
        x=pred_bridged.index, y=pred_bridged.values,
        name="Prévision IA",
        line=dict(color=color_pred, width=2.4, dash="dash"),
        hovertemplate="%{y:,.0f} MW<extra>Prévision</extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=title, font=dict(family="Nunito, sans-serif", size=13, color="#e0edd8"), x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class="main-title">🌿 Réseau Électrique · France</p>
<p class="main-sub">
    <span class="tag">XGBoost</span> Consommation · Solaire &nbsp;·&nbsp;
    <span class="tag">TFT · 1M params</span> Éolien &nbsp;·&nbsp;
    <span class="tag">RTE</span> éCO2mix temps réel
</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Paramètres")
    horizon_jours = st.slider("Horizon (jours)", min_value=1, max_value=30, value=4)
    run_btn = st.button("▶ Lancer la prévision", type="primary", use_container_width=True)
    st.divider()
    st.markdown("""
    <div style="font-family:'Nunito',sans-serif;font-size:0.75rem;color:#a8c8a0;line-height:2;">
    MODÈLE ÉOLIEN<br>
    <span style="color:#6ed98a;font-weight:700;">TFT · 1 024 667 params</span><br>
    hidden=64 · lr=1e-3<br>
    7 zones météo · 100m agl<br>
    Horizon +24h · QuantileLoss
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Acquisition et inférence en cours..."):

        df_eco2mix    = get_eco2mix_realtime()
        df_eco2mix_30 = df_eco2mix.resample("30min", label="left", closed="right").mean(numeric_only=True)
        last_known_dt = df_eco2mix_30[["consommation"]].dropna().index.max()

        old_run = load_rolling_forecast()

        # Cherche dans la DB de Supabase si la dernière prédiction date d'il y a moins d'une heure
        if old_run and (last_known_dt - old_run["run_dt"]).total_seconds() <= 3600:
            st.success("⚡ Prévision instantanée chargée depuis Supabase (générée il y a moins d'1h)")
            df_conso      = old_run["conso"].to_frame("Prédiction (MW)")
            df_eol        = old_run["eolien"].to_frame("Prédiction (MW)")
            df_sol        = old_run["solaire"].to_frame("Prédiction (MW)")
            df_residuelle = old_run["residuelle"].to_frame("Prédiction (MW)")

        # Si la dernière prédiction date d'il y a plus d'une heure rafraîchi une nouvelle prédiction
        else:
            model_conso, model_solaire, model_tft = load_models()
            start_date   = last_known_dt.date()
            df_weather   = get_weather_all(start_date, horizon_jours)

            df_lags = (
                df_eco2mix_30[["consommation"]]
                .rename(columns={"consommation": "value"})
                .dropna()
            )
            df_renew_lags = (
                df_eco2mix_30[["eolien", "solaire"]]
                .rename(columns={"eolien": "Eolien", "solaire": "Solaire"})
                .dropna()
                .iloc[:-1]
            )

            fr_holidays  = holidays.France()
            df_working_c = df_lags.copy()
            df_working_s = df_renew_lags[["Solaire"]].rename(columns={"Solaire": "value"}).copy()
            df_working_s.index  = pd.to_datetime(df_working_s.index)
            df_working_s["value"] = (df_working_s["value"] / CAPACITE_SOLAIRE).clip(lower=0, upper=1)

            steps_totaux    = horizon_jours * 24 * 2
            pred_conso_list = []
            pred_sol_list   = []

            # Boucle Conso + Solaire
            for i in range(1, steps_totaux + 1):
                target_dt   = last_known_dt + timedelta(minutes=30 * i)
                temperature = float(df_weather["temp_national"].asof(target_dt))

                past_c = df_working_c[
                    df_working_c.index <= target_dt - timedelta(minutes=30)
                ]["value"].dropna()

                feat_c = pd.DataFrame([{
                    "heure":         target_dt.hour,
                    "jour_semaine":  target_dt.weekday(),
                    "mois":          target_dt.month,
                    "est_ferie":     int(target_dt.date() in fr_holidays),
                    "temperature":   temperature,
                    "lag_48":        safe_last(df_working_c[df_working_c.index <= target_dt - timedelta(hours=24)]["value"]),
                    "lag_96":        safe_last(df_working_c[df_working_c.index <= target_dt - timedelta(hours=48)]["value"]),
                    "lag_336":       safe_last(df_working_c[df_working_c.index <= target_dt - timedelta(hours=168)]["value"]),
                    "rolling_mean_48":  past_c.tail(48).mean()  if not past_c.empty else 0.0,
                    "rolling_mean_336": past_c.tail(336).mean() if not past_c.empty else 0.0,
                }])
                pred_c = model_conso.predict(feat_c.astype(float))[0]
                pred_conso_list.append({"Date": target_dt, "Prédiction (MW)": pred_c})
                df_working_c.loc[target_dt] = pred_c

                # Solaire — load factor
                hs, hc = cyclic(target_dt.hour + target_dt.minute / 60.0, 24)
                ms, mc = cyclic(target_dt.month, 12)
                js, jc = cyclic(target_dt.timetuple().tm_yday, 365)

                past_s_48 = df_working_s[
                    df_working_s.index <= target_dt - timedelta(hours=24)
                ]["value"].dropna()

                def sol_lag(h):
                    return safe_last(
                        df_working_s[df_working_s.index <= target_dt - timedelta(hours=h)]["value"]
                    )

                feat_s_dict = {
                    "heure_sin": hs, "heure_cos": hc,
                    "mois_sin":  ms, "mois_cos":  mc,
                    "jour_sin":  js, "jour_cos":  jc,
                    "lag_48":           sol_lag(24),
                    "rolling_mean_48":  past_s_48.tail(48).mean() if len(past_s_48) >= 1 else 0.0,
                }
                for region in ZONES_SOLAIRE:
                    feat_s_dict[f"rad_short_{region}"] = float(df_weather[f"rad_short_{region}"].asof(target_dt))
                    feat_s_dict[f"rad_dir_{region}"]   = float(df_weather[f"rad_dir_{region}"].asof(target_dt))
                    feat_s_dict[f"cloud_{region}"]     = float(df_weather[f"cloud_{region}"].asof(target_dt))
                    feat_s_dict[f"temp_{region}"]      = float(df_weather[f"temp_{region}"].asof(target_dt))

                feature_cols_sol = [
                    "heure_sin", "heure_cos", "mois_sin", "mois_cos",
                    "jour_sin",  "jour_cos",  "lag_48",   "rolling_mean_48",
                ]
                for region in ["NouvAquitaine", "Occitanie", "PACA", "AuvergneRA"]:
                    feature_cols_sol.extend([
                        f"rad_short_{region}", f"rad_dir_{region}",
                        f"cloud_{region}",     f"temp_{region}",
                    ])

                feat_s    = pd.DataFrame([feat_s_dict])[feature_cols_sol].fillna(0.0)
                pred_s_lf = max(0.0, min(1.0, float(model_solaire.predict(feat_s.astype(float))[0])))
                pred_s_mw = pred_s_lf * CAPACITE_SOLAIRE

                pred_sol_list.append({"Date": target_dt, "Prédiction (MW)": pred_s_mw})
                df_working_s.loc[target_dt] = pred_s_lf

            # ── TFT Éolien ────────────────────────────────────────────────────
            pred_eol_list = []
            df_eol_hist   = df_renew_lags[["Eolien"]].copy()
            df_eol_hist["eolien_lf"] = df_eol_hist["Eolien"] / CURRENT_CAPACITY

            tft_chunk_steps = 48
            total_chunks    = math.ceil(steps_totaux / tft_chunk_steps)
            current_dt      = last_known_dt

            for chunk in range(total_chunks):
                steps_in_chunk = min(tft_chunk_steps, steps_totaux - chunk * tft_chunk_steps)
                start_hist_dt  = current_dt - timedelta(hours=48)
                end_pred_dt    = current_dt + timedelta(hours=24)
                chunk_idx      = pd.date_range(
                    start=start_hist_dt, end=end_pred_dt, freq="30min", inclusive="right"
                )
                df_tft = pd.DataFrame(index=chunk_idx)

                df_tft["group"]              = "France"
                df_tft["time_idx"]           = np.arange(len(df_tft))
                df_tft["hour"]               = df_tft.index.hour.astype(str)
                df_tft["month"]              = df_tft.index.month.astype(str)
                df_tft["installed_capacity"] = CURRENT_CAPACITY

                df_tft = df_tft.join(df_eol_hist[["eolien_lf"]], how="left")
                df_tft["eolien_lf"] = df_tft["eolien_lf"].ffill().fillna(0.0)

                zone_pcs = {}
                for region in ZONES_TFT:
                    ws = df_weather[f"wind_speed_{region}"].reindex(df_tft.index).ffill().bfill().fillna(0.0).values
                    wd = df_weather[f"wind_dir_{region}"].reindex(df_tft.index).ffill().bfill().fillna(0.0).values
                    tp = df_weather[f"temp_{region}"].reindex(df_tft.index).ffill().bfill().fillna(15.0).values
                    pr = df_weather[f"pres_{region}"].reindex(df_tft.index).ffill().bfill().fillna(1013.0).values

                    df_tft[f"wind_speed_{region}"] = ws
                    df_tft[f"wind_dir_{region}"]   = wd
                    df_tft[f"temp_{region}"]        = tp
                    df_tft[f"pres_{region}"]        = pr

                    pc = np.zeros_like(ws)
                    mask_cubic = (ws >= 3.0)  & (ws < 13.0)
                    mask_rated = (ws >= 13.0) & (ws < 25.0)
                    pc[mask_cubic] = ((ws[mask_cubic] - 3.0) / 10.0) ** 3
                    pc[mask_rated] = 1.0
                    air_density = (pr * 100.0) / (287.05 * (tp + 273.15))

                    df_tft[f"pc_{region}"]            = pc
                    df_tft[f"power_density_{region}"] = pc * (air_density / 1.225)
                    zone_pcs[region] = pc

                df_tft["pc_national"] = sum(zone_pcs[r] * ZONE_WEIGHTS[r] for r in ZONES_TFT)
                weather_cols = [
                    c for c in df_tft.columns
                    if any(x in c for x in ["wind_speed", "temp", "pres", "wind_dir", "pc_", "power_density"])
                ]
                df_tft[weather_cols] = df_tft[weather_cols].astype(np.float32)

                pred_tensor = model_tft.predict(df_tft)
                pred_lf     = pred_tensor[0].cpu().numpy()

                for step in range(steps_in_chunk):
                    target_dt_eol = current_dt + timedelta(minutes=30 * (step + 1))
                    val_mw = max(0.0, float(pred_lf[step]) * CURRENT_CAPACITY)
                    pred_eol_list.append({"Date": target_dt_eol, "Prédiction (MW)": val_mw})
                    df_eol_hist.loc[target_dt_eol, "eolien_lf"] = pred_lf[step]

                current_dt += timedelta(minutes=30 * steps_in_chunk)

            # Assemblage & stitch
            df_conso_new = pd.DataFrame(pred_conso_list).set_index("Date")["Prédiction (MW)"]
            df_eol_new   = pd.DataFrame(pred_eol_list).set_index("Date")["Prédiction (MW)"]
            df_sol_new   = pd.DataFrame(pred_sol_list).set_index("Date")["Prédiction (MW)"]

            if old_run:
                df_conso = stitch_forecasts(old_run["conso"],    df_conso_new, last_known_dt).to_frame("Prédiction (MW)")
                df_eol   = stitch_forecasts(old_run["eolien"],   df_eol_new,   last_known_dt).to_frame("Prédiction (MW)")
                df_sol   = stitch_forecasts(old_run["solaire"],  df_sol_new,   last_known_dt).to_frame("Prédiction (MW)")
            else:
                df_conso = df_conso_new.to_frame("Prédiction (MW)")
                df_eol   = df_eol_new.to_frame("Prédiction (MW)")
                df_sol   = df_sol_new.to_frame("Prédiction (MW)")

            df_residuelle = pd.DataFrame({
                "Prédiction (MW)": (
                    df_conso["Prédiction (MW)"]
                    - df_eol["Prédiction (MW)"]
                    - df_sol["Prédiction (MW)"]
                ).clip(lower=0)
            })

            save_rolling_forecast(last_known_dt, df_conso, df_eol, df_sol, df_residuelle)
            st.success("✅ Modèles exécutés avec succès. Timeline sauvegardée dans Supabase.")

        # Display (both branches converge here)
        n_hist = max(192 * 2, horizon_jours * 4 * 24)

        recent_conso = df_eco2mix["consommation"].dropna().tail(n_hist).rename("Réel (MW)")
        recent_eol   = trim_flat_tail(df_eco2mix["eolien"].dropna()).tail(n_hist).rename("Réel (MW)")
        recent_sol   = trim_flat_tail(df_eco2mix["solaire"].dropna()).tail(n_hist).rename("Réel (MW)")
        recent_residuelle = (
            recent_conso - recent_eol.fillna(0) - recent_sol.fillna(0)
        ).clip(lower=0).dropna().rename("Réel (MW)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Consommation",      f"{int(recent_conso.iloc[-1]):,} MW")
        col2.metric("Éolien",            f"{int(recent_eol.iloc[-1]):,} MW")
        col3.metric("Solaire",           f"{int(recent_sol.iloc[-1]):,} MW")
        col4.metric("Charge Résiduelle", f"{int(recent_residuelle.iloc[-1]):,} MW")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡  Consommation", "💨  Éolien", "☀  Solaire", "🌿  Charge Résiduelle"
        ])

        with tab1:
            st.plotly_chart(make_fig(
                recent_conso, df_conso["Prédiction (MW)"],
                "#e0edd8", "#e8c84a",
                "Consommation nationale",
                last_known_dt,
            ), use_container_width=True)

        with tab2:
            st.plotly_chart(make_fig(
                recent_eol, df_eol["Prédiction (MW)"],
                "#6ed98a", "#e8c84a",
                "Production éolienne — PyTorch TFT · 1M params",
                last_known_dt,
            ), use_container_width=True)

        with tab3:
            st.plotly_chart(make_fig(
                recent_sol, df_sol["Prédiction (MW)"],
                "#e8c84a", "#6ed98a",
                "Production solaire",
                last_known_dt,
            ), use_container_width=True)

        with tab4:
            st.plotly_chart(make_fig(
                recent_residuelle, df_residuelle["Prédiction (MW)"],
                "#90b8d8", "#e8c84a",
                "Charge résiduelle",
                last_known_dt,
            ), use_container_width=True)
