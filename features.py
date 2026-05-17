"""
features.py — Feature engineering, model prediction, sample data generation
"""
import random
import os
import joblib
import numpy as np
import pandas as pd
import keras
import streamlit as st

from config import (
    CLUSTER_NAMES, COUNTRY_TO_REGION,
    AGE_BINS, AGE_LABELS,
    WATCH_BINS, WATCH_LBL,
    ACT_BINS, ACT_LBL,
    ENGAGEMENT_MEDIAN_REF,
    SUBSCRIPTIONS, GENRES, REGIONS,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ── Model loading ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading models…")
def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))

    inputs  = keras.Input(shape=(37,), name="input_layer")
    x       = keras.layers.Dense(16, activation="relu",   name="dense")(inputs)
    x       = keras.layers.Dense(8,  activation="relu",   name="dense_1")(x)
    outputs = keras.layers.Dense(4,  activation="linear", name="latent_space")(x)
    encoder = keras.Model(inputs, outputs)
    encoder.load_weights(os.path.join(MODELS_DIR, "encoder_model.keras"))
    return scaler, kmeans, encoder


# ── Feature engineering ────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, use_qcut: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Normalize country → use only the country name for OHE (region is separate info)
    df["Engagement_Score"] = df["Watch_Time_Hours"] / (df["Days_Since_Last_Login"] + 1)
    df["Age_Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)

    if use_qcut and len(df) >= 4:
        df["Watch_Level"]    = pd.qcut(df["Watch_Time_Hours"],      q=4, labels=WATCH_LBL, duplicates="drop")
        df["Activity_Level"] = pd.qcut(df["Days_Since_Last_Login"], q=3, labels=ACT_LBL,   duplicates="drop")
        median_eng = df["Engagement_Score"].median()
    else:
        df["Watch_Level"]    = pd.cut(df["Watch_Time_Hours"],      bins=WATCH_BINS, labels=WATCH_LBL)
        df["Activity_Level"] = pd.cut(df["Days_Since_Last_Login"], bins=ACT_BINS,   labels=ACT_LBL)
        median_eng = ENGAGEMENT_MEDIAN_REF

    df["Low_Engagement_Premium"] = (
        (df["Subscription_Type"] == "Premium") & (df["Engagement_Score"] < median_eng)
    ).astype(int)

    for col, fallback in [("Watch_Level", "Medium"), ("Activity_Level", "Medium"), ("Age_Group", "Adult")]:
        df[col] = df[col].astype(str).replace("nan", fallback)

    return df


def encode_for_model(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(
        df,
        columns=["Country", "Subscription_Type", "Favorite_Genre", "Age_Group", "Watch_Level", "Activity_Level"],
        drop_first=False,
    )
    for col in feature_order:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[feature_order].astype(float)


def predict_clusters(df_raw: pd.DataFrame, use_qcut: bool, scaler, kmeans, encoder) -> pd.DataFrame:
    feature_order = list(scaler.feature_names_in_)

    eng      = engineer_features(df_raw, use_qcut=use_qcut)
    X        = encode_for_model(eng, feature_order)
    Xs       = scaler.transform(X)
    latent   = encoder.predict(Xs, verbose=0)
    clusters = kmeans.predict(latent)
    distances = kmeans.transform(latent)

    out = df_raw.copy()
    out["Cluster"]            = clusters
    out["Cluster_Label"]      = [CLUSTER_NAMES.get(int(c), f"Cluster {c}") for c in clusters]
    out["Engagement_Score"]   = eng["Engagement_Score"].values
    out["Cluster_Confidence"] = (1 / (1 + distances.min(axis=1)) * 100).round(1)

    # Attach region
    out["Region"] = out["Country"].map(COUNTRY_TO_REGION).fillna("Other")

    return out


# ── Sample data generator ──────────────────────────────────────────
# Pull a flat list of real country names (no "Other (...)" entries)
_REAL_COUNTRIES = [c for region_list in REGIONS.values() for c in region_list]


def generate_sample_data(n: int = 500) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    rows = []
    for _ in range(n):
        seg = random.choices(["heavy", "inactive", "casual"], weights=[0.35, 0.30, 0.35])[0]
        if seg == "heavy":
            watch = round(np.random.normal(8, 1.5), 1)
            days  = int(np.random.exponential(3))
        elif seg == "inactive":
            watch = round(np.random.normal(2, 1), 1)
            days  = int(np.random.normal(60, 20))
        else:
            watch = round(np.random.normal(4.5, 1.5), 1)
            days  = int(np.random.normal(15, 8))

        rows.append({
            "Age":                   max(5, min(80, int(np.random.normal(35, 12)))),
            "Country":               random.choice(_REAL_COUNTRIES),
            "Subscription_Type":     random.choice(SUBSCRIPTIONS),
            "Favorite_Genre":        random.choice(GENRES),
            "Watch_Time_Hours":      max(0.1, watch),
            "Days_Since_Last_Login": max(0, days),
        })

    return pd.DataFrame(rows)
