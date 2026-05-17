"""
ui_single.py — Single User analysis page
"""
import pandas as pd
import streamlit as st

from config import (
    CLUSTER_ICONS, CLUSTER_BADGE, CLUSTER_DESC, CLUSTER_RECOS,
    COUNTRY_TO_REGION, REGION_CONTEXT, REGION_PALETTE,
    SUBSCRIPTIONS, GENRES, ALL_COUNTRIES,
)
from features import predict_clusters
from gemini_utils import render_chat


def render(scaler, kmeans, encoder, gemini_api_key: str):
    st.markdown("# 👤 Single User Analysis")
    st.markdown("Enter a user profile and get their segment + AI insights.")

    with st.form("user_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age     = st.number_input("Age", 5, 90, 30)
            country = st.selectbox("Country", ALL_COUNTRIES, index=ALL_COUNTRIES.index("USA"))
        with c2:
            sub   = st.selectbox("Subscription", SUBSCRIPTIONS, index=1)
            genre = st.selectbox("Favorite Genre", GENRES, index=GENRES.index("Drama"))
        with c3:
            watch = st.number_input("Avg Watch Hours/Day", 0.0, 24.0, 4.0, 0.5)
            days  = st.number_input("Days Since Last Login", 0, 2000, 5)
        submitted = st.form_submit_button("🔮 Analyze User", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Age": age, "Country": country, "Subscription_Type": sub,
            "Favorite_Genre": genre, "Watch_Time_Hours": watch,
            "Days_Since_Last_Login": days,
        }])
        res = predict_clusters(row, use_qcut=False, scaler=scaler, kmeans=kmeans, encoder=encoder).iloc[0]
        st.session_state["su_result"] = {
            "label": res["Cluster_Label"],
            "eng":   float(res["Engagement_Score"]),
            "conf":  float(res["Cluster_Confidence"]),
            "age": age, "country": country, "sub": sub,
            "genre": genre, "watch": watch, "days": days,
        }
        st.session_state.pop("su_chat", None)

    if "su_result" not in st.session_state:
        return

    r     = st.session_state["su_result"]
    label = r["label"]
    eng   = r["eng"]
    conf  = r["conf"]
    icon  = CLUSTER_ICONS[label]
    badge = CLUSTER_BADGE[label]

    # ── Segment result box ─────────────────────────────────────────
    st.markdown(f"""
    <div class="result-box">
        <div style="margin-bottom:12px">
            <span class="cluster-badge {badge}">{icon} {label}</span>
        </div>
        <h2>{label}</h2>
        <p>{CLUSTER_DESC[label]}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    churn = (
        "🔴 High"   if label == "Inactive Users" else
        "🟢 Low"    if label == "Active Heavy Watchers" else
        "🟡 Medium"
    )
    m1.markdown(f'<div class="metric-card"><div class="label">Engagement Score</div><div class="value">{eng:.2f}</div><div class="sub">watch per login day</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="label">Churn Risk</div><div class="value" style="font-size:22px">{churn}</div><div class="sub">based on segment</div></div>', unsafe_allow_html=True)

    # ── Country / Region Insight ───────────────────────────────────
    country = r["country"]
    region  = COUNTRY_TO_REGION.get(country, "Unknown")
    region_note = REGION_CONTEXT.get(region, "")
    region_color = REGION_PALETTE.get(region, "#888")

    st.markdown("<div class='section-title'>🌍 Country & Region Insight</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-box" style="border-left-color:{region_color}; margin-top:0">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px">
            <span style="font-size:28px">🗺️</span>
            <div>
                <div style="font-size:18px; font-weight:700; color:#fff">{country}</div>
                <div style="font-size:12px; color:{region_color}; font-weight:600; letter-spacing:1px; text-transform:uppercase">{region}</div>
            </div>
        </div>
        <p style="color:#bbb; font-size:14px; line-height:1.7; margin:0">
            <strong style="color:#e8e8e8">Market profile:</strong> {region_note}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Regional recommendation nudge
    _region_nudges = {
        "North America":           "Consider upselling to a premium annual plan — ARPU tolerance is highest in this market.",
        "Latin America":           "Lead with value messaging; a discounted plan or bundle deal performs well here.",
        "Europe":                  "Highlight local-language content and data privacy transparency to build trust.",
        "Middle East & Africa":    "Mobile-optimized content and downloads for offline viewing resonate strongly.",
        "South Asia":              "Regional language content recommendations and mobile data-friendly streaming quality are key.",
        "East & Southeast Asia":   "Anime, K-drama, and local genre content should anchor all recommendations.",
        "Oceania":                 "Treat similar to North America — premium positioning and English-language originals work well.",
    }
    nudge = _region_nudges.get(region, "")
    if nudge:
        st.markdown(f"""
        <div class="reco-item" style="border-left:3px solid {region_color}; color:#e0e0e0; font-size:14px; margin-bottom:8px">
            💡 <strong>Regional tip:</strong> {nudge}
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────────────
    st.markdown("<div class='section-title'>🎯 Recommended Actions</div>", unsafe_allow_html=True)
    for rec in CLUSTER_RECOS[label]:
        st.markdown(f'<div class="reco-item">{rec}</div>', unsafe_allow_html=True)

    # ── Gemini Chat ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🤖 AI Analyst Chat</div>", unsafe_allow_html=True)
    user_data = {
        "Age": r["age"], "Country": r["country"], "Subscription_Type": r["sub"],
        "Favorite_Genre": r["genre"], "Watch_Time_Hours": r["watch"],
        "Days_Since_Last_Login": r["days"],
    }
    render_chat(
        session_key="su_chat",
        user_data=user_data,
        label=label,
        eng=eng,
        conf=conf,
        api_key=gemini_api_key,
    )