"""
ui_batch.py — Batch Analysis page with region-aware visualizations
"""
import pandas as pd
import streamlit as st
import plotly.express as px
from config import (
    CLUSTER_NAMES, CLUSTER_ICONS, CLUSTER_DESC, CLUSTER_RECOS,
    CLUSTER_PALETTE, REGION_PALETTE, PLOT_LAYOUT,
    SUBSCRIPTIONS, GENRES,
)
from features import predict_clusters, generate_sample_data


def render(scaler, kmeans, encoder):
    st.markdown("# 📁 Batch Analysis")

    tab1, tab2 = st.tabs(["📤 Upload CSV", "🎲 Sample Data"])
    df_in = None

    with tab1:
        ups = st.file_uploader("Drop CSV file(s)", type=["csv"], accept_multiple_files=True)
        if ups:
            df_in = pd.concat([pd.read_csv(u) for u in ups], ignore_index=True)
            if "Last_Login" in df_in.columns and "Days_Since_Last_Login" not in df_in.columns:
                df_in["Days_Since_Last_Login"] = (
                    pd.to_datetime("today") - pd.to_datetime(df_in["Last_Login"], errors="coerce")
                ).dt.days.fillna(0).astype(int)

    with tab2:
        n_users = st.slider("Number of users to generate", 100, 2000, 500, 100)
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            st.session_state["sample_df"] = generate_sample_data(n_users)
            st.success(f"✓ Generated {n_users:,} synthetic users!")

        if "sample_df" in st.session_state:
            sample = st.session_state["sample_df"]
            st.dataframe(sample.head(8), use_container_width=True)
            ca, cb = st.columns(2)
            with ca:
                if st.button("▶ Run Analysis", use_container_width=True, type="primary"):
                    df_in = sample
            with cb:
                st.download_button(
                    "⬇ Download Sample CSV",
                    data=sample.to_csv(index=False).encode("utf-8"),
                    file_name="sample_netflix_users.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    if df_in is None or len(df_in) == 0:
        return

    required = ["Age", "Country", "Subscription_Type", "Favorite_Genre", "Watch_Time_Hours", "Days_Since_Last_Login"]
    missing  = [c for c in required if c not in df_in.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    with st.spinner("Running segmentation…"):
        res = predict_clusters(df_in[required], use_qcut=True, scaler=scaler, kmeans=kmeans, encoder=encoder)

    total    = len(res)
    active   = int((res["Cluster_Label"] == "Active Heavy Watchers").sum())
    inactive = int((res["Cluster_Label"] == "Inactive Users").sum())
    casual   = int((res["Cluster_Label"] == "Casual Users").sum())

    # ── KPIs ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Overview</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="metric-card"><div class="label">Total Users</div><div class="value">{total:,}</div><div class="sub">analyzed</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="label">🔥 Active Heavy</div><div class="value">{active:,}</div><div class="sub">{active/total*100:.0f}% of base</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="label">😴 Inactive</div><div class="value">{inactive:,}</div><div class="sub">{inactive/total*100:.0f}% churn risk</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="label">📺 Casual</div><div class="value">{casual:,}</div><div class="sub">{casual/total*100:.0f}% growth opp.</div></div>', unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📈 Segment Visualizations</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        counts = res["Cluster_Label"].value_counts().reset_index()
        counts.columns = ["Cluster", "Count"]
        fig = px.pie(counts, values="Count", names="Cluster", hole=0.55,
                     color="Cluster", color_discrete_map=CLUSTER_PALETTE, title="Segment Distribution")
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_traces(textfont_color="#fff", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        agg = res.groupby("Cluster_Label")[["Age", "Watch_Time_Hours", "Days_Since_Last_Login", "Engagement_Score"]].mean().round(2).reset_index()
        fig = px.bar(agg.melt(id_vars="Cluster_Label"), x="Cluster_Label", y="value",
                     color="variable", barmode="group", title="Average Metrics per Segment",
                     color_discrete_sequence=["#e50914", "#f59e0b", "#60a5fa", "#a78bfa"])
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(res, x="Engagement_Score", color="Cluster_Label", nbins=40,
                           color_discrete_map=CLUSTER_PALETTE, title="Engagement Score Distribution",
                           opacity=0.8, barmode="overlay")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        sub_ct = res.groupby(["Subscription_Type", "Cluster_Label"]).size().reset_index(name="Count")
        fig = px.bar(sub_ct, x="Subscription_Type", y="Count", color="Cluster_Label",
                     barmode="stack", title="Subscription × Segment", color_discrete_map=CLUSTER_PALETTE)
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)



    # ── Region Charts ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>🌍 Regional Breakdown</div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        # Users per region colored by region
        reg_counts = res["Region"].value_counts().reset_index()
        reg_counts.columns = ["Region", "Count"]
        fig = px.bar(reg_counts, x="Region", y="Count", color="Region",
                     color_discrete_map=REGION_PALETTE,
                     title="Users per Region")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        # Segment mix per region (stacked 100%)
        reg_seg = res.groupby(["Region", "Cluster_Label"]).size().reset_index(name="Count")
        reg_total = reg_seg.groupby("Region")["Count"].transform("sum")
        reg_seg["Pct"] = (reg_seg["Count"] / reg_total * 100).round(1)
        fig = px.bar(reg_seg, x="Region", y="Pct", color="Cluster_Label",
                     color_discrete_map=CLUSTER_PALETTE,
                     title="Segment Mix by Region (%)",
                     barmode="stack")
        fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30, yaxis_title="% of Region")
        st.plotly_chart(fig, use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        # Avg engagement per region
        reg_eng = res.groupby("Region")["Engagement_Score"].mean().reset_index()
        reg_eng.columns = ["Region", "Avg Engagement"]
        reg_eng = reg_eng.sort_values("Avg Engagement", ascending=False)
        fig = px.bar(reg_eng, x="Region", y="Avg Engagement", color="Region",
                     color_discrete_map=REGION_PALETTE,
                     title="Average Engagement Score by Region")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        # Inactive % per region (churn heatmap as bar)
        inactive_pct = (
            res[res["Cluster_Label"] == "Inactive Users"]
            .groupby("Region").size()
            / res.groupby("Region").size()
            * 100
        ).reset_index()
        inactive_pct.columns = ["Region", "Inactive %"]
        inactive_pct = inactive_pct.sort_values("Inactive %", ascending=False)
        fig = px.bar(inactive_pct, x="Region", y="Inactive %", color="Inactive %",
                     color_continuous_scale=["#4ade80", "#f59e0b", "#f87171"],
                     title="Churn Risk % by Region (Inactive Users)")
        fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Playbook ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🧭 Segment Playbook</div>", unsafe_allow_html=True)
    pb_cols = st.columns(3)
    for col, (_, name) in zip(pb_cols, CLUSTER_NAMES.items()):
        with col:
            st.markdown(f"**{CLUSTER_ICONS[name]} {name}**")
            st.markdown(f"<small style='color:#aaa'>{CLUSTER_DESC[name]}</small>", unsafe_allow_html=True)
            for rec in CLUSTER_RECOS[name]:
                st.markdown(
                    f'<div class="reco-item" style="font-size:13px;padding:10px 14px">{rec}</div>',
                    unsafe_allow_html=True,
                )

    # ── Results table + download ───────────────────────────────────
    st.markdown("<div class='section-title'>📋 Results</div>", unsafe_allow_html=True)
    display_cols = [
        "Age", "Country", "Region", "Subscription_Type", "Favorite_Genre",
        "Watch_Time_Hours", "Days_Since_Last_Login",
        "Engagement_Score", "Cluster_Label", "Cluster_Confidence",
    ]
    st.dataframe(res[display_cols].head(500), use_container_width=True)
    st.download_button(
        "⬇ Download Full Results (CSV)",
        data=res[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="segmented_users.csv",
        mime="text/csv",
        use_container_width=True,
    )