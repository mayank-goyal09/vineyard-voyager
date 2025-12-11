import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# 1. Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Vineyard Voyager 🍷",
    page_icon="🍇",
    layout="wide"
)

# -------------------------------------------------
# 2. Dark pink + dark blue theme (CSS)
# -------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* App background: deep navy + wine gradient */
.stApp {
    background: radial-gradient(circle at top left, #2b0b3a 0%, #050816 40%, #020617 100%);
    color: #e5e7eb;
}

.block-container {
    max-width: 1350px;
    padding-top: 1rem;
}

/* Hide default header */
[data-testid="stHeader"] {background: transparent;}
header {background: transparent;}

/* Hero header */
.hero-card {
    background: radial-gradient(circle at top left, #4c1d95 0%, #1f2937 55%, #020617 100%);
    border-radius: 22px;
    padding: 20px 26px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 22px 55px rgba(148, 27, 109, 0.75);
    border: 1px solid rgba(236, 72, 153, 0.7);
    margin-bottom: 24px;
}
.hero-left {
    display: flex;
    gap: 18px;
    align-items: center;
}
.hero-icon {
    font-size: 3rem;
    padding: 14px;
    border-radius: 20px;
    background: radial-gradient(circle at 30% 20%, #f472b6 0, #ec4899 35%, #581c87 80%);
    box-shadow: 0 0 40px rgba(244, 114, 182, 0.9);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: #f9a8d4;
}
.hero-sub {
    font-size: 0.95rem;
    color: #e5e7eb;
    margin-top: 4px;
}
.hero-tags {
    margin-top: 8px;
}
.hero-tag {
    display: inline-block;
    margin-right: 6px;
    font-size: 0.76rem;
    color: #fdf2f8;
    background: rgba(236, 72, 153, 0.25);
    border-radius: 999px;
    padding: 3px 10px;
    border: 1px solid rgba(251, 113, 133, 0.7);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.hero-badge {
    font-size: 0.8rem;
    padding: 8px 16px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid #f472b6;
    color: #f9a8d4;
    text-align: center;
    line-height: 1.4;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, #0b1020 0%, #020617 65%);
    border-right: 1px solid rgba(148, 163, 184, 0.4);
}
[data-testid="stSidebar"] * {
    color: #e5e7eb;
}
.sidebar-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f9a8d4;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}

/* Generic panels */
.wine-panel {
    background: radial-gradient(circle at top left, rgba(15,23,42,0.98) 0%, rgba(15,23,42,0.95) 60%);
    border-radius: 18px;
    padding: 20px 22px;
    border: 1px solid rgba(148, 163, 184, 0.6);
    box-shadow: 0 18px 45px rgba(0,0,0,0.9);
    margin-bottom: 20px;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f9a8d4;
    margin-bottom: 4px;
}
.panel-sub {
    font-size: 0.82rem;
    color: #9ca3af;
    margin-bottom: 12px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: rgba(15,23,42,0.9);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    border: 1px solid rgba(148,163,184,0.7);
    box-shadow: 0 10px 24px rgba(0,0,0,0.8);
}
[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
    font-size: 0.78rem;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #f9a8d4 !important;
    font-weight: 700;
    font-size: 1.3rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.95);
    border-radius: 14px;
    border: 1px dashed #f472b6;
    padding: 1rem;
}
[data-testid="stFileUploader"] label {
    color: #f9a8d4 !important;
    font-weight: 600;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(120deg, #ec4899, #6366f1);
    color: #fdf2f8 !important;
    font-weight: 700;
    border-radius: 999px;
    border: none;
    padding: 0.55rem 1.4rem;
    font-size: 0.9rem;
    box-shadow: 0 12px 30px rgba(236,72,153,0.6);
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(120deg, #db2777, #4f46e5);
}

/* Download sample special */
.sample-download > button {
    background: linear-gradient(120deg, #22c55e, #16a34a) !important;
    color: #ecfdf3 !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.7);
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(15,23,42,0.95);
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.7);
    color: #f9a8d4 !important;
    font-weight: 600;
}

/* Messages */
.stSuccess, .stInfo, .stWarning {
    border-radius: 10px;
}
.stSuccess {
    background: rgba(22,163,74,0.18);
    border-left: 4px solid #22c55e;
}
.stInfo {
    background: rgba(59,130,246,0.18);
    border-left: 4px solid #3b82f6;
}
.stWarning {
    background: rgba(248,113,113,0.2);
    border-left: 4px solid #f97373;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 3. Hero header
# -------------------------------------------------
st.markdown(
    """
<div class="hero-card">
  <div class="hero-left">
    <div class="hero-icon">🍇</div>
    <div>
      <div class="hero-title">Vineyard Voyager</div>
      <div class="hero-sub">
        Explore natural wine groupings with Hierarchical (Agglomerative) Clustering and rich visual analytics.
      </div>
      <div class="hero-tags">
        <span class="hero-tag">Agglomerative Clustering</span>
        <span class="hero-tag">PCA Visualization</span>
        <span class="hero-tag">UCI Wine Dataset</span>
      </div>
    </div>
  </div>
  <div class="hero-badge">
    Sommelier View<br/>
    Data-Driven Wine Clusters
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="wine-panel">
  <div class="panel-title">🍷 How to use this app</div>
  <div class="panel-sub">
    1️⃣ Download the sample red wine CSV. &nbsp; 2️⃣ Upload that CSV (or your own wine data). &nbsp;
    3️⃣ Adjust clustering settings in the sidebar. &nbsp; 4️⃣ Explore interactive cluster visuals and export results.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 4. Load default red + white data
# -------------------------------------------------
@st.cache_data
def load_default_data():
    red = pd.read_csv("winequality-red.csv", sep=";")
    white = pd.read_csv("winequality-white.csv", sep=";")

    red["type"] = "red"
    white["type"] = "white"

    df_default = pd.concat([red, white], axis=0, ignore_index=True)
    return df_default, red  # combined, red-only

df_default, df_red = load_default_data()

# -------------------------------------------------
# 5. Sample dataset download
# -------------------------------------------------
st.subheader("📂 Sample Dataset")

st.markdown(
    """
<div class="wine-panel">
  <div class="panel-title">Download a ready-to-use red wine CSV</div>
  <div class="panel-sub">
    Based on the classic UCI Red Wine Quality dataset with physicochemical properties
    such as acidity, residual sugar, sulphates, alcohol, and quality score.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

csv_red = df_red.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Sample Red Wine CSV",
    data=csv_red,
    file_name="winequality-red-sample.csv",
    mime="text/csv",
    help="Download this, inspect in Excel/Sheets, and re-upload to explore clustering.",
    key="download_sample",
    type="primary",
)

st.markdown("---")

# -------------------------------------------------
# 6. Upload section
# -------------------------------------------------
st.subheader("📤 Upload Your Wine CSV")

st.markdown(
    """
<div class="wine-panel">
  <div class="panel-title">Bring your own wine data</div>
  <div class="panel-sub">
    Upload the sample CSV you downloaded, or your own dataset with similar numeric columns
    (e.g., acidity, sugar, sulphates, alcohol, quality).
  </div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drag and drop or browse for a CSV file",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Using uploaded data: **{uploaded_file.name}** ({len(df):,} rows)")
else:
    df = df_default.copy()
    st.info("ℹ️ No file uploaded. Falling back to default combined red + white wine dataset.")

st.markdown("---")

# -------------------------------------------------
# 7. Sidebar: feature selection & clustering settings
# -------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">⚙️ Clustering Controls</div>', unsafe_allow_html=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_features = [c for c in numeric_cols if c.lower() != "quality"]
if len(default_features) == 0:
    default_features = numeric_cols

feature_cols = st.sidebar.multiselect(
    "Numeric features to use",
    options=numeric_cols,
    default=default_features,
    help="These numeric columns will feed the clustering and PCA projection.",
)

if len(feature_cols) < 2:
    st.warning("Please select at least 2 numeric features for clustering.")
    st.stop()

X = df[feature_cols].values

st.sidebar.write(f"Samples: **{df.shape[0]}**")
st.sidebar.write(f"Features: **{len(feature_cols)}**")

st.sidebar.markdown('<div class="sidebar-header">📌 Clustering Parameters</div>', unsafe_allow_html=True)

n_clusters = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=10,
    value=4,
    step=1,
)

linkage = st.sidebar.selectbox(
    "Linkage method",
    options=["ward", "complete", "average"],
    help="Ward linkage works well with Euclidean distance and numeric data.",
)

metric = "euclidean"  # for Ward must be Euclidean

# -------------------------------------------------
# 8. Scale + Agglomerative Clustering
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

agg = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric=metric,
    linkage=linkage,
)

cluster_labels = agg.fit_predict(X_scaled)
df["cluster"] = cluster_labels

# -------------------------------------------------
# 9. PCA for visualization
# -------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["cluster"] = df["cluster"]
if "type" in df.columns:
    df_pca["type"] = df["type"]

explained = pca.explained_variance_ratio_

# -------------------------------------------------
# 10. Main layout: PCA plot + cluster summary
# -------------------------------------------------
st.subheader("📊 Cluster Landscape")

top_panel_col1, top_panel_col2, top_panel_col3 = st.columns(3)
top_panel_col1.metric("Clusters (K)", n_clusters)
top_panel_col2.metric("PCA Var. Explained", f"{explained.sum()*100:.1f}%")
top_panel_col3.metric("Samples", f"{len(df):,}")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="wine-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">PCA Scatter by Cluster</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel-sub">Projection onto first two principal components (PC1 & PC2 explain ~{explained.sum()*100:.1f}% of variance).</div>',
        unsafe_allow_html=True,
    )

    # Plotly scatter – clusters in bright colors on dark background
    color_map = px.colors.qualitative.Set3
    df_pca_plot = df_pca.copy()
    df_pca_plot["cluster_str"] = df_pca_plot["cluster"].astype(str)

    if "type" in df_pca_plot.columns:
        fig = px.scatter(
            df_pca_plot,
            x="PC1",
            y="PC2",
            color="cluster_str",
            symbol="type",
            color_discrete_sequence=color_map,
            opacity=0.75,
            hover_data=["cluster_str", "type"],
            height=480,
        )
    else:
        fig = px.scatter(
            df_pca_plot,
            x="PC1",
            y="PC2",
            color="cluster_str",
            color_discrete_sequence=color_map,
            opacity=0.75,
            hover_data=["cluster_str"],
            height=480,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.9)",
        font=dict(color="#e5e7eb"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.25)", zerolinecolor="rgba(148,163,184,0.4)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.25)", zerolinecolor="rgba(148,163,184,0.4)"),
        legend=dict(title="Cluster", bgcolor="rgba(15,23,42,0.9)", bordercolor="rgba(148,163,184,0.7)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="wine-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Cluster Composition</div>', unsafe_allow_html=True)

    st.markdown("**📦 Cluster Sizes**")
    st.dataframe(
        df["cluster"].value_counts().sort_index().rename("count").to_frame(),
        use_container_width=True,
        height=220,
    )

    if "type" in df.columns:
        st.markdown("**🍷 Wine Type by Cluster**")
        type_cluster_counts = df.groupby(["cluster", "type"]).size().unstack(fill_value=0)
        st.dataframe(type_cluster_counts, use_container_width=True, height=220)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 11. Feature distribution by cluster (boxplot)
# -------------------------------------------------
st.subheader("📊 Feature Distribution by Cluster")

st.markdown('<div class="wine-panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Compare a feature across clusters</div>', unsafe_allow_html=True)

feature_for_boxplot = st.selectbox(
    "Select a feature",
    options=feature_cols,
    index=0,
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.set_theme(style="darkgrid")

if "type" in df.columns:
    sns.boxplot(
        data=df,
        x="cluster",
        y=feature_for_boxplot,
        hue="type",
        ax=ax,
        palette="rocket",
    )
    ax.legend(title="Type")
else:
    sns.boxplot(
        data=df,
        x="cluster",
        y=feature_for_boxplot,
        ax=ax,
        palette="rocket",
    )

ax.set_xlabel("Cluster", color="white")
ax.set_ylabel(feature_for_boxplot, color="white")
ax.set_title(f"{feature_for_boxplot} by Cluster", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

fig.patch.set_facecolor("#020617")
ax.set_facecolor("#020617")

st.pyplot(fig)
plt.close(fig)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 12. Cluster profiles (mean feature values)
# -------------------------------------------------
st.subheader("📈 Cluster Profiles (Mean Values)")

st.markdown('<div class="wine-panel">', unsafe_allow_html=True)

profile_cols = feature_cols.copy()
if "quality" in df.columns and "quality" not in profile_cols:
    profile_cols.append("quality")

cluster_profile = df.groupby("cluster")[profile_cols].mean().round(2)
st.dataframe(cluster_profile, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 13. Cluster inspector
# -------------------------------------------------
st.subheader("🔍 Inspect a Single Cluster")

st.markdown('<div class="wine-panel">', unsafe_allow_html=True)

selected_cluster = st.selectbox(
    "Choose cluster to inspect",
    options=sorted(df["cluster"].unique()),
)

cluster_df = df[df["cluster"] == selected_cluster]

st.write(f"Rows in cluster {selected_cluster}: **{cluster_df.shape[0]}**")
st.dataframe(cluster_df.head(20), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 14. Download results
# -------------------------------------------------
st.subheader("💾 Download Results")

st.markdown('<div class="wine-panel">', unsafe_allow_html=True)

csv_clusters = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Data with Cluster Labels (CSV)",
    data=csv_clusters,
    file_name="wine_clusters.csv",
    mime="text/csv",
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<hr style="margin-top:2.5rem; border-color:#ec4899; border-width:1px;">
<div style="text-align:center; padding:1.2rem 0; color:#e5e7eb; font-size:0.85rem;">

  <div style="margin-bottom:6px; font-weight:600; color:#f9a8d4;">
    © 2025 Vineyard Voyager · Hierarchical Wine Clustering · Built by
    <span style="color:#f973c9; font-weight:800;">Mayank Goyal</span>
  </div>

  <div style="margin-bottom:4px;">
    <a href="https://www.linkedin.com/in/mayank-goyal-4b8756363" target="_blank"
       style="color:#93c5fd; text-decoration:none; margin-right:18px; font-weight:600;">
        🔗 LinkedIn
    </a>
    <a href="https://github.com/mayank-goyal09" target="_blank"
       style="color:#a5b4fc; text-decoration:none; font-weight:600;">
        💻 GitHub
    </a>
  </div>

  <div style="margin-top:6px; font-size:0.78rem; color:#9ca3af;">
    🍷 Agglomerative Clustering · PCA Visualization · UCI Wine Quality Dataset
  </div>

</div>
""", unsafe_allow_html=True)
