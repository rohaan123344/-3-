import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Getz Pharma Customer Segmentation
# Unsupervised learning with K-Means
# ------------------------------------------------------------

st.set_page_config(
    page_title="Getz Pharma | Customer Segmentation",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Getz Pharma Customer Segmentation")
st.write(
    "This app uses **K-Means clustering**, an unsupervised learning algorithm, "
    "to discover groups in customer/order data without a target label."
)

st.info(
    "For the assignment submission, use the actual Getz Pharma CSV dataset. "
    "The sales_analysis.csv file supplied with the class materials is general sales data, "
    "so it should not be presented as Getz Pharma data."
)

# Search for a dataset committed alongside this Python file.
DATA_CANDIDATES = [
    "getz_pharma_customer_segmentation_1000 (1).csv",
    "getz_pharma_customer_segmentation_1000.csv",
    "getz_pharma_customer_segmentation.csv",
    "getz_pharma.csv",
]

def load_local_dataset():
    for filename in DATA_CANDIDATES:
        path = Path(__file__).resolve().parent / filename
        if path.exists():
            return pd.read_csv(path), filename
    return None, None

local_df, local_filename = load_local_dataset()

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload the Getz Pharma CSV file", type=["csv"])
    if local_filename:
        st.caption(f"Found project file: {local_filename}")
    st.divider()
    st.header("Clustering settings")
    n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=3)
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
elif local_df is not None:
    df = local_df.copy()
    source_name = local_filename
else:
    st.warning(
        "No Getz Pharma dataset was found. Add the correct CSV file to the same folder "
        "as Pharma.py or upload it using the sidebar to continue."
    )
    st.code("getz_pharma_customer_segmentation_1000 (1).csv")
    st.stop()

if df.empty:
    st.error("The selected CSV has no rows.")
    st.stop()

# Clean column names and remove completely empty rows/columns.
df.columns = [str(col).strip() for col in df.columns]
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()

st.caption(f"Dataset loaded: **{source_name}** · {df.shape[0]:,} rows · {df.shape[1]} columns")

with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)
    left, right = st.columns(2)
    left.metric("Rows", f"{len(df):,}")
    right.metric("Columns", f"{len(df.columns)}")

# Choose numeric features; do not use an existing cluster/target label as an input.
numeric_df = df.select_dtypes(include=[np.number]).copy()
excluded_words = ("cluster", "target", "label", "segment", "class", "id", "orderid", "customerid")
feature_candidates = [
    col for col in numeric_df.columns
    if not any(word in col.lower().replace("_", "").replace(" ", "") for word in excluded_words)
]

if len(feature_candidates) < 2:
    st.error(
        "At least two usable numeric feature columns are needed for clustering. "
        "Check that the CSV contains numeric customer/order measurements."
    )
    st.write("Numeric columns detected:", list(numeric_df.columns))
    st.stop()

st.subheader("1. Select features")
st.write(
    "Choose numeric columns that describe the customers or their behaviour. "
    "Identifiers and pre-existing labels are excluded from the default selection."
)
default_features = feature_candidates[: min(5, len(feature_candidates))]
selected_features = st.multiselect(
    "Features used by K-Means",
    options=feature_candidates,
    default=default_features,
)
if len(selected_features) < 2:
    st.warning("Select at least two numeric features to continue.")
    st.stop()

# Convert selected values to numeric, impute missing values with each column's median,
# and remove rows where all selected feature values are missing.
X_raw = df[selected_features].apply(pd.to_numeric, errors="coerce")
valid_rows = X_raw.notna().any(axis=1)
X_raw = X_raw.loc[valid_rows].copy()
working_df = df.loc[valid_rows].copy()

if len(X_raw) < n_clusters:
    st.error(f"There are only {len(X_raw)} usable rows, which is fewer than K={n_clusters}.")
    st.stop()

for col in selected_features:
    median_value = X_raw[col].median()
    if pd.isna(median_value):
        st.error(f"Feature '{col}' contains no usable numeric values.")
        st.stop()
    X_raw[col] = X_raw[col].fillna(median_value)

# StandardScaler puts features on comparable scales before K-Means.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_seed), n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

result_df = working_df.copy()
result_df["Cluster"] = cluster_labels + 1  # use 1-based labels for easier reading

st.subheader("2. Clustering results")
m1, m2, m3 = st.columns(3)
m1.metric("Customers / records clustered", f"{len(result_df):,}")
m2.metric("Number of clusters", str(n_clusters))
m3.metric("Features used", str(len(selected_features)))

counts = result_df["Cluster"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
ax.set_title("Number of records in each cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of records")
st.pyplot(fig)
plt.close(fig)

st.subheader("3. Understand each cluster")
profile = result_df.groupby("Cluster")[selected_features].mean().round(2)
st.write("Average feature values for each cluster:")
st.dataframe(profile, use_container_width=True)

profile_fig, profile_ax = plt.subplots(figsize=(max(8, len(selected_features) * 1.2), 4.5))
sns.heatmap(profile, annot=True, fmt=".2f", cmap="Blues", ax=profile_ax)
profile_ax.set_title("Average feature values by cluster")
st.pyplot(profile_fig)
plt.close(profile_fig)

st.subheader("4. Visualise the clusters")
if len(selected_features) >= 2:
    if len(selected_features) == 2:
        plot_df = pd.DataFrame({
            "x": X_scaled[:, 0],
            "y": X_scaled[:, 1],
            "Cluster": result_df["Cluster"].astype(str).to_numpy(),
        })
        x_label = f"{selected_features[0]} (scaled)"
        y_label = f"{selected_features[1]} (scaled)"
    else:
        pca = PCA(n_components=2, random_state=int(random_seed))
        components = pca.fit_transform(X_scaled)
        plot_df = pd.DataFrame({
            "x": components[:, 0],
            "y": components[:, 1],
            "Cluster": result_df["Cluster"].astype(str).to_numpy(),
        })
        x_label = "Principal Component 1"
        y_label = "Principal Component 2"
        st.caption(
            "More than two features were selected, so PCA projects them into two dimensions "
            "for visualisation. K-Means still uses all selected features."
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="Cluster", palette="deep", s=65, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Customer clusters")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("5. Download the results")
st.write("Download the original rows with the cluster assigned to each record.")
csv_bytes = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download clustered CSV",
    data=csv_bytes,
    file_name="getz_pharma_clustered_results.csv",
    mime="text/csv",
)

st.subheader("What this project does")
st.markdown(
    """
- **Unsupervised learning:** no target column is used to teach the model.
- **Data preparation:** empty rows/columns are removed, selected features are converted to numeric,
  and missing feature values are filled with the median.
- **Feature scaling:** `StandardScaler` helps prevent features with large numeric ranges from
  dominating the distance calculations.
- **K-Means:** groups records into the selected number of clusters.
- **Cluster profiling:** average values help describe how the groups differ.
- **Visualisation:** a bar chart shows cluster sizes; a heatmap shows cluster profiles; a scatter
  plot shows a two-dimensional view of the groups.
"""
)
