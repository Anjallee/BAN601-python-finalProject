# ---------------------------------------------
# 📊 Interactive Data Profiling Dashboard (Modular)
# Overview decluttered; Column Summary moved to Data Quality tab
# ---------------------------------------------
from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import gaussian_kde
from pandas.api.types import is_numeric_dtype, is_string_dtype

# ----------------------------
# Global plotting preferences
# ----------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# ----------------------------
# Constants & Theme
# ----------------------------
MAX_ROWS_PAIRPLOT = 2000
MAX_COLS_PAIRPLOT = 6


# ====================================
# 0) Page Config & Helper Functions
# ====================================
def setup_page() -> None:
    # Configure Streamlit page and header.
    st.set_page_config(page_title="Data Profiling Dashboard", layout="wide")
    st.markdown(
        """
### 📊 Interactive Data Profiling Dashboard

Fast EDA across multiple CSV files with quality checks, correlations, outliers, and pair plots.
        """,
        unsafe_allow_html=True,
    )


def get_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # Return dataframe containing only numeric columns.
    return df.select_dtypes(include="number")


def list_csv_files(dirpath: str = ".") -> List[str]:
    # List .csv files in a directory.
    return [f for f in os.listdir(dirpath) if f.lower().endswith(".csv")]


@st.cache_data
def load_data(file_name: str) -> pd.DataFrame:
    # Fix nullable integer types (Int64) for Arrow compatibility
    # Convert nullable integers to float64 (Because float can handle NaN)
    df = pd.read_csv(file_name)
    for col in df.select_dtypes("Int64").columns:
        df[col] = df[col].astype("float64")
    return df


def pretty_index_df(df: pd.DataFrame, index_name: str = "R.No.") -> pd.DataFrame:
    # Display helper: 1-based index with a custom left header.
    tmp = df.reset_index(drop=True).copy()
    tmp.index = tmp.index + 1
    tmp.index.name = index_name
    return tmp


def data_quality_score(df: pd.DataFrame) -> Tuple[float, float, float]:
    # -------------------------------------------------
    # Compute a simple data quality score (0-100) using missing & duplicate penalties.
    # * 50% weight for missing ratio
    # * 50% weight for duplicate ratio
    # -------------------------------------------------
    missing_ratio = df.isna().sum().sum() / df.size  # fraction of total cells
    duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0.0
    penalty = (missing_ratio * 50) + (duplicate_ratio * 50)
    score = max(0, round(100 - penalty, 2))
    return score, round(missing_ratio * 100, 2), round(duplicate_ratio * 100, 2)


# -----------------------------------------------------------------------
def count_missing_cells(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


# -----------------------------------------------------------------------
def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


# -----------------------------------------------------------------------
def find_inconsistent_categories(df: pd.DataFrame) -> dict:
    """
    Detect category inconsistencies per *string-like* column:
      - Case variants (e.g., 'NY' vs 'ny')
      - Leading/trailing spaces
      - Multiple internal spaces
    Returns:
      {
        col: {
          "raw_unique": int,
          "normalized_unique": int,
          "leading_trailing_spaces": int,
          "multiple_spaces": int,
          "examples": List[Tuple[raw, norm]]
        }, ...
      }
    """
    issues = {}
    for col in df.columns:
        if is_string_dtype(df[col]):
            s = df[col].dropna().astype(str)
            s_norm = (
                s.str.strip()
                 .str.replace(r"\s+", " ", regex=True)
                 .str.lower()
            )

            raw_unique = int(s.nunique(dropna=True))
            norm_unique = int(s_norm.nunique(dropna=True))

            has_leading_trailing = (s != s.str.strip()).sum()
            has_multi_space = s.str.contains(r"\s{2,}", regex=True).sum()

            flag = (norm_unique < raw_unique) or (has_leading_trailing > 0) or (has_multi_space > 0)
            if flag:
                diffs = s[s != s_norm]
                examples = []
                if not diffs.empty:
                    sample_idx = diffs.sample(min(5, len(diffs)), random_state=42).index
                    for i in sample_idx:
                        examples.append((s.loc[i], s_norm.loc[i]))

                issues[col] = {
                    "raw_unique": raw_unique,
                    "normalized_unique": norm_unique,
                    "leading_trailing_spaces": int(has_leading_trailing),
                    "multiple_spaces": int(has_multi_space),
                    "examples": examples,
                }
    return issues


def profile_suspicious_numeric(
    df: pd.DataFrame, outlier_method: str = "IQR", outlier_threshold_ratio: float = 0.05
) -> dict:
    """
    Flag suspicious patterns per numeric column:
      - Constant (single unique value)
      - Zero-inflated (>=90% zeros)
      - All ≤ 0 or mixed sign (informational)
      - Heavy outliers (IQR or Z-score) if outlier ratio > threshold (default 5%)
    Returns:
      {
        col: {
          "unique_values": int,
          "zero_ratio": float,
          "outlier_ratio": float,
          "flags": List[str]
        }, ...
      }
    """
    susp = {}
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if s.empty:
                continue

            uniq = s.nunique()
            const = uniq == 1
            zero_ratio = (s == 0).mean()
            zero_inflated = zero_ratio >= 0.90

            all_non_positive = (s <= 0).all()
            has_neg = (s < 0).any()
            has_pos = (s > 0).any()

            # Outliers
            outlier_ratio = 0.0
            if len(s) > 3:
                if outlier_method == "IQR":
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    if iqr != 0:
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        outlier_ratio = ((s < lower) | (s > upper)).mean()
                else:  # "Z"
                    std = s.std(ddof=0)
                    if std:
                        z = (s - s.mean()) / std
                        outlier_ratio = (z.abs() > 3).mean()

            flags = []
            if const:
                flags.append("constant")
            if zero_inflated:
                flags.append(f"zero-inflated ({zero_ratio:.0%} zeros)")
            if all_non_positive:
                flags.append("all ≤ 0")
            elif has_neg and has_pos:
                flags.append("mixed-sign")
            if outlier_ratio >= outlier_threshold_ratio:
                flags.append(f"outliers≈{outlier_ratio:.0%}")

            if flags:
                susp[col] = {
                    "unique_values": int(uniq),
                    "zero_ratio": round(float(zero_ratio), 4),
                    "outlier_ratio": round(float(outlier_ratio), 4),
                    "flags": flags,
                }
    return susp


# ================================
# 1) Sidebar Controls
# ================================
def sidebar_controls() -> Tuple[str, Optional[str], int]:
    """
    Returns: (selected_file, selected_numeric_column_from_RAW_or_None, bins)

    Behavior:
      - Remembers the last chosen CSV across reruns using st.session_state['selected_file_preselect'].
      - If that file exists in the refreshed list, it remains selected.
    """
    st.sidebar.header("Controls")

    # List CSVs (absolute paths for stable matching)
    entries = [f for f in list_csv_files(".")]
    abs_paths = [os.path.abspath(p) for p in entries]

    if not abs_paths:
        st.sidebar.error("No CSV files found in the current directory.")
        st.stop()

    # Build display labels (just the base names) but keep absolute paths as the true values
    labels = [os.path.basename(p) for p in abs_paths]

    # Determine preselection index from session (if present and still exists)
    preselect = st.session_state.get("selected_file_preselect")
    if preselect and os.path.abspath(preselect) in abs_paths:
        idx = abs_paths.index(os.path.abspath(preselect))
    else:
        idx = 0  # default to first if no previous or no longer exists

    # Show selectbox; store absolute path as the returned value
    selected_label = st.sidebar.selectbox(
        "Select Dataset",
        options=labels,
        index=idx,
        key="select_dataset_label",  # UI key only
    )
    # Map back to absolute path
    selected_file = abs_paths[labels.index(selected_label)]

    # Persist current choice for next rerun
    st.session_state["selected_file_preselect"] = selected_file

    # Load raw and build the numeric column selector from this selected file
    df = load_data(selected_file)
    numeric_columns = list(df.select_dtypes(include="number").columns)
    selected_column = None
    if numeric_columns:
        selected_column = st.sidebar.selectbox("Select Numeric Column", numeric_columns, key="select_numeric_column")

    bins = st.sidebar.slider("Number of Bins", min_value=5, max_value=50, value=20)
    return selected_file, selected_column, bins
    
def sidebar_controls_old() -> Tuple[str, Optional[str], int]:
    """
    Render sidebar controls and return:
    - selected CSV file
    - selected numeric column (or None if none exist)
    - number of bins for histogram
    """
    st.sidebar.header("Controls")
    csv_files = list_csv_files(".")
    if not csv_files:
        st.sidebar.error("No CSV files found in the current directory.")
        st.stop()

    selected_file = st.sidebar.selectbox("Select Dataset", csv_files)
    df = load_data(selected_file)

    numeric_columns = list(df.select_dtypes(include="number").columns)
    selected_column = None
    if numeric_columns:
        selected_column = st.sidebar.selectbox("Select Numeric Column", numeric_columns)

    bins = st.sidebar.slider("Number of Bins", min_value=5, max_value=50, value=20)
    return selected_file, selected_column, bins


# ================================
# 2) Tabs
# ================================
# ------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------------
def render_overview(df: pd.DataFrame) -> None:
    """Overview tab: ONLY the Dataset Overview / Summary (KPIs + Quality Score)."""
    st.write("### Dataset Overview / Summary")

    # --- Quick metrics ---
    rows, cols = df.shape
    num_cols = len(df.select_dtypes(include="number").columns)
    missing_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Numeric Columns", num_cols)
    c4.metric("Missing Values (cells)", missing_cells)
    c5.metric("Duplicate Rows", dup_rows)

    # --- Data Quality Score banner ---
    indent = "\u00A0"  # non-breaking space
    score, miss_pct, dup_pct = data_quality_score(df)
    quality_str = (
        f"**Missing ratio** = {miss_pct}%{indent*10} **Duplicate ratio** = {dup_pct}%"
    )
    st.info(quality_str)

    
    # --- First 5 Rows ---
    st.markdown("---")
    st.subheader("First 5 Rows")
    st.dataframe(pretty_index_df(df.head(), "R.No."))
    

def render_overview_old (df: pd.DataFrame) -> None:
    indent = "\u00A0"  # non-breaking space
    st.write("### Dataset Overview / Summary")

    # --- Quick health computations ---
    missing_cells = count_missing_cells(df)
    duplicate_rows = count_duplicate_rows(df)
    incats = find_inconsistent_categories(df)      # dict per text column
    suspnum = profile_suspicious_numeric(df)       # dict per numeric column

    # --- Top metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", len(get_numeric_df(df).columns))
    col4.metric("Missing Values (cells)", missing_cells)
    col5.metric("Duplicate Rows", duplicate_rows)

    # --- Data Quality Score banner ---
    score, miss_pct, dup_pct = data_quality_score(df)
    quality_str = (
        f"***Missing ratio*** = {miss_pct}%{indent*10} **Duplicate ratio** = {dup_pct}%\n\n"
        f"***QUALITY SCORE*** (QS) = **{score}%**{indent*10} "
        f"i.e. QS = 100 - (Missing% × 0.5 + Duplicate% × 0.5)"
    )
    st.info(quality_str)

    # --- Compact info line for additional flags ---
    quick_flags = []
    if incats:
        quick_flags.append(f"Inconsistent categories: {indent*5} {len(incats)} col(s)")
    if suspnum:
        quick_flags.append(f"Suspicious numeric: {indent*5} {len(suspnum)} col(s)")
    if quick_flags:
        # Use Markdown list markers (- or *) so Streamlit renders multi-line bullets
        st.info("\n".join(f"- {msg}" for msg in quick_flags))
    else:
        st.success("No immediate category inconsistencies or numeric anomalies detected by heuristics.")

    # --- Details: Inconsistent categories ---
    if incats:
        st.markdown("---")
        st.subheader("Inconsistent Categories — Details")
        det_rows = []
        for col, info in incats.items():
            det_rows.append({
                "column": col,
                "unique_raw": info["raw_unique"],
                "unique_after_normalization": info["normalized_unique"],
                "leading/trailing_spaces": info["leading_trailing_spaces"],
                "multiple_spaces": info["multiple_spaces"],
                "examples (raw → normalized)": "; ".join([f"{a} → {b}" for a, b in info["examples"]]) if info["examples"] else ""
            })
        st.dataframe(pd.DataFrame(det_rows).sort_values("column"), use_container_width=True)

    # --- Details: Suspicious numeric columns ---
    if suspnum:
        st.markdown("---")
        st.subheader("Suspicious Numeric Columns — Details")
        det_rows = []
        for col, info in suspnum.items():
            det_rows.append({
                "column": col,
                "unique_values": info["unique_values"],
                "zero_ratio": info["zero_ratio"],
                "outlier_ratio": info["outlier_ratio"],
                "flags": ", ".join(info["flags"]),
            })
        st.dataframe(pd.DataFrame(det_rows).sort_values("column"), use_container_width=True)

    # --- First 5 Rows ---
    st.markdown("---")
    st.subheader("First 5 Rows")
    st.dataframe(pretty_index_df(df.head(), "R.No."))

    # --- Automatic numeric insights ---
    st.markdown("---")
    st.subheader("Automatic Dataset Insights")
    numeric_df = get_numeric_df(df)
    if not numeric_df.empty:
        # Highest variance column
        highest_var = numeric_df.var(numeric_only=True).idxmax()

        # Strongest correlation (absolute, excluding diagonal)
        corr_series = (
            numeric_df.corr(numeric_only=True)
                      .abs()
                      .unstack()
                      .sort_values(ascending=False)
        )
        filtered_corr = corr_series[corr_series < 1]  # remove self-correlation 1.0

        if not filtered_corr.empty:
            highest_corr = filtered_corr.idxmax()
            st.write(
                f"📌 Strongest correlation between: **{highest_corr[0]}** and **{highest_corr[1]}**"
            )
        else:
            st.write("📌 No cross-feature correlations found (beyond self-correlation).")

        st.write(f"📌 Column with highest variance: **{highest_var}**")


# -------------------------------------------------
# TAB 2 — DISTRIBUTION (Histogram)
# -------------------------------------------------
def render_distribution(df: pd.DataFrame, selected_column: Optional[str], bins: int) -> None:
    if selected_column is None or selected_column not in df.columns or not is_numeric_dtype(df[selected_column]):
        # Graceful fallback: pick the first numeric column if the sidebar choice is missing or invalid
        numeric_cols = list(get_numeric_df(df).columns)
        if not numeric_cols:
            st.warning("No numeric columns found in the selected file.")
            return
        selected_column = numeric_cols[0]

    st.subheader(f"Distribution of {selected_column}")

    # Toggle KDE density curve
    show_density = st.checkbox("Show Density Curve", value=True)

    # Remove NaNs (plots can't handle NaN)
    data = df[selected_column].dropna()

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, density=show_density, edgecolor="black")

    # KDE density curve
    if show_density and len(data) > 1:
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_vals, kde(x_vals), color="darkorange", linewidth=2)

    ax.set_xlabel(selected_column, fontsize=10, color="darkblue")
    ax.set_ylabel("Density" if show_density else "Frequency", fontsize=10, color="darkblue")
    ax.set_title(f"Histogram of Column = {selected_column}", fontsize=12, color="purple")
    st.pyplot(fig)

    st.markdown("---")
    # Summary stats
    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.markdown("---")
    # Compare Distributions
    st.subheader("Compare Distributions")
    numeric_cols = list(get_numeric_df(df).columns)
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns to compare distributions.")
        return

    col1, col2 = st.columns(2)
    col_a = col1.selectbox("Column A", numeric_cols, index=0)
    remaining = [c for c in numeric_cols if c != col_a]
    col_b = col2.selectbox("Column B", remaining, index=0 if remaining else 0)

    col1_plot, col2_plot = st.columns(2)
    fig1, ax1 = plt.subplots()
    sns.histplot(df[col_a], kde=True, ax=ax1, color="#4682b4")
    ax1.set_title(col_a)
    col1_plot.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(df[col_b], kde=True, ax=ax2, color="#ff6347")
    ax2.set_title(col_b)
    col2_plot.pyplot(fig2)

    st.markdown("---")


# --------------------------------------------------------------
# TAB 3 — DATA QUALITY
# --------------------------------------------------------------
def render_data_quality(df: pd.DataFrame) -> None:
    st.subheader("Data Quality Analysis")

    c1, c2 = st.columns(2)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])

    st.markdown("---")
    st.subheader("Column Summary (compact)")

    n_rows = max(len(df), 1)
    col_summary_long = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_values": df.nunique(dropna=True),
        "missing_values": df.isnull().sum(),
        "missing %": (df.isnull().sum() / n_rows * 100).round(2),
    }).sort_values("missing_values", ascending=False)

    # Present transposed so dataset columns are horizontal
    st.dataframe(col_summary_long.T)

    # NOTE: Missing Value Heatmap has been removed per request.
    

def render_data_quality_old(df: pd.DataFrame) -> None:
    st.subheader("Data Quality Analysis")

    # Dataset context
    c1, c2 = st.columns(2)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])

    st.markdown("---")

    # A) Column Summary (compact)
    st.subheader("Column Summary (compact)")

    n_rows = max(len(df), 1)  # avoid divide-by-zero if df is empty
    col_summary_long = pd.DataFrame({
        "dtype": df.dtypes.astype(str),                 # data type per column
        "unique_values": df.nunique(dropna=True),       # distinct non-null values
        "missing_values": df.isnull().sum(),            # count of NaNs per column
        "missing %": (df.isnull().sum() / n_rows * 100).round(2)
    })
    # Sort by Missing Values
    col_summary_long = col_summary_long.sort_values("missing_values", ascending=False)

    col_summary_wide = col_summary_long.T
    col_summary_wide.index.name = "metric"  # rows: dtype / unique_values / missing_values

    st.dataframe(col_summary_wide)
    st.markdown("---")

    # B) Missing Value Heatmap
    st.subheader("Missing Value Heatmap")
    if df.isnull().values.any():
        missing_map = df.isnull().astype(int)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            missing_map, cbar=True, cmap="Reds",
            linewidths=0.2, yticklabels=False, linecolor="lightgray", ax=ax,
        )
        ax.set_title("Missing Data Pattern Analysis", fontsize=10, color="purple")
        ax.set_xlabel("Columns", fontsize=10, color="darkblue")
        ax.set_ylabel("Rows", fontsize=10, color="darkblue")
        st.pyplot(fig)
    else:
        st.success("Dataset contains no missing values.")
    st.markdown("---")


# -------------------------------------------------
# TAB 4 — CORRELATION HEATMAP
# -------------------------------------------------
def render_correlation(df: pd.DataFrame) -> None:
    st.subheader("Correlation Heatmap")

    numeric_df = get_numeric_df(df)
    numeric_cols = list(numeric_df.columns)  # use a list for stable indexing

    # Need at least 2 numeric columns to compute correlation
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns to compute correlation.")
        return

    # Toggle to show/hide annotations on the heatmap
    show_ann = st.checkbox("Show numeric annotations", value=True)

    # Compute Pearson correlation once (reused for r in explorer)
    corr_matrix = numeric_df.corr(numeric_only=True)

    # Mask upper triangle above the diagonal to avoid duplicate info
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Heatmap (fixed annotation font size = 8)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr_matrix, mask=mask, annot=show_ann, annot_kws={"size": 8},
        cmap="coolwarm", fmt=".2f", linewidths=0.5, center=0, square=True, ax=ax,
    )
    # X-axis (column names across the top/bottom)
    ax.tick_params(axis="x", labelsize=9, colors="steelblue")
    # Y-axis (column names down the left)
    ax.tick_params(axis="y", labelsize=9, colors="steelblue")

    # Optional rotations
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")

    for lbl in ax.get_yticklabels():
        lbl.set_rotation(0)
        lbl.set_ha("right")

    ax.set_xlabel("Features", fontsize=10, color="darkblue")
    ax.set_title("Feature Correlation Heatmap", fontsize=10, color="purple")
    st.pyplot(fig, clear_figure=True)

    st.markdown("---")

    # Correlation Explorer (scatter Plot)
    st.subheader("Correlation Explorer")

    # Simple pair selector
    col1, col2 = st.columns(2)
    x_var = col1.selectbox("Select X Variable", numeric_cols)

    # Exclude X; if only one numeric column exists, fall back to all
    y_options = [c for c in numeric_cols if c != x_var] or numeric_cols
    y_var = col2.selectbox("Select Y Variable", y_options)

    # Reuse precomputed correlation; safe-guard if key is missing
    r_val = np.nan
    try:
        r_val = corr_matrix.loc[x_var, y_var]
    except Exception:
        pass

    # SPEED UPS FOR SCATTER
    MAX_SCATTER_POINTS = 2000
    xs = df[x_var]
    ys = df[y_var]
    valid = xs.notna() & ys.notna()
    xs = xs[valid]
    ys = ys[valid]
    n = len(xs)

    if n > MAX_SCATTER_POINTS:
        # consistent sub-sample for stable visuals
        sample_idx = xs.sample(MAX_SCATTER_POINTS, random_state=42).index
        xs_plot = xs.loc[sample_idx]
        ys_plot = ys.loc[sample_idx]
        note = f"Sampled {MAX_SCATTER_POINTS} of {n:,} points for performance."
    else:
        xs_plot, ys_plot = xs, ys
        note = None

    # Scatter plot for the chosen variables (rasterized & small markers)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        xs_plot, ys_plot,
        s=8,              # small marker size
        alpha=0.7,        # modest transparency
        color="#4b0082",
        rasterized=True,  # <-- speed up rendering
    )
    ax.set_xlabel(x_var, fontsize=10, color="purple")
    ax.set_ylabel(y_var, fontsize=10, color="darkblue")

    title = f"{x_var} vs {y_var}"
    if pd.notna(r_val):
        title += f" (r = {r_val:.2f})"

    ax.set_title(title, fontsize=12, color="purple")
    st.pyplot(fig, clear_figure=True)
    if note:
        st.caption(note)
    st.markdown("---")


# -------------------------------------------------
# TAB 5 — OUTLIER DETECTION
# -------------------------------------------------
def render_outliers(df: pd.DataFrame) -> None:
    st.subheader("Boxplot")
    st.markdown(
        """
**Using Outlier Detection (IQR Method)**
        """,
        unsafe_allow_html=True
    )

    numeric_df = get_numeric_df(df)
    if numeric_df.shape[1] == 0:
        st.warning("No numeric columns available.")
        return

    # ---------- BOX PLOTS ----------
    fig, ax = plt.subplots()
    # Drop rows with any NaN in numeric_df for boxplot calculation
    ax.boxplot(numeric_df.dropna().values, labels=numeric_df.columns)
    ax.set_title("Boxplot of Numeric Variables", fontsize=10, color="purple")
    # X-axis (column names across the top/bottom)
    ax.tick_params(axis="x", labelsize=9, colors="steelblue")
    # Y-axis (column names down the left)
    ax.tick_params(axis="y", labelsize=9, colors="steelblue")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.markdown("---")

    # ---------- OUTLIER CALCULATION ----------
    st.subheader("Detected Outliers")
    outlier_rows = pd.DataFrame()

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Combine conditions and ignore NaNs
        mask = ((df[column] < lower) | (df[column] > upper)) & df[column].notna()
        outliers = df[mask]
        outlier_rows = pd.concat([outlier_rows, outliers], axis=0, ignore_index=False)

    if outlier_rows.empty:
        st.success("No significant outliers detected.")
    else:
        st.write("Rows containing potential outliers:")
        st.dataframe(outlier_rows.drop_duplicates())

    st.markdown("---")


# -------------------------------------------------
# TAB 6 — PAIR PLOT
# -------------------------------------------------
def render_pairplot(df: pd.DataFrame) -> None:
    st.subheader("Smart Pair Plot Explorer")

    numeric_df = get_numeric_df(df)
    if numeric_df.shape[1] < 2:
        st.warning("Need at least two numeric columns for pair plot.")
        return

    # (1) Sample rows for performance if needed
    plot_df = numeric_df.copy()
    if len(plot_df) > MAX_ROWS_PAIRPLOT:
        plot_df = plot_df.sample(MAX_ROWS_PAIRPLOT, random_state=42)
        st.info(f"Dataset sampled to {MAX_ROWS_PAIRPLOT} rows for performance.")

    # (2) Pick most informative variables by variance
    variances = plot_df.var(numeric_only=True).sort_values(ascending=False)
    top_cols = list(variances.head(MAX_COLS_PAIRPLOT).index)
    plot_df = plot_df[top_cols]

    st.markdown(
        """
**Top 6 Variables with High Variance Used in Pair Plot**
        """,
        unsafe_allow_html=True
    )
    df_cols = pd.DataFrame([top_cols])
    df_cols.columns = range(1, len(top_cols) + 1)
    st.dataframe(df_cols, hide_index=True)

    # (3) Strongest correlation pair
    indent = "\u00A0"  # non-breaking space
    corr_abs = plot_df.corr(numeric_only=True).abs()
    np.fill_diagonal(corr_abs.values, 0)
    strongest_pair = corr_abs.stack().idxmax()
    col_x, col_y = strongest_pair
    strongest_value = plot_df.corr(numeric_only=True).loc[col_x, col_y]
    strongest_str = (
        f"Strongest correlation: {indent*10} **{col_x}  vs  {col_y}** "
        f"{indent*10} (r = {strongest_value:.2f})"
    )
    st.info(strongest_str)
    st.markdown("---")

    # (4) Top correlation pairs table
    corr_matrix = plot_df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper.stack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    corr_pairs["Correlation"] = corr_pairs["Correlation"].round(3)
    corr_pairs = corr_pairs.sort_values("Correlation", ascending=False)

    st.subheader("Top Correlations")
    corr_str = (f"Total unique (Top) correlation pairs: {indent*3} {len(corr_pairs)}")
    st.markdown(f"**{corr_str}**", unsafe_allow_html=True)
    st.dataframe(corr_pairs, hide_index=True, use_container_width=True)
    st.markdown("---")

    # (5) Pair plot
    st.subheader("Smart Pair Plot")
    g = sns.pairplot(
        plot_df,
        corner=True,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 40},
    )
    # Add per-variable titles on the diagonal
    for ax, var in zip(g.diag_axes, g.x_vars):
        ax.set_title(var, y=0.9, fontsize=10)

    # (6) Highlight strongest correlation pair (both orientations)
    for i, x in enumerate(g.x_vars):
        for j, y in enumerate(g.y_vars):
            ax = g.axes[j, i]
            if ax is None:
                continue
            if (x == col_x and y == col_y) or (x == col_y and y == col_x):
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)

    st.pyplot(g.fig)
    st.markdown("---")


# ====================================
# 3) Clean: Pipeline (All-in-One)
# ====================================
def render_clean_pipeline(df: pd.DataFrame, selected_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean: Pipeline (All-in-One)

    Steps:
      1) Categorical standardization (trim + lowercase)
      2) Date parsing (acceptance ≥ 0.60)
      3) Numeric conversion (acceptance ≥ 0.70)
      4) Negative values → NaN (auto-detect + user selection)   <-- fixed widget/state pattern
      5) Outlier detection (3×IQR) — flag only (no modification)
      6) Median imputation for numeric NaNs
      6b) Categorical imputation (selector: None / Constant 'Unknown' / Mode / Group-aware + fallback)
      7) Duplicate removal (full-row, keep='first')

    Behavior:
      - No download buttons.
      - Apply & Save writes to <same folder>/clean_<original_name>.csv
        * If the clean file exists → asks overwrite confirmation; else creates silently.
      - After saving, st.rerun() so the app refreshes; preselects the saved file path.
    """
    st.subheader("🛠️ Clean: Pipeline (All-in-One)")

    # Use last saved pipeline DF if present; else the incoming df
    source = st.session_state.get("clean_df_pipeline", df).copy()
    rows_before, cols_before = source.shape

    # -------------------------
    # Pipeline Settings (2 rows)
    # -------------------------
    st.markdown("### ⚙️ Pipeline Settings")

    def checkbox_tile(container, key: str, label: str, default: bool = True, caption: Optional[str] = None) -> bool:
        val = container.checkbox(label, value=default, key=key)
        if caption:
            container.caption(caption)
        return val

    row1 = st.columns(3, gap="large")
    do_cat_std    = checkbox_tile(row1[0], "do_cat_std",    "Categorical standardization (trim + lower)", True)
    do_date_parse = checkbox_tile(row1[1], "do_date_parse", "Date parsing (accept ≥60%)", True)
    do_num_parse  = checkbox_tile(row1[2], "do_num_parse",  "Numeric conversion (accept ≥70%)", True)

    row2 = st.columns(3, gap="large")
    do_neg_to_nan   = checkbox_tile(row2[0], "do_neg_to_nan",   "Negative values → NaN (selected columns)", True)
    do_outlier_flag = checkbox_tile(row2[1], "do_outlier_flag", "Outlier detection (3×IQR) — flag only", True)
    do_num_median   = checkbox_tile(row2[2], "do_num_median",   "Median imputation (numeric)", True)

    # NEW: categorical missing strategy
    cat_strategy = st.selectbox(
        "Categorical missing strategy",
        ["None (leave as NA)", "Constant: 'Unknown'", "Mode per column", "Group-aware (recommended)"],
        index=3,
        key="cat_missing_strategy"
    )

    st.divider()
    st.info(
        "### Defaults used by the pipeline\n"
        "- **Duplicate removal:** full-row deduplication with `keep='first'` (applied at the end).\n"
        "- **Numeric missing values:** **Median** imputation (robust to skew/outliers).\n"
        "- **Categorical/Text missing values:** controlled by the selector above."
    )
    st.divider()

    # Thresholds
    DATE_ACCEPT_THR = 0.60
    NUM_ACCEPT_THR  = 0.70

    # -------------------------------
    # Negative → NaN (UI only if ON)
    # -------------------------------
    if do_neg_to_nan:
        st.markdown("#### 🚫 Negative Values → NaN")

        # 0) Candidate numeric columns from the current 'source'
        neg_candidates = [c for c in source.columns if is_numeric_dtype(source[c])]

        # 1) Name-based patterns (define BEFORE using)
        name_patterns = [
            r"\bweight\b",
            r"\bheight\b",
            r"\bwait[_\s-]*time\b",
            r"\bduration\b",
            r"\bage\b",
            r"\bbmi\b",
            r"(?:\bquantity\b|\bqty\b)",
            r"\bcount\b",
            r"(?:\bprice\b|\bamount\b|\bcost\b)",
            r"\bdistance\b",
        ]
        name_regex = re.compile("|".join(name_patterns), flags=re.IGNORECASE)

        # 2) Reset stored selection when the dataset changes
        current_file = os.path.abspath(selected_file)
        if st.session_state.get("neg_cols_file") != current_file:
            st.session_state["neg_cols"] = [c for c in neg_candidates if name_regex.search(c)]
            st.session_state["neg_cols_file"] = current_file

        # 3) Auto-detect button (writes to internal key, not the widget key)
        if st.button("⚙️ Auto-detect non-negative columns", key="btn_autodetect_neg_cols"):
            def autodetect_nonnegative_cols(df_in: pd.DataFrame, candidates: List[str]) -> List[str]:
                suggest = set()
                # (a) name-based
                for c in candidates:
                    if name_regex.search(c):
                        suggest.add(c)
                # (b) distribution-based
                for c in candidates:
                    s = df_in[c]
                    s = pd.to_numeric(s, errors="coerce") if not is_numeric_dtype(s) else s
                    s = s.dropna()
                    if len(s) < 20:
                        continue
                    if (s >= 0).mean() >= 0.90 and (s < 0).mean() > 0 and s.median() >= 0:
                        suggest.add(c)
                # return in DF order
                return [c for c in candidates if c in suggest]

            st.session_state["neg_cols"] = autodetect_nonnegative_cols(source, neg_candidates)
            st.success(f"Auto-detected: {st.session_state['neg_cols']}")

        # 4) Build sanitized defaults from internal storage
        defaults = st.session_state.get("neg_cols", [])
        if not isinstance(defaults, list):
            defaults = []
        defaults = [c for c in defaults if c in neg_candidates]  # ensure ⊆ options
        if not defaults:
            defaults = [c for c in neg_candidates if name_regex.search(c)]
            st.session_state["neg_cols"] = defaults

        # 5) Render widget using a DIFFERENT key to avoid state conflicts
        neg_cols = st.multiselect(
            "Columns where negatives are invalid (to be set → NaN)",
            options=neg_candidates,
            default=defaults,
            key="neg_cols_sel"  # different from internal storage key
        )

        # 6) Sync widget selection back to internal key (safe)
        st.session_state["neg_cols"] = list(neg_cols)

    else:
        neg_cols = []

    st.divider()
    run_btn = st.button("🚀 Run Cleaning (Preview)", key="btn_run_pipeline")

    # Return placeholders
    last_df: pd.DataFrame = st.session_state.get("pipeline_preview_df", source)
    last_log: List[str] = st.session_state.get("pipeline_preview_log", [])

    # ---------- Helpers ----------
    def log(msg: str, op_log: List[str]):
        op_log.append(msg)

    def iqr_bounds(series: pd.Series, k: float = 3.0) -> Tuple[float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return (float("-inf"), float("inf"))
        return (q1 - k * iqr, q3 + k * iqr)

    # Group-aware categorical imputation helper
    def fill_by_group_mode(df_in: pd.DataFrame, target: str, by: str) -> pd.DataFrame:
        if target not in df_in.columns or by not in df_in.columns:
            return df_in
        group_mode = df_in.groupby(by)[target].apply(
            lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else np.nan
        )
        def mapper(row):
            if pd.isna(row[target]):
                gm = group_mode.get(row[by], np.nan)
                return gm if pd.notna(gm) else row[target]
            return row[target]
        df_out = df_in.copy()
        df_out[target] = df_out.apply(mapper, axis=1)
        # Global fallback
        if df_out[target].isna().any():
            m = df_out[target].mode(dropna=True)
            df_out[target] = df_out[target].fillna(m.iloc[0] if not m.empty else "Unknown")
        return df_out

    # =================
    # RUN THE PIPELINE
    # =================
    if run_btn:
        cleaned = source.copy()
        op_log: List[str] = []
        outlier_report = {}

        # 1) Categorical standardization
        if do_cat_std:
            text_cols = [c for c in cleaned.columns if is_string_dtype(cleaned[c]) or cleaned[c].dtype.name == "category"]
            changes = 0
            for c in text_cols:
                s = cleaned[c]
                s_notna = s.notna()
                s_str = s.astype(str)
                s_new = s_str.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
                mask = s_notna & (s_str != s_new)
                if mask.any():
                    cleaned.loc[mask, c] = s_new[mask]
                    changes += int(mask.sum())
            log(f"Categorical standardization: trimmed/normalized {changes} cell(s) across {len(text_cols)} column(s).", op_log)

        # 2) Date parsing (≥ 0.60)
        if do_date_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_datetime(s, errors="coerce", utc=False)
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= DATE_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Date parsing: converted columns (acceptance≥{:.0%}): {}."
                    .format(DATE_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Date parsing: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 3) Numeric conversion (≥ 0.70)
        if do_num_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_numeric(s.str.replace(",", ""), errors="coerce")
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= NUM_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Numeric conversion: converted columns (acceptance≥{:.0%}): {}."
                    .format(NUM_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Numeric conversion: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 4) Negative → NaN
        if do_neg_to_nan:
            selected_neg_cols = st.session_state.get("neg_cols", [])
            replaced = 0
            for c in selected_neg_cols:
                if c in cleaned.columns and is_numeric_dtype(cleaned[c]):
                    mask = cleaned[c] < 0
                    cnt = int(mask.sum())
                    if cnt > 0:
                        cleaned.loc[mask, c] = np.nan
                        replaced += cnt
            log(f"Negative→NaN: replaced {replaced} negative value(s) across {len(selected_neg_cols)} column(s): {selected_neg_cols}.", op_log)

        # 5) Outlier detection (3×IQR) — flag only
        if do_outlier_flag:
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    s = cleaned[c].dropna()
                    if len(s) < 4:
                        continue
                    low, high = iqr_bounds(s, k=3.0)
                    mask = (cleaned[c] < low) | (cleaned[c] > high)
                    count = int(mask.sum())
                    if count > 0:
                        outlier_report[c] = {"lower": float(low), "upper": float(high), "count": count}
            if outlier_report:
                total = sum(v["count"] for v in outlier_report.values())
                log(f"Outlier detection (3×IQR): flagged {total} value(s) across {len(outlier_report)} numeric column(s).", op_log)
            else:
                log("Outlier detection (3×IQR): no outliers flagged.", op_log)

        # 6) Median imputation (numeric)
        if do_num_median:
            imputed_cells = 0
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    na_before = int(cleaned[c].isna().sum())
                    if na_before > 0:
                        med = cleaned[c].median()
                        cleaned[c] = cleaned[c].fillna(med)
                        na_after = int(cleaned[c].isna().sum())
                        imputed_cells += (na_before - na_after)
            log(f"Median imputation (numeric): filled {imputed_cells} missing numeric cell(s).", op_log)

        # 6b) Categorical imputation
        cat_cols = [c for c in cleaned.columns if (cleaned[c].dtype == "object" or str(cleaned[c].dtype) == "category")]
        if cat_strategy == "Constant: 'Unknown'":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (constant 'Unknown'): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Mode per column":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    mode_vals = cleaned[c].mode(dropna=True)
                    if not mode_vals.empty:
                        cleaned[c] = cleaned[c].fillna(mode_vals.iloc[0])
                    else:
                        cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (mode/Unknown fallback): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Group-aware (recommended)":
            # Example group-aware policy for penguins-like schema
            if set(["species", "island"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="island", by="species")
            if set(["species", "sex"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="sex", by="species")
            # Global safety net
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (group-aware + 'Unknown' fallback): filled {filled_cells_cat} cell(s).", op_log)

        else:
            log("Categorical imputation: None (left as NA).", op_log)

        # 7) Duplicate removal — full-row, keep='first'
        rows_pre = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").copy()
        removed = rows_pre - len(cleaned)
        log(f"Duplicate removal: full-row; keep='first'. Rows removed: {removed}.", op_log)

        # ----- Preview -----
        st.success(
            f"Preview complete — rows before: {rows_before:,}, after: {len(cleaned):,}. "
            f"Columns: {cols_before} → {cleaned.shape[1]}."
        )
        st.markdown("**Head (5) of cleaned preview**")
        st.dataframe(pretty_index_df(cleaned.head(), "R.No."), use_container_width=True)

        # Persist preview for Apply step
        st.session_state["pipeline_preview_df"]  = cleaned.copy()
        st.session_state["pipeline_preview_log"] = op_log.copy()

        last_df  = cleaned
        last_log = op_log

    # --- Apply & Save (always visible after a preview) ---
    if st.session_state.get("pipeline_preview_df") is not None:
        if st.button("✅ Apply & Save to Session (Pipeline)", key="apply_pipeline_trigger"):
            # Target path: clean_<original>.csv
            base_name  = os.path.basename(selected_file)
            dir_name   = os.path.dirname(os.path.abspath(selected_file)) or os.path.abspath(".")
            clean_name = f"clean_{base_name}"
            clean_path = os.path.join(dir_name, clean_name)

            file_exists = os.path.exists(clean_path)

            if file_exists:
                st.session_state["show_confirm_overwrite"] = True
                st.warning(
                    "You are about to **overwrite** the existing cleaned file:\n\n"
                    f"**{clean_path}**\n\n"
                    "This action cannot be undone. Proceed?"
                )
                c1, c2 = st.columns(2)
                proceed = c1.button("🛑 Overwrite now", type="primary", key="btn_overwrite_confirm")
                cancel  = c2.button("Cancel", key="btn_overwrite_cancel")

                if proceed:
                    cleaned = st.session_state["pipeline_preview_df"].copy()
                    op_log  = st.session_state.get("pipeline_preview_log", [])

                    # Save to session
                    st.session_state["clean_df_pipeline"] = cleaned
                    st.session_state.setdefault("cleaning_log", [])
                    st.session_state["cleaning_log"].extend(op_log)

                    try:
                        cleaned.to_csv(clean_path, index=False)
                        st.success(f"Overwritten: {clean_path}")
                    except Exception as e:
                        st.error(f"Failed to write cleaned CSV: {e}")
                    finally:
                        st.session_state["show_confirm_overwrite"] = False
                        st.session_state.pop("pipeline_preview_df", None)
                        st.session_state.pop("pipeline_preview_log", None)
                        # Keep the saved file selected after rerun (if your sidebar reads this)
                        st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                        st.rerun()

                if cancel:
                    st.session_state["show_confirm_overwrite"] = False
                    st.info("Overwrite canceled.")
            else:
                # First-time create: save directly (no confirmation)
                cleaned = st.session_state["pipeline_preview_df"].copy()
                op_log  = st.session_state.get("pipeline_preview_log", [])

                st.session_state["clean_df_pipeline"] = cleaned
                st.session_state.setdefault("cleaning_log", [])
                st.session_state["cleaning_log"].extend(op_log)

                try:
                    cleaned.to_csv(clean_path, index=False)
                    st.success(f"Saved cleaned CSV: {clean_path}")
                except Exception as e:
                    st.error(f"Failed to write cleaned CSV: {e}")
                finally:
                    st.session_state.pop("pipeline_preview_df", None)
                    st.session_state.pop("pipeline_preview_log", None)
                    st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                    st.rerun()
    else:
        st.info("Run **Cleaning (Preview)** first to enable Apply & Save.")

    # Initial hint if nothing has run
    if not run_btn and st.session_state.get("pipeline_preview_df") is None:
        st.info(f"Ready. Current dataset shape: {rows_before:,} × {cols_before}. Configure options and click **Run Cleaning**.")

    return last_df, last_log
    

def render_clean_pipeline_ooo(df: pd.DataFrame, selected_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean: Pipeline (All-in-One) — simplified UI (no downloads, write to clean_<original>.csv)

    Steps:
      1) Categorical standardization (trim + lowercase)
      2) Date parsing (acceptance ≥ 0.60)
      3) Numeric conversion (acceptance ≥ 0.70)
      4) Negative values → NaN (auto-detect + user selection)
      5) Outlier detection (3×IQR) — flag only (no modification)
      6) Median imputation for numeric NaNs
      6b) Categorical imputation (selectable: None / Constant 'Unknown' / Mode / Group-aware + fallback)
      7) Duplicate removal (full-row, keep='first')

    Behavior:
      - No download buttons.
      - Apply & Save writes to <same folder>/clean_<original_name>.csv
        * If file exists → asks overwrite confirmation; else creates silently.
      - After saving, the app reruns (st.rerun()) and preselects the saved file (if you implemented preselect).
    """
    st.subheader("🛠️ Clean: Pipeline (All-in-One)")

    # Start from last saved pipeline output if present; else from incoming df
    source = st.session_state.get("clean_df_pipeline", df).copy()
    rows_before, cols_before = source.shape

    # -------------------------
    # Pipeline Settings (2 rows)
    # -------------------------
    st.markdown("### ⚙️ Pipeline Settings")

    def checkbox_tile(container, key: str, label: str, default: bool = True, caption: Optional[str] = None) -> bool:
        val = container.checkbox(label, value=default, key=key)
        if caption:
            container.caption(caption)
        return val

    row1 = st.columns(3, gap="large")
    do_cat_std    = checkbox_tile(row1[0], "do_cat_std",    "Categorical standardization (trim + lower)", True)
    do_date_parse = checkbox_tile(row1[1], "do_date_parse", "Date parsing (accept ≥60%)", True)
    do_num_parse  = checkbox_tile(row1[2], "do_num_parse",  "Numeric conversion (accept ≥70%)", True)

    row2 = st.columns(3, gap="large")
    do_neg_to_nan   = checkbox_tile(row2[0], "do_neg_to_nan",   "Negative values → NaN (selected columns)", True)
    do_outlier_flag = checkbox_tile(row2[1], "do_outlier_flag", "Outlier detection (3×IQR) — flag only", True)
    do_num_median   = checkbox_tile(row2[2], "do_num_median",   "Median imputation (numeric)", True)

    # NEW: Categorical missing strategy selector
    cat_strategy = st.selectbox(
        "Categorical missing strategy",
        ["None (leave as NA)", "Constant: 'Unknown'", "Mode per column", "Group-aware (recommended)"],
        index=3,
        key="cat_missing_strategy"
    )

    st.divider()
    st.info(
        "### Defaults used by the pipeline\n"
        "- **Duplicate removal:** full-row deduplication with `keep='first'` (applied at the end).\n"
        "- **Numeric missing values:** **Median** imputation (robust to skew/outliers).\n"
        "- **Categorical/Text missing values:** controlled by the selector above."
    )
    st.divider()

    # Hardcoded thresholds
    DATE_ACCEPT_THR = 0.60
    NUM_ACCEPT_THR  = 0.70

    # -------------------------------
    # Negative → NaN (UI only if ON)
    # -------------------------------
    if do_neg_to_nan:
        st.markdown("#### 🚫 Negative Values → NaN")

        # Candidate numeric columns
        neg_candidates = [c for c in source.columns if is_numeric_dtype(source[c])]

        # Name-based patterns
        name_patterns = [
            r"\bweight\b",
            r"\bheight\b",
            r"\bwait[_\s-]*time\b",
            r"\bduration\b",
            r"\bage\b",
            r"\bbmi\b",
            r"(?:\bquantity\b|\bqty\b)",
            r"\bcount\b",
            r"(?:\bprice\b|\bamount\b|\bcost\b)",
            r"\bdistance\b",
        ]
        name_regex = re.compile("|".join(name_patterns), flags=re.IGNORECASE)

        # Reset the stored selection when the selected dataset changes (prevents stale defaults)
        current_file = os.path.abspath(selected_file)
        last_file = st.session_state.get("neg_cols_file")
        if last_file != current_file:
            st.session_state["neg_cols"] = [c for c in neg_candidates if name_regex.search(c)]
            st.session_state["neg_cols_file"] = current_file

        # Defaults: prefer session value; fall back to name-based guess
        default_neg_cols = st.session_state.get("neg_cols", [c for c in neg_candidates if name_regex.search(c)])
        # Safety: ensure defaults ⊆ options
        default_neg_cols = [c for c in default_neg_cols if c in neg_candidates]
        st.session_state["neg_cols"] = default_neg_cols

        neg_cols = st.multiselect(
            "Columns where negatives are invalid (to be set → NaN)",
            options=neg_candidates,
            default=default_neg_cols,
            key="neg_cols"
        )

        # Auto-detect (optional)
        if st.button("⚙️ Auto-detect non-negative columns", key="btn_autodetect_neg_cols"):
            def autodetect_nonnegative_cols(df_in: pd.DataFrame, candidates: List[str]) -> List[str]:
                suggest = set()
                # name-based
                for c in candidates:
                    if name_regex.search(c):
                        suggest.add(c)
                # distribution-based
                for c in candidates:
                    s = df_in[c]
                    s = pd.to_numeric(s, errors="coerce") if not is_numeric_dtype(s) else s
                    s = s.dropna()
                    if len(s) < 20:
                        continue
                    neg_ratio = (s < 0).mean()
                    nonneg_ratio = (s >= 0).mean()
                    median_nonneg = (s.median() >= 0)
                    if nonneg_ratio >= 0.90 and neg_ratio > 0 and median_nonneg:
                        suggest.add(c)
                return [c for c in candidates if c in suggest]

            st.session_state["neg_cols"] = autodetect_nonnegative_cols(source, neg_candidates)
            st.success(f"Auto-detected: {st.session_state['neg_cols']}")
    else:
        neg_cols = []

    st.divider()
    run_btn = st.button("🚀 Run Cleaning (Preview)", key="btn_run_pipeline")

    # Return placeholders
    last_df: pd.DataFrame = st.session_state.get("pipeline_preview_df", source)
    last_log: List[str] = st.session_state.get("pipeline_preview_log", [])

    # ---------- Helpers ----------
    def log(msg: str, op_log: List[str]):
        op_log.append(msg)

    def iqr_bounds(series: pd.Series, k: float = 3.0) -> Tuple[float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return (float("-inf"), float("inf"))
        return (q1 - k * iqr, q3 + k * iqr)

    # Small helper for group-aware categorical imputation
    def fill_by_group_mode(df_in: pd.DataFrame, target: str, by: str) -> pd.DataFrame:
        if target not in df_in.columns or by not in df_in.columns:
            return df_in
        group_mode = df_in.groupby(by)[target].apply(
            lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else np.nan
        )
        def mapper(row):
            if pd.isna(row[target]):
                gm = group_mode.get(row[by], np.nan)
                return gm if pd.notna(gm) else row[target]
            return row[target]
        df_out = df_in.copy()
        df_out[target] = df_out.apply(mapper, axis=1)
        # Global fallback for any remaining NAs in this target
        if df_out[target].isna().any():
            m = df_out[target].mode(dropna=True)
            df_out[target] = df_out[target].fillna(m.iloc[0] if not m.empty else "Unknown")
        return df_out

    # =================
    # RUN THE PIPELINE
    # =================
    if run_btn:
        cleaned = source.copy()
        op_log: List[str] = []
        outlier_report = {}

        # 1) Categorical standardization
        if do_cat_std:
            text_cols = [c for c in cleaned.columns if is_string_dtype(cleaned[c]) or cleaned[c].dtype.name == "category"]
            changes = 0
            for c in text_cols:
                s = cleaned[c]
                s_notna = s.notna()
                s_str = s.astype(str)
                s_new = s_str.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
                mask = s_notna & (s_str != s_new)
                if mask.any():
                    cleaned.loc[mask, c] = s_new[mask]
                    changes += int(mask.sum())
            log(f"Categorical standardization: trimmed/normalized {changes} cell(s) across {len(text_cols)} column(s).", op_log)

        # 2) Date parsing (≥ 0.60)
        if do_date_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_datetime(s, errors="coerce", utc=False)
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= DATE_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Date parsing: converted columns (acceptance≥{:.0%}): {}."
                    .format(DATE_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Date parsing: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 3) Numeric conversion (≥ 0.70)
        if do_num_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_numeric(s.str.replace(",", ""), errors="coerce")
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= NUM_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Numeric conversion: converted columns (acceptance≥{:.0%}): {}."
                    .format(NUM_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Numeric conversion: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 4) Negative → NaN
        if do_neg_to_nan and st.session_state.get("neg_cols"):
            replaced = 0
            for c in st.session_state["neg_cols"]:
                if c in cleaned.columns and is_numeric_dtype(cleaned[c]):
                    mask = cleaned[c] < 0
                    cnt = int(mask.sum())
                    if cnt > 0:
                        cleaned.loc[mask, c] = np.nan
                        replaced += cnt
            log(f"Negative→NaN: replaced {replaced} negative value(s) across {len(st.session_state['neg_cols'])} column(s): {st.session_state['neg_cols']}.", op_log)

        # 5) Outlier detection (3×IQR) — flag only
        if do_outlier_flag:
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    s = cleaned[c].dropna()
                    if len(s) < 4:
                        continue
                    low, high = iqr_bounds(s, k=3.0)
                    mask = (cleaned[c] < low) | (cleaned[c] > high)
                    count = int(mask.sum())
                    if count > 0:
                        outlier_report[c] = {"lower": float(low), "upper": float(high), "count": count}
            if outlier_report:
                total = sum(v["count"] for v in outlier_report.values())
                log(f"Outlier detection (3×IQR): flagged {total} value(s) across {len(outlier_report)} numeric column(s).", op_log)
            else:
                log("Outlier detection (3×IQR): no outliers flagged.", op_log)

        # 6) Median imputation (numeric)
        if do_num_median:
            imputed_cells = 0
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    na_before = int(cleaned[c].isna().sum())
                    if na_before > 0:
                        med = cleaned[c].median()
                        cleaned[c] = cleaned[c].fillna(med)
                        na_after = int(cleaned[c].isna().sum())
                        imputed_cells += (na_before - na_after)
            log(f"Median imputation (numeric): filled {imputed_cells} missing numeric cell(s).", op_log)

        # 6b) Categorical imputation (based on cat_strategy)
        cat_cols = [c for c in cleaned.columns if (cleaned[c].dtype == "object" or str(cleaned[c].dtype) == "category")]
        if cat_strategy == "Constant: 'Unknown'":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (constant 'Unknown'): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Mode per column":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    mode_vals = cleaned[c].mode(dropna=True)
                    if not mode_vals.empty:
                        val = mode_vals.iloc[0]
                        cleaned[c] = cleaned[c].fillna(val)
                    else:
                        cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (mode/Unknown fallback): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Group-aware (recommended)":
            # Example policy: impute 'island' by mode within 'species', and 'sex' by mode within 'species'
            # (Adjust pairs as per your schema)
            if set(["species", "island"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="island", by="species")
            if set(["species", "sex"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="sex", by="species")

            # Global pass: fill any remaining categorical NAs with 'Unknown' as a safety net
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (group-aware + 'Unknown' fallback): filled {filled_cells_cat} cell(s).", op_log)

        else:
            # "None (leave as NA)": do nothing
            log("Categorical imputation: None (left as NA).", op_log)

        # 7) Duplicate removal — full-row, keep='first'
        rows_pre = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").copy()
        removed = rows_pre - len(cleaned)
        log(f"Duplicate removal: full-row; keep='first'. Rows removed: {removed}.", op_log)

        # ----- Preview UI -----
        st.success(
            f"Preview complete — rows before: {rows_before:,}, after: {len(cleaned):,}. "
            f"Columns: {cols_before} → {cleaned.shape[1]}."
        )
        st.markdown("**Head (5) of cleaned preview**")
        st.dataframe(pretty_index_df(cleaned.head(), "R.No."), use_container_width=True)

        # Persist preview for Apply step (survives rerun)
        st.session_state["pipeline_preview_df"]  = cleaned.copy()
        st.session_state["pipeline_preview_log"] = op_log.copy()

        last_df  = cleaned
        last_log = op_log

    # --- Apply & Save (always visible after a preview) ---
    if st.session_state.get("pipeline_preview_df") is not None:
        if st.button("✅ Apply & Save to Session (Pipeline)", key="apply_pipeline_trigger"):
            # Build the prefixed target path
            base_name  = os.path.basename(selected_file)
            dir_name   = os.path.dirname(os.path.abspath(selected_file)) or os.path.abspath(".")
            clean_name = f"clean_{base_name}"
            clean_path = os.path.join(dir_name, clean_name)

            file_exists = os.path.exists(clean_path)

            if file_exists:
                st.session_state["show_confirm_overwrite"] = True
                st.warning(
                    "You are about to **overwrite** the existing cleaned file:\n\n"
                    f"**{clean_path}**\n\n"
                    "This action cannot be undone. Proceed?"
                )
                c1, c2 = st.columns(2)
                proceed = c1.button("🛑 Overwrite now", type="primary", key="btn_overwrite_confirm")
                cancel  = c2.button("Cancel", key="btn_overwrite_cancel")

                if proceed:
                    cleaned = st.session_state["pipeline_preview_df"].copy()
                    op_log  = st.session_state.get("pipeline_preview_log", [])

                    # Save to session
                    st.session_state["clean_df_pipeline"] = cleaned
                    st.session_state.setdefault("cleaning_log", [])
                    st.session_state["cleaning_log"].extend(op_log)

                    # Overwrite the existing clean_ file
                    try:
                        cleaned.to_csv(clean_path, index=False)
                        st.success(f"Overwritten: {clean_path}")
                    except Exception as e:
                        st.error(f"Failed to write cleaned CSV: {e}")
                    finally:
                        st.session_state["show_confirm_overwrite"] = False
                        st.session_state.pop("pipeline_preview_df", None)
                        st.session_state.pop("pipeline_preview_log", None)
                        # Keep this file selected after rerun (if you implemented preselect logic)
                        st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                        st.rerun()

                if cancel:
                    st.session_state["show_confirm_overwrite"] = False
                    st.info("Overwrite canceled.")
            else:
                # First-time create: save directly, no confirmation
                cleaned = st.session_state["pipeline_preview_df"].copy()
                op_log  = st.session_state.get("pipeline_preview_log", [])

                st.session_state["clean_df_pipeline"] = cleaned
                st.session_state.setdefault("cleaning_log", [])
                st.session_state["cleaning_log"].extend(op_log)

                try:
                    cleaned.to_csv(clean_path, index=False)
                    st.success(f"Saved cleaned CSV: {clean_path}")
                except Exception as e:
                    st.error(f"Failed to write cleaned CSV: {e}")
                finally:
                    st.session_state.pop("pipeline_preview_df", None)
                    st.session_state.pop("pipeline_preview_log", None)
                    st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                    st.rerun()
    else:
        st.info("Run **Cleaning (Preview)** first to enable Apply & Save.")

    # Initial hint if nothing has run
    if not run_btn and st.session_state.get("pipeline_preview_df") is None:
        st.info(f"Ready. Current dataset shape: {rows_before:,} × {cols_before}. Configure options and click **Run Cleaning**.")

    return last_df, last_log
    
    
def render_clean_pipeline_org(df: pd.DataFrame, selected_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean: Pipeline (All-in-One) — simplified UI (no downloads, overwrite with confirmation)

    Steps:
      1) Categorical standardization (trim + lowercase)
      2) Date parsing (acceptance ≥ 0.60)
      3) Numeric conversion (acceptance ≥ 0.70)
      4) Negative values → NaN (auto-detect + user selection)
      5) Outlier detection (3×IQR) — flag only (no modification)
      6) Median imputation for numeric NaNs
      7) Duplicate removal (full-row, keep='first')

    Behavior:
      - No download buttons.
      - Apply & Save shows a confirmation prompt and writes to clean_<original>.csv
        (overwrites only if that file already exists).
      - After save, the app reruns so the "Select Dataset" reflects the updated file.
      - Minimal, robust fix to prevent default-not-in-options error in Negative → NaN.
    """
    st.subheader("🛠️ Clean: Pipeline (All-in-One)")

    # Use last saved pipeline DF if present; else use incoming df
    source = st.session_state.get("clean_df_pipeline", df).copy()
    rows_before, cols_before = source.shape

    # -------------------------
    # Pipeline Settings (2 rows)
    # -------------------------
    st.markdown("### ⚙️ Pipeline Settings")

    def checkbox_tile(container, key: str, label: str, default: bool = True, caption: Optional[str] = None) -> bool:
        val = container.checkbox(label, value=default, key=key)
        if caption:
            container.caption(caption)
        return val

    row1 = st.columns(3, gap="large")
    do_cat_std    = checkbox_tile(row1[0], "do_cat_std",    "Categorical standardization (trim + lower)", True)
    do_date_parse = checkbox_tile(row1[1], "do_date_parse", "Date parsing (accept ≥60%)", True)
    do_num_parse  = checkbox_tile(row1[2], "do_num_parse",  "Numeric conversion (accept ≥70%)", True)

    row2 = st.columns(3, gap="large")
    do_neg_to_nan   = checkbox_tile(row2[0], "do_neg_to_nan",   "Negative values → NaN (selected columns)", True)
    do_outlier_flag = checkbox_tile(row2[1], "do_outlier_flag", "Outlier detection (3×IQR) — flag only", True)
    do_num_median   = checkbox_tile(row2[2], "do_num_median",   "Median imputation (numeric)", True)

    st.divider()
    st.info(
        "### Defaults used by the pipeline\n"
        "- **Duplicate removal:** full-row deduplication with `keep='first'` (applied at the end).\n"
        "- **Numeric missing values:** **Median** imputation (robust to skew/outliers).\n"
        "- **Categorical/Text missing values:** **No automatic imputation**; only label standardization (trim/lower)."
    )
    st.divider()

    # Hardcoded thresholds
    DATE_ACCEPT_THR = 0.60
    NUM_ACCEPT_THR  = 0.70

    # -------------------------------
    # Negative → NaN (UI only if ON)
    # -------------------------------
    if do_neg_to_nan:
        st.markdown("#### 🚫 Negative Values → NaN")

        # Numeric columns available right now
        neg_candidates = [c for c in source.columns if is_numeric_dtype(source[c])]

        # Name-based patterns
        name_patterns = [
            r"\bweight\b",
            r"\bheight\b",
            r"\bwait[_\s-]*time\b",
            r"\bduration\b",
            r"\bage\b",
            r"\bbmi\b",
            r"(?:\bquantity\b|\bqty\b)",
            r"\bcount\b",
            r"(?:\bprice\b|\bamount\b|\bcost\b)",
            r"\bdistance\b",
        ]
        name_regex = re.compile("|".join(name_patterns), flags=re.IGNORECASE)

        # --- Minimal, effective fix ---
        # Reset the stored selection when the selected dataset changes.
        current_file = os.path.abspath(selected_file)
        last_file = st.session_state.get("neg_cols_file")
        if last_file != current_file:
            st.session_state["neg_cols"] = [c for c in neg_candidates if name_regex.search(c)]
            st.session_state["neg_cols_file"] = current_file

        # Build defaults: prefer session value; fall back to name-based guess
        default_neg_cols = st.session_state.get("neg_cols", [c for c in neg_candidates if name_regex.search(c)])

        # SAFETY: ensure defaults ⊆ options (prevents StreamlitAPIException)
        if not isinstance(default_neg_cols, list):
            default_neg_cols = []
        default_neg_cols = [c for c in default_neg_cols if c in neg_candidates]

        # Edge case: no numeric columns
        if not neg_candidates:
            st.info("No numeric columns available for Negative → NaN.")
            neg_cols = []
        else:
            # Keep session consistent and render widget
            st.session_state["neg_cols"] = default_neg_cols
            neg_cols = st.multiselect(
                "Columns where negatives are invalid (to be set → NaN)",
                options=neg_candidates,
                default=default_neg_cols,
                key="neg_cols"
            )

        # Auto-detect button (unchanged)
        if st.button("⚙️ Auto-detect non-negative columns", key="btn_autodetect_neg_cols"):
            def autodetect_nonnegative_cols(df_in: pd.DataFrame, candidates: List[str]) -> List[str]:
                suggest = set()
                # (a) Name-based
                for c in candidates:
                    if name_regex.search(c):
                        suggest.add(c)
                # (b) Distribution-based
                for c in candidates:
                    s = df_in[c]
                    s = pd.to_numeric(s, errors="coerce") if not is_numeric_dtype(s) else s
                    s = s.dropna()
                    if len(s) < 20:
                        continue
                    neg_ratio = (s < 0).mean()
                    nonneg_ratio = (s >= 0).mean()
                    median_nonneg = (s.median() >= 0)
                    if nonneg_ratio >= 0.90 and neg_ratio > 0 and median_nonneg:
                        suggest.add(c)
                return [c for c in candidates if c in suggest]

            new_sel = autodetect_nonnegative_cols(source, neg_candidates)
            st.session_state["neg_cols"] = new_sel
            st.success(f"Auto-detected: {new_sel}")
    else:
        neg_cols = []

    st.divider()
    run_btn = st.button("🚀 Run Cleaning (Preview)", key="btn_run_pipeline")

    # Return placeholders
    last_df: pd.DataFrame = st.session_state.get("pipeline_preview_df", source)
    last_log: List[str] = st.session_state.get("pipeline_preview_log", [])

    # ---------- Helpers ----------
    def log(msg: str, op_log: List[str]):
        op_log.append(msg)

    def iqr_bounds(series: pd.Series, k: float = 3.0) -> Tuple[float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return (float("-inf"), float("inf"))
        return (q1 - k * iqr, q3 + k * iqr)

    # =================
    # RUN THE PIPELINE
    # =================
    if run_btn:
        cleaned = source.copy()
        op_log: List[str] = []

        # 1) Categorical standardization
        if do_cat_std:
            text_cols = [c for c in cleaned.columns if is_string_dtype(cleaned[c]) or cleaned[c].dtype.name == "category"]
            changes = 0
            for c in text_cols:
                s = cleaned[c]
                s_notna = s.notna()
                s_str = s.astype(str)
                s_new = s_str.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
                mask = s_notna & (s_str != s_new)
                if mask.any():
                    cleaned.loc[mask, c] = s_new[mask]
                    changes += int(mask.sum())
            log(f"Categorical standardization: trimmed/normalized {changes} cell(s) across {len(text_cols)} column(s).", op_log)

        # 2) Date parsing (≥ 0.60)
        if do_date_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_datetime(s, errors="coerce", utc=False)
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= DATE_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Date parsing: converted columns (acceptance≥{:.0%}): {}."
                    .format(DATE_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Date parsing: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 3) Numeric conversion (≥ 0.70)
        if do_num_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_numeric(s.str.replace(",", ""), errors="coerce")
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= NUM_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Numeric conversion: converted columns (acceptance≥{:.0%}): {}."
                    .format(NUM_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Numeric conversion: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 4) Negative → NaN
        if do_neg_to_nan and neg_cols:
            replaced = 0
            for c in neg_cols:
                if is_numeric_dtype(cleaned[c]):
                    mask = cleaned[c] < 0
                    cnt = int(mask.sum())
                    if cnt > 0:
                        cleaned.loc[mask, c] = np.nan
                        replaced += cnt
            log(f"Negative→NaN: replaced {replaced} negative value(s) across {len(neg_cols)} column(s): {neg_cols}.", op_log)

        # 5) Outlier detection (3×IQR) — flag only
        if do_outlier_flag:
            outlier_report = {}
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    s = cleaned[c].dropna()
                    if len(s) < 4:
                        continue
                    low, high = iqr_bounds(s, k=3.0)
                    mask = (cleaned[c] < low) | (cleaned[c] > high)
                    count = int(mask.sum())
                    if count > 0:
                        outlier_report[c] = {"lower": float(low), "upper": float(high), "count": count}
            if outlier_report:
                total = sum(v["count"] for v in outlier_report.values())
                log(f"Outlier detection (3×IQR): flagged {total} value(s) across {len(outlier_report)} numeric column(s).", op_log)
            else:
                log("Outlier detection (3×IQR): no outliers flagged.", op_log)

        # 6) Median imputation
        if do_num_median:
            imputed_cells = 0
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    na_before = int(cleaned[c].isna().sum())
                    if na_before > 0:
                        med = cleaned[c].median()
                        cleaned[c] = cleaned[c].fillna(med)
                        na_after = int(cleaned[c].isna().sum())
                        imputed_cells += (na_before - na_after)
            log(f"Median imputation: filled {imputed_cells} missing numeric cell(s).", op_log)

        # 7) Duplicate removal
        rows_pre = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").copy()
        removed = rows_pre - len(cleaned)
        log(f"Duplicate removal: full-row; keep='first'. Rows removed: {removed}.", op_log)

        # ----- Preview UI -----
        st.success(
            f"Preview complete — rows before: {rows_before:,}, after: {len(cleaned):,}. "
            f"Columns: {cols_before} → {cleaned.shape[1]}."
        )
        st.markdown("**Head (5) of cleaned preview**")
        st.dataframe(pretty_index_df(cleaned.head(), "R.No."), use_container_width=True)

        # Persist preview for Apply step (survives rerun)
        st.session_state["pipeline_preview_df"]  = cleaned.copy()
        st.session_state["pipeline_preview_log"] = op_log.copy()

        last_df  = cleaned
        last_log = op_log

    # --- Apply & Save (always visible after a preview) ---
    if st.session_state.get("pipeline_preview_df") is not None:
        if st.button("✅ Apply & Save to Session (Pipeline)", key="apply_pipeline_trigger"):
            # Build the prefixed target path
            base_name  = os.path.basename(selected_file)
            dir_name   = os.path.dirname(os.path.abspath(selected_file)) or os.path.abspath(".")
            clean_name = f"clean_{base_name}"
            clean_path = os.path.join(dir_name, clean_name)

            file_exists = os.path.exists(clean_path)

            # Overwrite confirmation only if file already exists
            if file_exists:
                st.session_state["show_confirm_overwrite"] = True
                st.warning(
                    "You are about to **overwrite** the existing cleaned file:\n\n"
                    f"**{clean_path}**\n\n"
                    "This action cannot be undone. Proceed?"
                )
                c1, c2 = st.columns(2)
                proceed = c1.button("🛑 Overwrite now", type="primary", key="btn_overwrite_confirm")
                cancel  = c2.button("Cancel", key="btn_overwrite_cancel")

                if proceed:
                    cleaned = st.session_state["pipeline_preview_df"].copy()
                    op_log  = st.session_state.get("pipeline_preview_log", [])

                    # Save to session
                    st.session_state["clean_df_pipeline"] = cleaned
                    st.session_state.setdefault("cleaning_log", [])
                    st.session_state["cleaning_log"].extend(op_log)

                    # Overwrite the existing clean_ file
                    try:
                        cleaned.to_csv(clean_path, index=False)
                        st.success(f"Overwritten: {clean_path}")
                    except Exception as e:
                        st.error(f"Failed to write cleaned CSV: {e}")
                    finally:
                        st.session_state["show_confirm_overwrite"] = False
                        st.session_state.pop("pipeline_preview_df", None)
                        st.session_state.pop("pipeline_preview_log", None)
                        # Keep this file selected after rerun (if you implemented preselect logic)
                        st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                        st.rerun()

                if cancel:
                    st.session_state["show_confirm_overwrite"] = False
                    st.info("Overwrite canceled.")

            else:
                # First-time create: save directly, no confirmation
                cleaned = st.session_state["pipeline_preview_df"].copy()
                op_log  = st.session_state.get("pipeline_preview_log", [])

                st.session_state["clean_df_pipeline"] = cleaned
                st.session_state.setdefault("cleaning_log", [])
                st.session_state["cleaning_log"].extend(op_log)

                try:
                    cleaned.to_csv(clean_path, index=False)
                    st.success(f"Saved cleaned CSV: {clean_path}")
                except Exception as e:
                    st.error(f"Failed to write cleaned CSV: {e}")
                finally:
                    st.session_state.pop("pipeline_preview_df", None)
                    st.session_state.pop("pipeline_preview_log", None)
                    # Decide which file to preselect after rerun (clean file or original)
                    st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                    st.rerun()

    else:
        st.info("Run **Cleaning (Preview)** first to enable Apply & Save.")

    # Initial hint if nothing has run
    if not run_btn and st.session_state.get("pipeline_preview_df") is None:
        st.info(f"Ready. Current dataset shape: {rows_before:,} × {cols_before}. Configure options and click **Run Cleaning**.")

    return last_df, last_log
    
    
def render_clean_pipeline_old(df: pd.DataFrame, selected_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean: Pipeline (All-in-One) — simplified UI (no downloads, overwrite with confirmation)

    Steps:
      1) Categorical standardization (trim + lowercase)
      2) Date parsing (acceptance ≥ 0.60)
      3) Numeric conversion (acceptance ≥ 0.70)
      4) Negative values → NaN (auto-detect + user selection)
      5) Outlier detection (3×IQR) — flag only (no modification)
      6) Median imputation for numeric NaNs
      7) Duplicate removal (full-row, keep='first')

    Behavior:
      - No download buttons.
      - Apply & Save shows a confirmation prompt and overwrites the original CSV.
      - After overwrite, the app reruns so the "Select Dataset" reflects the updated file.
      - No debug captions.
    """
    st.subheader("🛠️ Clean: Pipeline (All-in-One)")

    # Use last saved pipeline DF if present; else use incoming df
    source = st.session_state.get("clean_df_pipeline", df).copy()
    rows_before, cols_before = source.shape

    # -------------------------
    # Pipeline Settings (2 rows)
    # -------------------------
    st.markdown("### ⚙️ Pipeline Settings")

    def checkbox_tile(container, key: str, label: str, default: bool = True, caption: Optional[str] = None) -> bool:
        val = container.checkbox(label, value=default, key=key)
        if caption:
            container.caption(caption)
        return val

    row1 = st.columns(3, gap="large")
    do_cat_std    = checkbox_tile(row1[0], "do_cat_std",    "Categorical standardization (trim + lower)", True)
    do_date_parse = checkbox_tile(row1[1], "do_date_parse", "Date parsing (accept ≥60%)", True)
    do_num_parse  = checkbox_tile(row1[2], "do_num_parse",  "Numeric conversion (accept ≥70%)", True)

    row2 = st.columns(3, gap="large")
    do_neg_to_nan   = checkbox_tile(row2[0], "do_neg_to_nan",   "Negative values → NaN (selected columns)", True)
    do_outlier_flag = checkbox_tile(row2[1], "do_outlier_flag", "Outlier detection (3×IQR) — flag only", True)
    do_num_median   = checkbox_tile(row2[2], "do_num_median",   "Median imputation (numeric)", True)

    st.divider()
    st.info(
        "### Defaults used by the pipeline\n"
        "- **Duplicate removal:** full-row deduplication with `keep='first'` (applied at the end).\n"
        "- **Numeric missing values:** **Median** imputation (robust to skew/outliers).\n"
        "- **Categorical/Text missing values:** **No automatic imputation**; only label standardization (trim/lower)."
    )
    st.divider()

    # Hardcoded thresholds
    DATE_ACCEPT_THR = 0.60
    NUM_ACCEPT_THR  = 0.70

    # -------------------------------
    # Negative → NaN (UI only if ON)
    # -------------------------------
    if do_neg_to_nan:
        st.markdown("#### 🚫 Negative Values → NaN")

        neg_candidates = [c for c in source.columns if is_numeric_dtype(source[c])]

        name_patterns = [
            r"\bweight\b",
            r"\bheight\b",
            r"\bwait[_\s-]*time\b",
            r"\bduration\b",
            r"\bage\b",
            r"\bbmi\b",
            r"(?:\bquantity\b|\bqty\b)",
            r"\bcount\b",
            r"(?:\bprice\b|\bamount\b|\bcost\b)",
            r"\bdistance\b",
        ]
        name_regex = re.compile("|".join(name_patterns), flags=re.IGNORECASE)

        def autodetect_nonnegative_cols(df_in: pd.DataFrame, candidates: List[str]) -> List[str]:
            suggest = set()
            # (a) Name-based
            for c in candidates:
                if name_regex.search(c):
                    suggest.add(c)
            # (b) Distribution-based
            for c in candidates:
                s = df_in[c]
                s = pd.to_numeric(s, errors="coerce") if not is_numeric_dtype(s) else s
                s = s.dropna()
                if len(s) < 20:
                    continue
                neg_ratio = (s < 0).mean()
                nonneg_ratio = (s >= 0).mean()
                median_nonneg = (s.median() >= 0)
                if nonneg_ratio >= 0.90 and neg_ratio > 0 and median_nonneg:
                    suggest.add(c)
            return [c for c in candidates if c in suggest]

        if st.button("⚙️ Auto-detect non-negative columns", key="btn_autodetect_neg_cols"):
            st.session_state["neg_cols"] = autodetect_nonnegative_cols(source, neg_candidates)
            st.success(f"Auto-detected: {st.session_state['neg_cols']}")

        default_neg_cols = st.session_state.get("neg_cols", [c for c in neg_candidates if name_regex.search(c)])
        neg_cols = st.multiselect(
            "Columns where negatives are invalid (to be set → NaN)",
            options=neg_candidates,
            default=default_neg_cols,
            key="neg_cols"
        )
    else:
        neg_cols = []

    st.divider()
    run_btn = st.button("🚀 Run Cleaning (Preview)", key="btn_run_pipeline")

    # Return placeholders
    last_df: pd.DataFrame = st.session_state.get("pipeline_preview_df", source)
    last_log: List[str] = st.session_state.get("pipeline_preview_log", [])

    # ---------- Helpers ----------
    def log(msg: str, op_log: List[str]):
        op_log.append(msg)

    def iqr_bounds(series: pd.Series, k: float = 3.0) -> Tuple[float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return (float("-inf"), float("inf"))
        return (q1 - k * iqr, q3 + k * iqr)

    # =================
    # RUN THE PIPELINE
    # =================
    if run_btn:
        cleaned = source.copy()
        op_log: List[str] = []

        # 1) Categorical standardization
        if do_cat_std:
            text_cols = [c for c in cleaned.columns if is_string_dtype(cleaned[c]) or cleaned[c].dtype.name == "category"]
            changes = 0
            for c in text_cols:
                s = cleaned[c]
                s_notna = s.notna()
                s_str = s.astype(str)
                s_new = s_str.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
                mask = s_notna & (s_str != s_new)
                if mask.any():
                    cleaned.loc[mask, c] = s_new[mask]
                    changes += int(mask.sum())
            log(f"Categorical standardization: trimmed/normalized {changes} cell(s) across {len(text_cols)} column(s).", op_log)

        # 2) Date parsing (≥ 0.60)
        if do_date_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_datetime(s, errors="coerce", utc=False)
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= DATE_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Date parsing: converted columns (acceptance≥{:.0%}): {}."
                    .format(DATE_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Date parsing: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 3) Numeric conversion (≥ 0.70)
        if do_num_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_numeric(s.str.replace(",", ""), errors="coerce")
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= NUM_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log("Numeric conversion: converted columns (acceptance≥{:.0%}): {}."
                    .format(NUM_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])), op_log)
            if skipped_cols:
                log("Numeric conversion: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])), op_log)

        # 4) Negative → NaN
        if do_neg_to_nan and neg_cols:
            replaced = 0
            for c in neg_cols:
                if is_numeric_dtype(cleaned[c]):
                    mask = cleaned[c] < 0
                    cnt = int(mask.sum())
                    if cnt > 0:
                        cleaned.loc[mask, c] = np.nan
                        replaced += cnt
            log(f"Negative→NaN: replaced {replaced} negative value(s) across {len(neg_cols)} column(s): {neg_cols}.", op_log)

        # 5) Outlier detection (3×IQR) — flag only
        if do_outlier_flag:
            outlier_report = {}
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    s = cleaned[c].dropna()
                    if len(s) < 4:
                        continue
                    low, high = iqr_bounds(s, k=3.0)
                    mask = (cleaned[c] < low) | (cleaned[c] > high)
                    count = int(mask.sum())
                    if count > 0:
                        outlier_report[c] = {"lower": float(low), "upper": float(high), "count": count}
            if outlier_report:
                total = sum(v["count"] for v in outlier_report.values())
                log(f"Outlier detection (3×IQR): flagged {total} value(s) across {len(outlier_report)} numeric column(s).", op_log)
            else:
                log("Outlier detection (3×IQR): no outliers flagged.", op_log)

        # 6) Median imputation
        if do_num_median:
            imputed_cells = 0
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    na_before = int(cleaned[c].isna().sum())
                    if na_before > 0:
                        med = cleaned[c].median()
                        cleaned[c] = cleaned[c].fillna(med)
                        na_after = int(cleaned[c].isna().sum())
                        imputed_cells += (na_before - na_after)
            log(f"Median imputation: filled {imputed_cells} missing numeric cell(s).", op_log)

        # 7) Duplicate removal
        rows_pre = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").copy()
        removed = rows_pre - len(cleaned)
        log(f"Duplicate removal: full-row; keep='first'. Rows removed: {removed}.", op_log)

        # ----- Preview UI -----
        st.success(
            f"Preview complete — rows before: {rows_before:,}, after: {len(cleaned):,}. "
            f"Columns: {cols_before} → {cleaned.shape[1]}."
        )
        st.markdown("**Head (5) of cleaned preview**")
        st.dataframe(pretty_index_df(cleaned.head(), "R.No."), use_container_width=True)

        # Persist preview for Apply step (survives rerun)
        st.session_state["pipeline_preview_df"]  = cleaned.copy()
        st.session_state["pipeline_preview_log"] = op_log.copy()

        last_df  = cleaned
        last_log = op_log

    # --- Apply & Save (always visible after a preview) ---
    if st.session_state.get("pipeline_preview_df") is not None:
        if st.button("✅ Apply & Save to Session (Pipeline)", key="apply_pipeline_trigger"):
            # Build the prefixed target path
            base_name  = os.path.basename(selected_file)
            dir_name   = os.path.dirname(os.path.abspath(selected_file)) or os.path.abspath(".")
            clean_name = f"clean_{base_name}"
            clean_path = os.path.join(dir_name, clean_name)

            file_exists = os.path.exists(clean_path)

            # Show the appropriate confirmation (overwrite only if file already exists)
            if file_exists:
                st.session_state["show_confirm_overwrite"] = True
                st.warning(
                    "You are about to **overwrite** the existing cleaned file:\n\n"
                    f"**{clean_path}**\n\n"
                    "This action cannot be undone. Proceed?"
                )
                c1, c2 = st.columns(2)
                proceed = c1.button("🛑 Overwrite now", type="primary", key="btn_overwrite_confirm")
                cancel  = c2.button("Cancel", key="btn_overwrite_cancel")

                if proceed:
                    cleaned = st.session_state["pipeline_preview_df"].copy()
                    op_log  = st.session_state.get("pipeline_preview_log", [])

                    # Save to session
                    st.session_state["clean_df_pipeline"] = cleaned
                    st.session_state.setdefault("cleaning_log", [])
                    st.session_state["cleaning_log"].extend(op_log)

                    # Overwrite the existing clean_ file
                    try:
                        cleaned.to_csv(clean_path, index=False)
                        st.success(f"Overwritten: {clean_path}")
                    except Exception as e:
                        st.error(f"Failed to write cleaned CSV: {e}")
                    finally:
                        st.session_state["show_confirm_overwrite"] = False
                        st.session_state.pop("pipeline_preview_df", None)
                        st.session_state.pop("pipeline_preview_log", None)
                        # After successfully saving to clean_path and before st.rerun():
                        st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)       ####
                        st.rerun()

                if cancel:
                    st.session_state["show_confirm_overwrite"] = False
                    st.info("Overwrite canceled.")

            else:
                # --- First-time create (no overwrite). Choose ONE of the two behaviors below ---

                # (A) Create silently without confirmation (simple UX):
                cleaned = st.session_state["pipeline_preview_df"].copy()
                op_log  = st.session_state.get("pipeline_preview_log", [])

                st.session_state["clean_df_pipeline"] = cleaned
                st.session_state.setdefault("cleaning_log", [])
                st.session_state["cleaning_log"].extend(op_log)

                try:
                    cleaned.to_csv(clean_path, index=False)
                    st.success(f"Saved cleaned CSV: {clean_path}")
                except Exception as e:
                    st.error(f"Failed to write cleaned CSV: {e}")
                finally:
                    st.session_state.pop("pipeline_preview_df", None)
                    st.session_state.pop("pipeline_preview_log", None)
                    st.session_state["selected_file_preselect"] = os.path.abspath(selected_file)    ####
                    st.rerun()

    else:
        st.info("Run **Cleaning (Preview)** first to enable Apply & Save.")
                
    # Initial hint if nothing has run
    if not run_btn and st.session_state.get("pipeline_preview_df") is None:
        st.info(f"Ready. Current dataset shape: {rows_before:,} × {cols_before}. Configure options and click **Run Cleaning**.")

    return last_df, last_log
    
    

# ================================
# 4) Main
# ================================
def main() -> None:
    setup_page()

    # --- Sidebar controls ---
    selected_file, selected_numeric_col, bins = sidebar_controls()

    # --- Load the selected dataset (RAW for EDA tabs) ---
    df = load_data(selected_file)

    # --- Tabs layout ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📄 Overview",
            "📈 Distribution",
            "🔎 Data Quality",
            "📊 Correlation",
            "📦 Outliers",
            "🔗 Pair Plot",
            "🛠️ Clean: Pipeline (All-in-One)",
        ]
    )

    with tab1:
        render_overview(df)
    with tab2:
        render_distribution(df, selected_numeric_col, bins)
    with tab3:
        render_data_quality(df)
    with tab4:
        render_correlation(df)
    with tab5:
        render_outliers(df)
    with tab6:
        render_pairplot(df)
    with tab7:
        # pass selected_file so the pipeline can save to the same folder with prefix 'clean_'
        _cleaned_pipeline, _log_pipeline = render_clean_pipeline(df, selected_file)


if __name__ == "__main__":
    main()
    