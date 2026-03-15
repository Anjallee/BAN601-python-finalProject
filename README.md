# Interactive Data Profiling Dashboard (Streamlit)

A fast, practical EDA and cleaning tool for CSV files.  
It lets you explore distributions, correlations, outliers, and run a one-click **Clean: Pipeline** with sensible defaults and **selectable categorical imputation**.

---

## Features

- **Multi‑CSV selection** from the current working directory
- **EDA tabs**: Overview, Distribution, Data Quality, Correlation, Outliers, Pair Plot
- **Clean: Pipeline (All‑in‑One)**
  - Categorical standardization (trim/whitespace collapse/lowercase)
  - Date parsing (acceptance threshold)
  - Numeric parsing (acceptance threshold)
  - **Negative values → NaN** (auto‑detect + user selection)
  - Outlier flagging (3×IQR) – report only
  - **Numeric median imputation**
  - **Categorical imputation strategy (selectable):**
    - None (leave as NA)
    - Constant: `'Unknown'`
    - Mode per column
    - Group‑aware (mode within a grouping, then `'Unknown'` fallback)
  - Duplicate removal (full-row, `keep='first'`)
- **Save cleaned data** as `clean_<original>.csv` in the **same folder**
  - First save: creates the file
  - Next saves: shows **overwrite confirmation**
- **State-aware sidebar** (optional): you can keep the just‑saved file selected after rerun

---

## Quick Start

1. **Install** dependencies (Streamlit, pandas, numpy, seaborn, matplotlib, scipy).  
2. **Place your CSVs** in the app’s working directory.  
3. **Run** the app:
   ```bash
   streamlit run Full_EDA_project.py
   
   
