"""
MoO3 Interface Intelligence Platform
A complete AI-powered battery material discovery dashboard.
Works with any raw Excel/CSV dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoO3 Interface Intelligence Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; }
    .main-header p  { color: #a8d8ea; margin: 0.5rem 0 0; font-size: 1.1rem; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0.3rem 0 0; opacity: 0.85; }

    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #3498db;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f8f0;
        border-left: 4px solid #27ae60;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────────────────────

def extract_numeric(value):
    """Extract the first numeric value from a messy string."""
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", "")
    # grab the first float-like token
    matches = re.findall(r"[-+]?\d*\.?\d+", s)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return np.nan
    return np.nan


def smart_load(uploaded_file):
    """Load Excel or CSV; try multiple engines; return DataFrame + sheet info."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding_errors="replace")
        return df, ["CSV"]
    # Excel
    for engine in ["openpyxl", "xlrd"]:
        try:
            uploaded_file.seek(0)
            xl = pd.ExcelFile(uploaded_file, engine=engine)
            sheets = xl.sheet_names
            dfs = []
            for sh in sheets:
                tmp = pd.read_excel(xl, sheet_name=sh, header=None)
                # find the real header row (first row with ≥3 non-null cells)
                header_row = 0
                for i, row in tmp.iterrows():
                    if row.notna().sum() >= 3:
                        header_row = i
                        break
                tmp.columns = tmp.iloc[header_row]
                tmp = tmp.iloc[header_row + 1:].reset_index(drop=True)
                tmp.columns = [str(c).strip() for c in tmp.columns]
                dfs.append(tmp)
            combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            return combined, sheets
        except Exception:
            continue
    raise ValueError("Could not read the uploaded file. Please upload a valid .xlsx or .csv.")


def clean_dataframe(df):
    """Standardise column names, forward-fill sparse rows, coerce numeric columns."""
    # Drop completely empty rows/cols
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]
    # Forward-fill the Material / Composite columns (sparse Excel tables)
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ["material", "composite", "method", "morphology", "structure"]):
            df[col] = df[col].replace("", np.nan).replace("Unknown", np.nan).ffill()
    # Coerce numeric columns
    numeric_hints = ["capacity", "size", "cycle", "rate", "c rate", "pcent", "pce",
                     "surface area", "crystallite", "thickness", "diameter", "length",
                     "width", "voltage", "current", "density"]
    for col in df.columns:
        lc = col.lower()
        if any(h in lc for h in numeric_hints):
            df[col + " [num]"] = df[col].apply(extract_numeric)
    df = df.fillna("Unknown")
    return df


def get_numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def get_all_numeric(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def encode_for_ml(df, target_col):
    """Label-encode categoricals and impute; return X, y, encoders, feature_names."""
    working = df.copy()
    encoders = {}

    # Encode categorical columns
    for col in working.columns:
        if col == target_col:
            continue
        if working[col].dtype == object:
            le = LabelEncoder()
            working[col] = le.fit_transform(working[col].astype(str))
            encoders[col] = le

    # Clean target column
    y_raw = working[target_col].apply(extract_numeric)
    mask = y_raw.notna()
    working = working[mask]
    y = y_raw[mask].values

    # Prepare features
    features = [c for c in working.columns if c != target_col]
    X = working[features].apply(pd.to_numeric, errors="coerce")

    # ✅ Drop empty + useless columns
    X = X.dropna(axis=1, how='all')
    X = X.loc[:, X.nunique() > 1]

    # Update features
    features = X.columns.tolist()

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Final safety
    if X_imp.shape[1] != len(features):
        features = [f"feature_{i}" for i in range(X_imp.shape[1])]

    X_df = pd.DataFrame(X_imp, columns=features)

    return X_df, y, encoders, features, imputer
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔋 MoO3 Platform")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset (.xlsx or .csv)",
    type=["xlsx", "xls", "csv"],
    help="Upload your literature survey or any material dataset."
)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dataset", "📈 Analytics", "🤖 Train Model",
     "⚡ Prediction", "🏆 Recommendations", "📖 About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project:** Interface-sensitive ML model for MoO₃-based heterostructures  \n"
    "**Institute:** BMSIT&M, Bengaluru"
)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df_raw = None
df = None

if uploaded_file:
    try:
        df_raw, sheets = smart_load(uploaded_file)
        df = clean_dataframe(df_raw.copy())
        st.sidebar.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} cols")
    except Exception as e:
        st.sidebar.error(f"❌ Load error: {e}")

# ── HOME PAGE ────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🔋 MoO₃ Interface Intelligence Platform</h1>
        <p>AI-Powered Battery Material Discovery Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><h3>📊</h3><p>Dataset Analytics</p></div>""",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><h3>🤖</h3><p>ML Prediction</p></div>""",
                    unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><h3>🏆</h3><p>Recommendations</p></div>""",
                    unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><h3>🔍</h3><p>Feature Analysis</p></div>""",
                    unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 What this platform does")
        st.markdown("""
- **Ingests** any raw Excel / CSV material dataset automatically
- **Cleans** messy data — extracts numbers from mixed-text cells
- **Visualises** material distributions, capacity trends, correlations
- **Trains** Random Forest, Gradient Boosting, and Ridge models
- **Predicts** battery capacity for new material configurations
- **Recommends** top-performing materials from the dataset
        """)
    with col_b:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
1. Upload your `.xlsx` or `.csv` dataset using the sidebar
2. Explore the **Dataset** tab to verify your data
3. Go to **Analytics** for automatic visualisations
4. Train a model in **Train Model**
5. Make predictions in **Prediction**
6. Discover top materials in **Recommendations**
        """)

    if df is not None:
        st.markdown("---")
        st.markdown("### 📌 Dataset Snapshot")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rows", len(df))
        r2.metric("Columns", len(df.columns))
        mat_col = next((c for c in df.columns if "material" in c.lower()), None)
        r3.metric("Unique Materials", df[mat_col].nunique() if mat_col else "N/A")
        num_cols = get_all_numeric(df)
        r4.metric("Numeric Features", len(num_cols))
    else:
        st.markdown("""
        <div class="info-box">
        👈 <strong>Upload a dataset from the sidebar to begin!</strong>
        Works with the MoO₃ literature survey Excel or any battery material spreadsheet.
        </div>
        """, unsafe_allow_html=True)

# ── DATASET PAGE ─────────────────────────────────────────────────────────────
elif page == "📊 Dataset":
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Total Columns", len(df.columns))
    num_cols = get_all_numeric(df)
    c3.metric("Numeric Columns", len(num_cols))
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    c4.metric("Text Columns", len(obj_cols))

    st.markdown("#### 🔎 Columns Detected")
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—"
        col_info.append({"Column": col, "Type": dtype,
                         "Unique Values": n_unique, "Sample": sample[:60]})
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)

    st.markdown("#### 📋 Raw Data")
    page_size = st.slider("Rows per page", 10, 100, 25)
    page_num = st.number_input("Page", min_value=1,
                               max_value=max(1, len(df) // page_size + 1), value=1)
    start = (page_num - 1) * page_size
    st.dataframe(df.iloc[start: start + page_size], use_container_width=True)

    st.markdown("#### 📥 Download Cleaned Data")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download as CSV", csv_buf.getvalue(),
                       "cleaned_data.csv", "text/csv")

# ── ANALYTICS PAGE ───────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.markdown('<div class="section-header">📈 Dataset Analytics</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    tabs = st.tabs(["📦 Materials", "📉 Capacity", "🔗 Correlations", "🗂 Distributions"])

    # ── Tab 1: Materials ──────────────────────────────────────────────────
    with tabs[0]:
        mat_cols = [c for c in df.columns if any(k in c.lower()
                    for k in ["material", "composite", "morphology", "structure", "method"])]
        if mat_cols:
            sel = st.selectbox("Select categorical column", mat_cols)
            vc = df[sel].replace("Unknown", np.nan).dropna().value_counts().head(15)
            if len(vc):
                fig = px.bar(x=vc.index, y=vc.values,
                             labels={"x": sel, "y": "Count"},
                             title=f"Top {len(vc)} — {sel}",
                             color=vc.values,
                             color_continuous_scale="Blues")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.pie(values=vc.values, names=vc.index,
                              title=f"Distribution — {sel}")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No obvious categorical columns detected.")

    # ── Tab 2: Capacity ───────────────────────────────────────────────────
    with tabs[1]:
        num_cols = get_all_numeric(df)
        cap_cols = [c for c in num_cols if "capacity" in c.lower()]
        if not cap_cols:
            cap_cols = num_cols  # fallback to all numeric

        if cap_cols:
            sel2 = st.selectbox("Select numeric column", cap_cols)
            vals = pd.to_numeric(df[sel2], errors="coerce").dropna()
            if len(vals):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Mean", f"{vals.mean():.1f}")
                col_b.metric("Max", f"{vals.max():.1f}")
                col_c.metric("Std Dev", f"{vals.std():.1f}")

                fig = px.histogram(vals, nbins=30,
                                   title=f"Distribution of {sel2}",
                                   labels={"value": sel2})
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure(go.Box(y=vals, name=sel2,
                                        boxpoints="outliers",
                                        marker_color="steelblue"))
                fig2.update_layout(title=f"Box Plot — {sel2}")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numeric columns detected.")

    # ── Tab 3: Correlations ───────────────────────────────────────────────
    with tabs[2]:
        num_df = df[get_all_numeric(df)].apply(pd.to_numeric, errors="coerce")
        num_df = num_df.dropna(axis=1, thresh=3)
        if len(num_df.columns) >= 2:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            title="Feature Correlation Matrix",
                            aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📊 Scatter Plot")
            cols_for_scatter = num_df.columns.tolist()
            if len(cols_for_scatter) >= 2:
                sx = st.selectbox("X axis", cols_for_scatter, index=0)
                sy = st.selectbox("Y axis", cols_for_scatter,
                                  index=min(1, len(cols_for_scatter) - 1))
                mat_col = next((c for c in df.columns if "material" in c.lower()), None)
                fig3 = px.scatter(
                    x=pd.to_numeric(df[sx], errors="coerce"),
                    y=pd.to_numeric(df[sy], errors="coerce"),
                    color=df[mat_col] if mat_col else None,
                    labels={"x": sx, "y": sy},
                    title=f"{sx} vs {sy}",
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── Tab 4: Distributions ──────────────────────────────────────────────
    with tabs[3]:
        num_cols2 = get_all_numeric(df)
        if num_cols2:
            selected_multi = st.multiselect("Select columns to visualise",
                                            num_cols2, default=num_cols2[:3])
            for col in selected_multi:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) > 1:
                    fig = px.violin(y=vals, box=True, points="all",
                                    title=f"{col}", labels={"y": col})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available.")

# ── TRAIN MODEL PAGE ─────────────────────────────────────────────────────────
elif page == "🤖 Train Model":
    st.markdown('<div class="section-header">🤖 Train Machine Learning Model</div>',
                unsafe_allow_html=True)

    if df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    num_cols = get_all_numeric(df)
    if not num_cols:
        st.error("No numeric columns found. The cleaner will add '[num]' columns automatically "
                 "when it detects capacity/size hints. Check your dataset in the Dataset tab.")
        st.stop()

    st.markdown("#### ⚙️ Model Configuration")
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        target_col = st.selectbox("🎯 Target column (what to predict)", num_cols,
                                  index=0)
        model_choice = st.selectbox("🧠 Algorithm",
                                    ["Random Forest", "Gradient Boosting", "Ridge Regression"])

    with col_cfg2:
        test_size = st.slider("Test split %", 10, 40, 20) / 100
        n_folds = st.slider("Cross-validation folds", 2, 10, 5)

    if model_choice == "Random Forest":
        n_est = st.slider("Number of trees", 50, 500, 200, step=50)
        max_depth = st.select_slider("Max depth", [None, 5, 10, 15, 20], value=None)
    elif model_choice == "Gradient Boosting":
        n_est = st.slider("Number of estimators", 50, 300, 100, step=50)
        lr = st.slider("Learning rate", 0.01, 0.3, 0.1, step=0.01)
    else:
        alpha = st.slider("Regularisation (alpha)", 0.01, 100.0, 1.0)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Encoding data and training…"):
            try:
                X, y, encoders, features, imputer = encode_for_ml(df, target_col)

                if X.shape[1] == 0:
                    st.error("❌ No valid features available after preprocessing.")
                    st.stop()

                if len(y) < 5:
                    st.error(f"❌ Only {len(y)} valid rows in target. Need at least 5.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                if model_choice == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=n_est, max_depth=max_depth, random_state=42, n_jobs=-1
                    )
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=n_est, learning_rate=lr, random_state=42
                    )
                else:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    model = Ridge(alpha=alpha)
                    st.session_state["scaler"] = scaler

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                # Cross validation
                cv_model = (RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
                            if model_choice == "Random Forest" else model)
                kf = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=42)
                cv_scores = cross_val_score(cv_model, X, y, cv=kf, scoring="r2")

                st.session_state.update({
                    "model": model, "encoders": encoders, "features": features,
                    "imputer": imputer, "target_col": target_col,
                    "model_name": model_choice,
                    "metrics": {"R²": round(r2, 4), "MAE": round(mae, 2),
                                "RMSE": round(rmse, 2), "CV R²": round(cv_scores.mean(), 4)},
                })

                st.markdown('<div class="success-box">✅ Training complete!</div>',
                            unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{r2:.4f}")
                m2.metric("MAE", f"{mae:.2f}")
                m3.metric("RMSE", f"{rmse:.2f}")
                m4.metric(f"CV R² ({n_folds}-fold)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                # Actual vs Predicted
                fig_pred = px.scatter(
                    x=y_test, y=preds,
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Actual vs Predicted",
                    
                )
                fig_pred.add_shape(type="line",
                                   x0=y_test.min(), x1=y_test.max(),
                                   y0=y_test.min(), y1=y_test.max(),
                                   line=dict(dash="dash", color="red"))
                st.plotly_chart(fig_pred, use_container_width=True)

                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame({
                        "Feature": features,
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=False).head(20)

                    fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                                   orientation="h", title="Top 20 Feature Importances",
                                   color="Importance", color_continuous_scale="Viridis")
                    fig_fi.update_layout(yaxis={"autorange": "reversed"})
                    st.plotly_chart(fig_fi, use_container_width=True)

                    st.markdown("#### 🔑 Top 5 Most Important Features")
                    for i, row in fi_df.head(5).iterrows():
                        st.markdown(f"- **{row['Feature']}** — {row['Importance']:.4f}")

                # Residuals
                residuals = y_test - preds
                fig_res = px.histogram(residuals, nbins=30,
                                       title="Residual Distribution",
                                       labels={"value": "Residual"})
                st.plotly_chart(fig_res, use_container_width=True)

            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── PREDICTION PAGE ──────────────────────────────────────────────────────────
elif page == "⚡ Prediction":
    st.markdown('<div class="section-header">⚡ Capacity Prediction</div>', unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Train Model' page.")
        st.stop()

    model = st.session_state["model"]
    encoders = st.session_state["encoders"]
    features = st.session_state["features"]
    imputer = st.session_state["imputer"]
    target_col = st.session_state["target_col"]
    metrics = st.session_state["metrics"]

    st.markdown(f"""
    <div class="info-box">
    🤖 Active model: <strong>{st.session_state['model_name']}</strong> &nbsp;|&nbsp;
    Target: <strong>{target_col}</strong> &nbsp;|&nbsp;
    R² = <strong>{metrics['R²']}</strong> &nbsp;|&nbsp;
    MAE = <strong>{metrics['MAE']}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎛️ Set Input Features")
    st.markdown("Adjust each feature below, then click **Predict**.")

    input_dict = {}
    cols_per_row = 3
    feature_chunks = [features[i:i+cols_per_row]
                      for i in range(0, len(features), cols_per_row)]

    for chunk in feature_chunks:
        row_cols = st.columns(len(chunk))
        for col_widget, feat in zip(row_cols, chunk):
            with col_widget:
                if feat in encoders:
                    options = encoders[feat].classes_.tolist()
                    chosen = st.selectbox(feat, options, key=f"pred_{feat}")
                    input_dict[feat] = encoders[feat].transform([chosen])[0]
                else:
                    # Use median of training data as default
                    if df is not None and feat in df.columns:
                        med_val = pd.to_numeric(df[feat], errors="coerce").median()
                        default = float(med_val) if not np.isnan(med_val) else 0.0
                    else:
                        default = 0.0
                    input_dict[feat] = st.number_input(feat, value=default, key=f"pred_{feat}")

    if st.button("🔮 Predict", type="primary"):
        try:
            inp = pd.DataFrame([input_dict])[features]
            inp = inp.apply(pd.to_numeric, errors="coerce")
            inp_imp = pd.DataFrame(imputer.transform(inp), columns=features)

            if "scaler" in st.session_state:
                inp_imp = pd.DataFrame(
                    st.session_state["scaler"].transform(inp_imp), columns=features
                )

            prediction = model.predict(inp_imp)[0]

            st.markdown(f"""
            <div class="success-box">
            <h3>🎯 Predicted {target_col}: <strong>{prediction:.2f}</strong></h3>
            </div>
            """, unsafe_allow_html=True)

            # Confidence interval via forest variance
            if hasattr(model, "estimators_"):
                tree_preds = [est.predict(inp_imp)[0] for est in model.estimators_]
                ci_low = np.percentile(tree_preds, 5)
                ci_high = np.percentile(tree_preds, 95)
                st.info(f"90% Confidence Interval: **{ci_low:.2f} – {ci_high:.2f}**")

                fig_dist = px.histogram(tree_preds, nbins=30,
                                        title="Prediction Distribution (Tree Ensemble)",
                                        labels={"value": target_col})
                fig_dist.add_vline(x=prediction, line_dash="dash",
                                   annotation_text="Prediction")
                st.plotly_chart(fig_dist, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── RECOMMENDATIONS PAGE ─────────────────────────────────────────────────────
elif page == "🏆 Recommendations":
    st.markdown('<div class="section-header">🏆 Material Recommendation Engine</div>',
                unsafe_allow_html=True)

    if df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    num_cols = get_all_numeric(df)
    if not num_cols:
        st.error("No numeric columns to rank by.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        target_metric = st.selectbox("📊 Rank by", num_cols)
    with col2:
        top_n = st.slider("Top N materials", 5, 30, 10)
    with col3:
        asc = st.checkbox("Sort ascending (lower = better)", value=False)

    temp_df = df.copy()
    temp_df["_rank_val"] = pd.to_numeric(temp_df[target_metric], errors="coerce")
    top_df = (temp_df.dropna(subset=["_rank_val"])
              .sort_values("_rank_val", ascending=asc)
              .head(top_n)
              .drop(columns=["_rank_val"]))

    st.markdown(f"#### 🥇 Top {top_n} Materials by {target_metric}")
    st.dataframe(top_df, use_container_width=True)

    # Visual
    mat_col = next((c for c in df.columns if "material" in c.lower()), df.columns[0])
    comp_col = next((c for c in df.columns if "composite" in c.lower()), None)

    y_label = comp_col if comp_col else mat_col
    vals = pd.to_numeric(top_df[target_metric], errors="coerce")
    labels = top_df[y_label].astype(str).str[:40]

    fig = px.bar(
        x=vals,
        y=labels,
        orientation="h",
        title=f"Top {top_n} by {target_metric}",
        labels={"x": target_metric, "y": y_label},
        color=vals,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig, use_container_width=True)

    # Download
    csv_buf = io.StringIO()
    top_df.to_csv(csv_buf, index=False)
    st.download_button("📥 Download Top Materials", csv_buf.getvalue(),
                       "top_materials.csv", "text/csv")

    # Multi-metric radar (if enough numeric cols)
    numeric_for_radar = [c for c in get_all_numeric(df) if c != target_metric][:5]
    if len(numeric_for_radar) >= 3:
        st.markdown("#### 🕸️ Radar Comparison (Top 5)")
        top5 = top_df.head(5)
        fig_radar = go.Figure()
        for _, row in top5.iterrows():
            vals_r = [extract_numeric(row.get(c, 0)) or 0 for c in numeric_for_radar]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_r,
                theta=numeric_for_radar,
                fill="toself",
                name=str(row.get(y_label, ""))[:25],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Multi-feature Radar — Top 5 Materials",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── ABOUT PAGE ───────────────────────────────────────────────────────────────
elif page == "📖 About":
    st.markdown('<div class="section-header">📖 About the Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
## Project Title
**Interface-sensitive machine learning model for predicting electrochemical
performance of MoO₃-based heterostructures**

## Background
MoO₃ (Molybdenum Trioxide) is a promising anode material for lithium-ion batteries
owing to its high theoretical capacity and layered crystal structure. However, pristine
MoO₃ suffers from poor conductivity and large volume changes. Heterostructures with
MXenes, graphene, and carbon matrices overcome these limitations — but their
interface behaviour is poorly understood.

## Research Gap
Existing ML models focus on:
- Compositional and structural bulk descriptors
- Individual material properties

They **ignore**:
- Binding energy
- Lattice mismatch
- Charge transfer resistance
- Interface chemistry and surface terminations

## Our Approach
This platform builds an **interface-sensitive ML framework** that:
1. Integrates experimental data, literature surveys, and DFT descriptors
2. Extracts and engineers interface-specific features
3. Trains ensemble ML models (Random Forest, Gradient Boosting, Neural Networks)
4. Predicts specific capacity, rate capability, and cycle life
5. Recommends optimal heterostructure configurations

## SDG Alignment
- **SDG 7** — Affordable and Clean Energy
- **SDG 9** — Industry, Innovation, and Infrastructure

## Future Work
- XGBoost and Neural Network integration
- SHAP explainability module
- Interface descriptor augmentation via DFT
- Active learning loop for experimental validation
        """)

    with col2:
        st.markdown("""
### 🏫 Institution
**BMS Institute of Technology & Management**  
Yelahanka, Bengaluru — 560064

### 📚 Department
Artificial Intelligence & Machine Learning

### 🔬 Key Technologies
- Python 3.11+
- Streamlit
- scikit-learn
- Plotly
- Pandas / NumPy

### 📐 Algorithms Used
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression
- K-Fold Cross Validation

### 📏 Metrics
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
        """)

    st.markdown("---")
    st.markdown("### 📚 Key References")
    refs = [
        "Z. Wei, Q. He, and Y. Zhao, *Machine learning for battery research*, J. Power Sources, 2022.",
        "T. Sun et al., *Identifying MOFs for electrochemical energy storage via DFT and ML*, npj Comput. Mater., 2025.",
        "S. Chen et al., *ML-guided construction of MoS₂/MoO₃ heterostructures*, J. Colloid Interface Sci., 2026.",
        "H. Zhang et al., *Phase-engineered MoO₃/MoO₂ heterostructures for Zn²⁺/H⁺ storage*, J. Colloid Interface Sci., 2025.",
        "J. Ding et al., *Fe₂O₃/MoO₃@NG heterostructure for lithium-ion batteries*, ACS Appl. Mater. Interfaces, 2022.",
    ]
    for ref in refs:
        st.markdown(f"- {ref}")
