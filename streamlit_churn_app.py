import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📉",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background-color: #0f1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85;
}

.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    color: #f9fafb;
}

.risk-high { color: #ef4444; }
.risk-medium { color: #f59e0b; }
.risk-low { color: #10b981; }

.strategy-card {
    background: #1a1d27;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

.strategy-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}

.strategy-desc {
    font-size: 0.88rem;
    color: #d1d5db;
    line-height: 1.5;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    margin-top: 1.2rem;
}

.sidebar-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 1rem;
}

div[data-testid="stSidebar"] {
    background-color: #13151f;
    border-right: 1px solid #1f2235;
}

div[data-testid="stSidebar"] label {
    font-size: 0.82rem;
    color: #9ca3af;
    font-weight: 500;
}

.stSelectbox > div > div {
    background-color: #1a1d27;
    border-color: #2a2d3e;
    color: #f9fafb;
}

.stNumberInput > div > div > input {
    background-color: #1a1d27;
    border-color: #2a2d3e;
    color: #f9fafb;
}

.stSlider > div {
    color: #f9fafb;
}

h1 {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #0f1117 !important;
    letter-spacing: -0.02em;
}

h3 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    letter-spacing: 0.01em;
}

.stDivider {
    border-color: #1f2235;
}

p, .stMarkdown {
    color: #9ca3af;
    font-size: 0.88rem;
}

.batch-stat {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.batch-stat-value {
    font-size: 1.8rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    color: #f9fafb;
}

.batch-stat-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("churn_artifacts.pkl")
    return artifacts

artifacts       = load_artifacts()
model           = artifacts["model"]
scaler          = artifacts["scaler"]
threshold       = artifacts["threshold"]
feature_columns = artifacts["feature_columns"]

MONTHLY_CHARGE_MEDIAN = 70.35

# ── Preprocessing (single) ────────────────────────────────────────────────────
def preprocess_input(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["TotalCharges"]  = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48+"], include_lowest=True
    )
    df["avg_monthly_spend"]   = df["TotalCharges"] / (df["tenure"] + 1)
    df["contract_risk"]       = np.where(df["Contract"] == "Month-to-month", 1, 0)

    services = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df["service_count"]       = (df[services] == "Yes").sum(axis=1)
    df["long_term_customer"]  = np.where(df["tenure"] > 24, 1, 0)
    df["high_monthly_charge"] = np.where(df["MonthlyCharges"] > MONTHLY_CHARGE_MEDIAN, 1, 0)
    df["no_security"]         = np.where(df["OnlineSecurity"] == "No", 1, 0)
    df["auto_payment"]        = np.where(df["PaymentMethod"].str.contains("automatic", case=False, na=False), 1, 0)

    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    df_scaled  = scaler.transform(df_encoded)
    return df_scaled, df_encoded

# ── Preprocessing (batch CSV) ─────────────────────────────────────────────────
def preprocess_batch(raw_df: pd.DataFrame):
    df = raw_df.copy()

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    if "Churn" in df.columns:
        df.drop(columns=["Churn"], inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48+"], include_lowest=True
    )
    df["avg_monthly_spend"]   = df["TotalCharges"] / (df["tenure"] + 1)
    df["contract_risk"]       = np.where(df["Contract"] == "Month-to-month", 1, 0)

    services = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df["service_count"]       = (df[services] == "Yes").sum(axis=1)
    df["long_term_customer"]  = np.where(df["tenure"] > 24, 1, 0)
    df["high_monthly_charge"] = np.where(df["MonthlyCharges"] > MONTHLY_CHARGE_MEDIAN, 1, 0)
    df["no_security"]         = np.where(df["OnlineSecurity"] == "No", 1, 0)
    df["auto_payment"]        = np.where(df["PaymentMethod"].str.contains("automatic", case=False, na=False), 1, 0)

    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    df_scaled  = scaler.transform(df_encoded)
    return df_scaled

# ── Retention strategies ──────────────────────────────────────────────────────
def get_strategies(contract, tenure, monthly_charges, online_security, internet_service):
    strategies = []
    if contract == "Month-to-month":
        strategies.append(("Contract Risk", "This customer is on a month-to-month plan — the highest churn risk segment. Offer a 10–15% discount to switch to an annual or two-year contract."))
    if tenure <= 12:
        strategies.append(("New Customer", "Customer has been subscribed for 12 months or less. Proactive onboarding outreach or a welcome offer in the first 90 days can significantly reduce early churn."))
    if monthly_charges > MONTHLY_CHARGE_MEDIAN:
        strategies.append(("High Monthly Charge", "This customer pays above the median monthly charge. Personalized pricing tiers or loyalty discounts may reduce the likelihood of churn."))
    if online_security == "No" and internet_service != "No":
        strategies.append(("No Security Service", "Customer has no online security service. A free 3-month trial of the security add-on can improve engagement and perceived value."))
    if not strategies:
        strategies.append(("Low Risk Profile", "This customer shows a low churn risk profile. Continue regular engagement and service quality monitoring to maintain loyalty."))
    return strategies

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Risk Predictor")
st.markdown("XGBoost · Threshold 0.3 · Recall-optimized · IBM Telco Dataset")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["👤 Individual Prediction", "📂 Batch CSV Prediction"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Individual Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Customer Profile</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
        gender     = st.selectbox("Gender", ["Female", "Male"])
        senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner    = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.markdown('<div class="section-label">Subscription</div>', unsafe_allow_html=True)
        tenure   = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        st.markdown('<div class="section-label">Internet & Services</div>', unsafe_allow_html=True)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        if internet_service != "No":
            online_security   = st.selectbox("Online Security", ["No", "Yes"])
            online_backup     = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support      = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"

        st.markdown('<div class="section-label">Phone</div>', unsafe_allow_html=True)
        phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method    = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
        total_charges   = monthly_charges * max(tenure, 1)

        st.divider()
        predict_btn = st.button("Predict Churn Risk", type="primary")

    if not predict_btn:
        st.markdown("### Example Customer Profiles")
        st.markdown('<p style="color:#6b7280;font-size:0.85rem;margin-bottom:1.5rem">These profiles illustrate how the model identifies churn risk across different customer segments.</p>', unsafe_allow_html=True)

        ex_col1, ex_col2, ex_col3 = st.columns(3, gap="medium")
        with ex_col1:
            st.markdown("""
            <div class="metric-card" style="border-left: 3px solid #ef4444;">
                <div class="metric-label">🔴 High Risk Profile</div>
                <div style="margin-top:0.8rem;font-size:0.85rem;color:#d1d5db;line-height:1.8">
                    <b style="color:#9ca3af">Contract</b><br>Month-to-month<br>
                    <b style="color:#9ca3af">Tenure</b><br>3 months<br>
                    <b style="color:#9ca3af">Monthly Charges</b><br>$95.00<br>
                    <b style="color:#9ca3af">Internet</b><br>Fiber optic<br>
                    <b style="color:#9ca3af">Security</b><br>No<br>
                    <b style="color:#9ca3af">Payment</b><br>Electronic check
                </div>
                <div style="margin-top:1rem;padding:0.5rem 0.8rem;background:#2d1b1b;border-radius:6px;font-size:0.8rem;color:#ef4444;font-weight:600">
                    ⚠️ Likely to Churn
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ex_col2:
            st.markdown("""
            <div class="metric-card" style="border-left: 3px solid #f59e0b;">
                <div class="metric-label">🟡 Medium Risk Profile</div>
                <div style="margin-top:0.8rem;font-size:0.85rem;color:#d1d5db;line-height:1.8">
                    <b style="color:#9ca3af">Contract</b><br>Month-to-month<br>
                    <b style="color:#9ca3af">Tenure</b><br>18 months<br>
                    <b style="color:#9ca3af">Monthly Charges</b><br>$72.00<br>
                    <b style="color:#9ca3af">Internet</b><br>DSL<br>
                    <b style="color:#9ca3af">Security</b><br>Yes<br>
                    <b style="color:#9ca3af">Payment</b><br>Mailed check
                </div>
                <div style="margin-top:1rem;padding:0.5rem 0.8rem;background:#2d2510;border-radius:6px;font-size:0.8rem;color:#f59e0b;font-weight:600">
                    ⚡ Monitor Closely
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ex_col3:
            st.markdown("""
            <div class="metric-card" style="border-left: 3px solid #10b981;">
                <div class="metric-label">🟢 Low Risk Profile</div>
                <div style="margin-top:0.8rem;font-size:0.85rem;color:#d1d5db;line-height:1.8">
                    <b style="color:#9ca3af">Contract</b><br>Two year<br>
                    <b style="color:#9ca3af">Tenure</b><br>48 months<br>
                    <b style="color:#9ca3af">Monthly Charges</b><br>$55.00<br>
                    <b style="color:#9ca3af">Internet</b><br>DSL<br>
                    <b style="color:#9ca3af">Security</b><br>Yes<br>
                    <b style="color:#9ca3af">Payment</b><br>Bank transfer (automatic)
                </div>
                <div style="margin-top:1rem;padding:0.5rem 0.8rem;background:#0f2d1f;border-radius:6px;font-size:0.8rem;color:#10b981;font-weight:600">
                    ✅ Likely to Stay
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style="text-align:center;color:#4b5563;font-size:0.82rem;padding:1rem 0">
            Fill in the customer profile on the left sidebar and click <b style="color:#6366f1">Predict Churn Risk</b> to get started.
        </div>
        """, unsafe_allow_html=True)

    else:
        senior_value = 1 if senior == "Yes" else 0

        input_df = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior_value,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        X_scaled, X_encoded = preprocess_input(input_df)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob >= threshold)

        if prob >= 0.65:
            risk_label = "High Risk";   risk_class = "risk-high";   risk_emoji = "🔴"
        elif prob >= 0.4:
            risk_label = "Medium Risk"; risk_class = "risk-medium"; risk_emoji = "🟡"
        else:
            risk_label = "Low Risk";    risk_class = "risk-low";    risk_emoji = "🟢"

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### Prediction Result")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Churn Probability</div>
                <div class="metric-value {risk_class}">{prob * 100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value" style="font-size:1.4rem">{risk_emoji} {risk_label}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model Decision (threshold = {threshold})</div>
                <div class="metric-value" style="font-size:1.2rem">{"⚠️ Likely to Churn" if pred else "✅ Likely to Stay"}</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(prob))
            st.caption(f"Probability: {prob:.4f}")

            st.markdown("### Retention Strategies")
            strategies = get_strategies(contract, tenure, monthly_charges, online_security, internet_service)
            for title, desc in strategies:
                st.markdown(f"""
                <div class="strategy-card">
                    <div class="strategy-title">{title}</div>
                    <div class="strategy-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        with right:
            st.markdown("### SHAP Feature Impact")
            st.caption("Which features influenced this prediction most")
            try:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_encoded)

                shap_df = pd.DataFrame({
                    "Feature": X_encoded.columns,
                    "SHAP":    shap_values[0]
                }).reindex(
                    pd.Series(shap_values[0]).abs().sort_values(ascending=False).index
                ).head(12).iloc[::-1]

                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor("#1a1d27")
                ax.set_facecolor("#1a1d27")
                colors = ["#ef4444" if v > 0 else "#6366f1" for v in shap_df["SHAP"]]
                ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors, height=0.6)
                ax.axvline(0, color="#374151", linewidth=1)
                ax.set_xlabel("SHAP Value", color="#6b7280", fontsize=9)
                ax.tick_params(colors="#9ca3af", labelsize=8.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("#2a2d3e")
                ax.spines["left"].set_color("#2a2d3e")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption("🔴 Red = increases churn risk · 🔵 Blue = decreases churn risk")
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

            st.divider()
            with st.expander("View processed input"):
                st.dataframe(input_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📂 Batch Customer Prediction")
    st.markdown('<p style="color:#6b7280;font-size:0.85rem">Upload the original IBM Telco CSV — feature engineering is applied automatically inside the app.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        id_col = df_raw["customerID"].values if "customerID" in df_raw.columns else range(len(df_raw))
        st.success(f"✅ {len(df_raw):,} customers loaded")

        with st.spinner("Applying feature engineering and predicting..."):
            try:
                X_scaled  = preprocess_batch(df_raw.copy())
                probs     = model.predict_proba(X_scaled)[:, 1]
                preds     = (probs >= threshold).astype(int)

                result_df = pd.DataFrame({
                    "CustomerID":         id_col,
                    "Churn_Probability%": np.round(probs * 100, 1),
                    "Predicted_Churn":    ["Yes" if p == 1 else "No" for p in preds],
                    "Risk_Level":         ["🔴 High" if p >= 0.65 else "🟡 Medium" if p >= 0.4 else "🟢 Low" for p in probs],
                    "Contract":           df_raw["Contract"].values if "Contract" in df_raw.columns else "",
                    "Tenure_Months":      df_raw["tenure"].values   if "tenure"   in df_raw.columns else "",
                    "Monthly_Charges":    df_raw["MonthlyCharges"].values if "MonthlyCharges" in df_raw.columns else "",
                    "Internet_Service":   df_raw["InternetService"].values if "InternetService" in df_raw.columns else "",
                    "Online_Security":    df_raw["OnlineSecurity"].values if "OnlineSecurity" in df_raw.columns else "",
                })

                st.divider()

                st.markdown("### 📊 Prediction Summary")
                total      = len(result_df)
                n_churn    = int(preds.sum())
                churn_rate = n_churn / total * 100
                high_n     = int((probs >= 0.65).sum())
                medium_n   = int(((probs >= 0.4) & (probs < 0.65)).sum())
                low_n      = int((probs < 0.4).sum())

                s1, s2, s3, s4 = st.columns(4)
                for col, val, label in zip(
                    [s1, s2, s3, s4],
                    [f"{total:,}", f"{n_churn:,}", f"{total - n_churn:,}", f"{probs.mean()*100:.1f}%"],
                    ["Total Customers", f"Predicted to Churn ({churn_rate:.1f}%)", "Predicted to Stay", "Avg Churn Probability"]
                ):
                    col.markdown(f"""
                    <div class="batch-stat">
                        <div class="batch-stat-value">{val}</div>
                        <div class="batch-stat-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                st.markdown("### 📈 Risk Level Distribution")
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor("#1a1d27")
                ax.set_facecolor("#1a1d27")
                bars = ax.bar(
                    ["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
                    [high_n, medium_n, low_n],
                    color=["#ef4444", "#f59e0b", "#10b981"],
                    width=0.5
                )
                for bar, val in zip(bars, [high_n, medium_n, low_n]):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max(high_n, medium_n, low_n) * 0.01,
                            str(val), ha="center", fontsize=12,
                            fontweight="bold", color="#f9fafb")
                ax.set_ylabel("Number of Customers", color="#6b7280", fontsize=9)
                ax.tick_params(colors="#9ca3af", labelsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("#2a2d3e")
                ax.spines["left"].set_color("#2a2d3e")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.divider()

                st.markdown("### 🔴 High Risk Customers (≥ 65% churn probability)")
                high_risk = result_df[result_df["Churn_Probability%"] >= 65].sort_values(
                    "Churn_Probability%", ascending=False).reset_index(drop=True)

                if len(high_risk) > 0:
                    st.dataframe(high_risk, use_container_width=True)
                    st.caption(f"{len(high_risk):,} high-risk customers identified — priority targets for retention campaign")
                else:
                    st.info("No high-risk customers found.")

                st.divider()

                st.markdown("### 📋 Full Prediction Results")
                st.dataframe(
                    result_df.sort_values("Churn_Probability%", ascending=False),
                    use_container_width=True
                )

                csv_out = result_df.sort_values("Churn_Probability%", ascending=False).to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Full Results as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.caption("Model: XGBoost | Threshold: 0.3 | Feature engineering applied automatically | CISC 602 Capstone — Donggyu Noh")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Make sure the uploaded CSV matches the IBM Telco dataset format.")