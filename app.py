import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
import os

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# LOAD MODEL (pakai cache biar ga di-load ulang tiap refresh)
@st.cache_resource
def load_model():
    model_path = os.path.join('model', 'churn_model.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join('data', 'Telco-Customer-Churn.csv'))

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"❌ Gagal load file: {e}")
    st.stop()  # Hentikan app kalau model/data gagal di-load

# HEADER
st.title("📡 Telco Customer Churn Prediction")
st.markdown("""
Aplikasi ini digunakan oleh **Tim Customer Service** untuk memprediksi pelanggan yang berisiko 
berhenti berlangganan. Input data pelanggan di bawah untuk melihat status churn-nya.
""")
st.divider()

# TAB NAVIGASI
tab1, tab2 = st.tabs(["🔮 Prediksi Churn", "📊 Dashboard Data"])

# TAB 1: FORM PREDIKSI
with tab1:
    st.subheader("Input Data Pelanggan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demografi**")
        gender         = st.selectbox("Gender", ["Female", "Male"])
        senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Partner", ["No", "Yes"])
        dependents     = st.selectbox("Dependents", ["No", "Yes"])
        tenure         = st.slider("Tenure (Bulan)", 0, 72, 12)
        phone_service  = st.selectbox("Phone Service", ["No", "Yes"])
        paperless      = st.selectbox("Paperless Billing", ["No", "Yes"])

    with col2:
        st.markdown("**🛠️ Layanan Internet**")
        internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        multi_lines    = st.selectbox("Multiple Lines",      ["No", "Yes", "No phone service"])
        security       = st.selectbox("Online Security",     ["No", "Yes", "No internet service"])
        backup         = st.selectbox("Online Backup",       ["No", "Yes", "No internet service"])
        device_prot    = st.selectbox("Device Protection",   ["No", "Yes", "No internet service"])
        tech_support   = st.selectbox("Tech Support",        ["No", "Yes", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV",        ["No", "Yes", "No internet service"])
        streaming_mv   = st.selectbox("Streaming Movies",    ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**💰 Akun & Keuangan**")
        contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment        = st.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"
                         ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.5)
        total_charges   = st.number_input("Total Charges ($)",   min_value=0.0, value=600.0, step=1.0)

    st.write("")
    predict_btn = st.button("🚀 Cek Status Pelanggan", use_container_width=True, type="primary")

    if predict_btn:
        # --- Susun DataFrame sesuai urutan kolom saat training ---
        # Urutan: categorical → numerical → ordinal (sesuai ColumnTransformer di notebook)
        input_data = pd.DataFrame([{
            # Ordinal columns
            'gender'          : gender,
            'Partner'         : partner,
            'Dependents'      : dependents,
            'PhoneService'    : phone_service,
            'PaperlessBilling': paperless,
            'Contract'        : contract,
            # Categorical (OneHot) columns
            'InternetService' : internet,
            'MultipleLines'   : multi_lines,
            'OnlineSecurity'  : security,
            'OnlineBackup'    : backup,
            'DeviceProtection': device_prot,
            'TechSupport'     : tech_support,
            'StreamingTV'     : streaming_tv,
            'StreamingMovies' : streaming_mv,
            'PaymentMethod'   : payment,
            # Numerical columns
            'SeniorCitizen'   : 1 if senior == "Yes" else 0,
            'tenure'          : tenure,
            'MonthlyCharges'  : monthly_charges,
            'TotalCharges'    : pd.to_numeric(total_charges, errors='coerce')
        }])

        # --- Prediksi ---
        prediction = model.predict(input_data)[0]   # output: 0 atau 1

        # --- Cek apakah model support probabilitas ---
        proba_text = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]  # probabilitas churn (kelas 1)
            proba_text = f"Probabilitas Churn: **{proba * 100:.1f}%**"

        st.divider()
        # Output 1 = Churn, 0 = Loyal
        if prediction == 1:
            st.error("🚨 **Hasil: Pelanggan Berisiko CHURN!**")
            if proba_text:
                st.metric("Probabilitas Churn", f"{proba * 100:.1f}%", delta="Risiko Tinggi", delta_color="inverse")
            st.warning("💡 **Rekomendasi:** Segera hubungi pelanggan & tawarkan promo retensi atau diskon tagihan.")
        else:
            st.success("✅ **Hasil: Pelanggan Cenderung LOYAL**")
            if proba_text:
                st.metric("Probabilitas Churn", f"{proba * 100:.1f}%", delta="Risiko Rendah")
            st.info("👍 **Pelanggan ini tidak menunjukkan tanda-tanda akan berhenti berlangganan.**")

# TAB 2: DASHBOARD DATA
with tab2:
    st.subheader("📊 Gambaran Umum Dataset")

    # Metrik ringkasan
    total   = len(df)
    churn   = df[df['Churn'] == 'Yes'].shape[0]
    loyal   = df[df['Churn'] == 'No'].shape[0]
    rate    = churn / total * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Pelanggan", f"{total:,}")
    m2.metric("Churn",  f"{churn:,}",  f"{rate:.1f}%",  delta_color="inverse")
    m3.metric("Loyal",  f"{loyal:,}",  f"{100-rate:.1f}%")
    m4.metric("Churn Rate", f"{rate:.1f}%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Distribusi Churn**")
        churn_counts = df['Churn'].value_counts().rename_axis('Status').reset_index(name='Jumlah')
        st.bar_chart(churn_counts.set_index('Status'))

    with col_b:
        st.markdown("**Distribusi Contract Type**")
        contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='count')
        contract_pivot = contract_churn.pivot(index='Contract', columns='Churn', values='count').fillna(0)
        st.bar_chart(contract_pivot)

    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Monthly Charges — Churn vs Loyal**")
        charges_data = df[['MonthlyCharges', 'Churn']].copy()
        charges_churn = charges_data[charges_data['Churn'] == 'Yes']['MonthlyCharges'].rename('Churn')
        charges_loyal = charges_data[charges_data['Churn'] == 'No']['MonthlyCharges'].rename('Loyal')
        st.line_chart(pd.concat([charges_churn.reset_index(drop=True),
                                 charges_loyal.reset_index(drop=True)], axis=1))

    with col_d:
        st.markdown("**Internet Service vs Churn**")
        inet_churn = df.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        inet_pivot  = inet_churn.pivot(index='InternetService', columns='Churn', values='count').fillna(0)
        st.bar_chart(inet_pivot)

    st.divider()
    with st.expander("🔎 Lihat Sample Data (10 baris pertama)"):
        st.dataframe(df.head(10), use_container_width=True)

# FOOTER
st.divider()
st.caption("📡 Telco Churn Predictor | Developed by Group 1 — AI/ML Advanced · GDGoC UIN Jakarta")
