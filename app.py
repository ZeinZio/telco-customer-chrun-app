import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# CUSTOM CSS — Premium Dark Theme
# -------------------------------------------------------
st.markdown("""
<style>
    /* === Import Google Font === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* === Global === */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* === Hero Banner === */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c7d2fe 0%, #a78bfa 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #a5b4fc;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 750px;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        letter-spacing: 0.5px;
    }

    /* === Metric Cards === */
    .metric-card {
        background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99,102,241,0.5);
        box-shadow: 0 8px 30px rgba(99,102,241,0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0e7ff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #8b8fa8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .delta-red { color: #f87171; }
    .delta-green { color: #34d399; }
    .delta-blue { color: #60a5fa; }

    /* === Section Headers === */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c7d2fe;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.8rem;
        border-left: 3px solid #6366f1;
    }

    /* === Result Cards === */
    .result-churn {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 50%, #991b1b 100%);
        border: 1px solid rgba(248,113,113,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-loyal {
        background: linear-gradient(135deg, #022c22 0%, #064e3b 50%, #065f46 100%);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-churn .result-title { color: #fca5a5; }
    .result-loyal .result-title { color: #6ee7b7; }
    .result-desc {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .result-churn .result-desc { color: #fecaca; }
    .result-loyal .result-desc { color: #a7f3d0; }

    /* === Probability Gauge === */
    .gauge-container {
        background: rgba(15,12,41,0.6);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
        text-align: center;
    }
    .gauge-bar-bg {
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        overflow: hidden;
        margin: 0.8rem 0;
    }
    .gauge-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 1s ease;
    }
    .gauge-label {
        font-size: 0.8rem;
        color: #8b8fa8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .gauge-value {
        font-size: 2rem;
        font-weight: 800;
    }

    /* === Recommendation Card === */
    .reco-card {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .reco-card h4 {
        color: #a5b4fc;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .reco-card ul {
        color: #c7d2fe;
        font-size: 0.9rem;
        line-height: 1.8;
        padding-left: 1.2rem;
    }

    /* === Team Card === */
    .team-card {
        background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .team-card h3 {
        background: linear-gradient(135deg, #c7d2fe, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .team-members {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .member-chip {
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.25);
        color: #c7d2fe;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* === Footer === */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }
    .footer a {
        color: #818cf8;
        text-decoration: none;
    }

    /* === Tab styling === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }

    /* === Hide Streamlit branding === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODEL & DATA
# -------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join('model', 'churn_model.pkl'))

@st.cache_data
def load_data():
    data = pd.read_csv(os.path.join('data', 'Telco-Customer-Churn.csv'))
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    return data

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"❌ Gagal load file: {e}")
    st.stop()

# -------------------------------------------------------
# HERO BANNER
# -------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤖 AI-POWERED PREDICTION ENGINE</div>
    <div class="hero-title">Telco Customer Churn Predictor</div>
    <div class="hero-subtitle">
        Sistem prediksi cerdas berbasis Machine Learning untuk mengidentifikasi pelanggan
        yang berisiko <b>berhenti berlangganan</b>. Dibangun untuk membantu Tim Customer Service
        memberikan penawaran retensi yang tepat sasaran.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Churn", "📊 Dashboard Analytics", "👥 Tentang Kami"])

# -------------------------------------------------------
# TAB 1 — PREDICTION
# -------------------------------------------------------
with tab1:
    st.markdown('<div class="section-header">📋 Input Data Pelanggan</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Demografi")
        gender         = st.selectbox("Gender", ["Female", "Male"], key="p_gender")
        senior         = st.selectbox("Senior Citizen", ["No", "Yes"], key="p_senior")
        partner        = st.selectbox("Partner", ["No", "Yes"], key="p_partner")
        dependents     = st.selectbox("Dependents", ["No", "Yes"], key="p_dep")
        tenure         = st.slider("Tenure (Bulan)", 0, 72, 12, key="p_tenure")
        phone_service  = st.selectbox("Phone Service", ["No", "Yes"], key="p_phone")
        paperless      = st.selectbox("Paperless Billing", ["No", "Yes"], key="p_paper")

    with col2:
        st.markdown("##### 🛠️ Layanan Internet")
        internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="p_inet")
        multi_lines    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="p_ml")
        security       = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="p_sec")
        backup         = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="p_bk")
        device_prot    = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="p_dp")
        tech_support   = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="p_ts")
        streaming_tv   = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="p_stv")
        streaming_mv   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="p_smv")

    with col3:
        st.markdown("##### 💰 Akun & Keuangan")
        contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="p_con")
        payment        = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="p_pay")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.5, key="p_mc")
        total_charges   = st.number_input("Total Charges ($)", min_value=0.0, value=600.0, step=1.0, key="p_tc")

    st.write("")
    predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True, type="primary")

    if predict_btn:
        input_data = pd.DataFrame([{
            'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone_service,
            'PaperlessBilling': paperless,
            'Contract': contract,
            'InternetService': internet,
            'MultipleLines': multi_lines,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': device_prot,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mv,
            'PaymentMethod': payment,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': float(total_charges)
        }])

        prediction = model.predict(input_data)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]

        st.markdown("---")

        # --- Result display ---
        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            if prediction == 1:
                bar_color = "linear-gradient(90deg, #ef4444, #f97316)"
                bar_pct = f"{proba*100:.0f}" if proba else "75"
                st.markdown(f"""
                <div class="result-churn">
                    <div class="result-icon">🚨</div>
                    <div class="result-title">Pelanggan Berisiko CHURN!</div>
                    <div class="result-desc">
                        Pelanggan ini menunjukkan pola perilaku yang mengindikasikan kemungkinan
                        tinggi untuk berhenti berlangganan dalam waktu dekat.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                bar_color = "linear-gradient(90deg, #10b981, #34d399)"
                bar_pct = f"{proba*100:.0f}" if proba else "25"
                st.markdown(f"""
                <div class="result-loyal">
                    <div class="result-icon">✅</div>
                    <div class="result-title">Pelanggan Cenderung LOYAL</div>
                    <div class="result-desc">
                        Pelanggan ini tidak menunjukkan tanda-tanda akan berhenti berlangganan.
                        Tetap jaga kualitas layanan untuk mempertahankan loyalitas.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            if proba is not None:
                pct = proba * 100
                if pct > 60:
                    gauge_color = "linear-gradient(90deg, #ef4444, #f97316)"
                    val_color = "#f87171"
                elif pct > 35:
                    gauge_color = "linear-gradient(90deg, #f59e0b, #eab308)"
                    val_color = "#fbbf24"
                else:
                    gauge_color = "linear-gradient(90deg, #10b981, #34d399)"
                    val_color = "#34d399"

                st.markdown(f"""
                <div class="gauge-container">
                    <div class="gauge-label">Probabilitas Churn</div>
                    <div class="gauge-value" style="color: {val_color}">{pct:.1f}%</div>
                    <div class="gauge-bar-bg">
                        <div class="gauge-bar-fill" style="width: {pct:.0f}%; background: {gauge_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            if prediction == 1:
                st.markdown("""
                <div class="reco-card">
                    <h4>💡 Rekomendasi Aksi</h4>
                    <ul>
                        <li>Hubungi pelanggan dalam <b>24 jam</b></li>
                        <li>Tawarkan upgrade kontrak dengan <b>diskon khusus</b></li>
                        <li>Berikan <b>benefit tambahan</b> (online security, tech support)</li>
                        <li>Evaluasi kepuasan pelanggan melalui <b>survey singkat</b></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="reco-card">
                    <h4>💡 Saran Retensi</h4>
                    <ul>
                        <li>Pertahankan kualitas layanan saat ini</li>
                        <li>Kirim <b>penawaran loyalty reward</b> berkala</li>
                        <li>Monitor perubahan pola penggunaan</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        if prediction == 1:
            st.snow()

# -------------------------------------------------------
# TAB 2 — DASHBOARD ANALYTICS
# -------------------------------------------------------
with tab2:
    st.markdown('<div class="section-header">📈 Ringkasan Dataset</div>', unsafe_allow_html=True)

    total_cust = len(df)
    churn_cust = df[df['Churn'] == 'Yes'].shape[0]
    loyal_cust = df[df['Churn'] == 'No'].shape[0]
    churn_rate = churn_cust / total_cust * 100
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-label">Total Pelanggan</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_cust:,}</div>
            <div class="metric-label">Churn</div>
            <div class="metric-delta delta-red">▲ {churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{loyal_cust:,}</div>
            <div class="metric-label">Loyal</div>
            <div class="metric-delta delta-green">▲ {100-churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_monthly:.0f}</div>
            <div class="metric-label">Avg Monthly Charges</div>
            <div class="metric-delta delta-blue">~{avg_tenure:.0f} bulan tenure</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-header">📊 Visualisasi Data</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Distribusi Churn**")
        churn_counts = df['Churn'].value_counts().rename_axis('Status').reset_index(name='Jumlah')
        st.bar_chart(churn_counts.set_index('Status'), color=["#818cf8"])

    with col_b:
        st.markdown("**Contract Type vs Churn**")
        contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='count')
        contract_pivot = contract_churn.pivot(index='Contract', columns='Churn', values='count').fillna(0)
        st.bar_chart(contract_pivot, color=["#34d399", "#f87171"])

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Internet Service vs Churn**")
        inet_churn = df.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        inet_pivot = inet_churn.pivot(index='InternetService', columns='Churn', values='count').fillna(0)
        st.bar_chart(inet_pivot, color=["#34d399", "#f87171"])

    with col_d:
        st.markdown("**Payment Method vs Churn**")
        pay_churn = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='count')
        pay_pivot = pay_churn.pivot(index='PaymentMethod', columns='Churn', values='count').fillna(0)
        st.bar_chart(pay_pivot, color=["#34d399", "#f87171"])

    st.write("")
    st.markdown('<div class="section-header">🔍 Insight Utama</div>', unsafe_allow_html=True)

    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        mtm = df[(df['Contract'] == 'Month-to-month') & (df['Churn'] == 'Yes')].shape[0]
        mtm_total = df[df['Contract'] == 'Month-to-month'].shape[0]
        st.info(f"📌 **{mtm/mtm_total*100:.0f}%** pelanggan Month-to-month melakukan churn — kontrak bulanan paling berisiko!")
    with ins2:
        fiber = df[(df['InternetService'] == 'Fiber optic') & (df['Churn'] == 'Yes')].shape[0]
        fiber_total = df[df['InternetService'] == 'Fiber optic'].shape[0]
        st.info(f"📌 **{fiber/fiber_total*100:.0f}%** pelanggan Fiber Optic melakukan churn — perlu evaluasi kualitas layanan!")
    with ins3:
        echeck = df[(df['PaymentMethod'] == 'Electronic check') & (df['Churn'] == 'Yes')].shape[0]
        echeck_total = df[df['PaymentMethod'] == 'Electronic check'].shape[0]
        st.info(f"📌 **{echeck/echeck_total*100:.0f}%** pengguna Electronic Check churn — dorong migrasi ke auto-payment!")

    st.write("")
    with st.expander("🔎 Lihat Raw Data (10 baris pertama)"):
        st.dataframe(df.head(10), use_container_width=True)

# -------------------------------------------------------
# TAB 3 — ABOUT / TEAM
# -------------------------------------------------------
with tab3:
    st.markdown('<div class="section-header">📖 Tentang Project</div>', unsafe_allow_html=True)

    st.markdown("""
    Industri telekomunikasi beroperasi di pasar yang sangat kompetitif. Biaya mengakuisisi pelanggan baru
    (**Customer Acquisition Cost**) jauh lebih mahal daripada mempertahankan pelanggan yang sudah ada.

    Aplikasi ini membantu **Tim Customer Service** mengidentifikasi pelanggan berisiko tinggi secara *real-time*
    menggunakan model **Machine Learning (Logistic Regression)** yang telah dilatih dengan dataset
    [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) dari Kaggle.
    """)

    st.write("")

    tech1, tech2, tech3, tech4 = st.columns(4)
    with tech1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">🐍</div>
            <div class="metric-label">Python</div>
        </div>
        """, unsafe_allow_html=True)
    with tech2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">📡</div>
            <div class="metric-label">Streamlit</div>
        </div>
        """, unsafe_allow_html=True)
    with tech3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">🤖</div>
            <div class="metric-label">Scikit-Learn</div>
        </div>
        """, unsafe_allow_html=True)
    with tech4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem;">📊</div>
            <div class="metric-label">Pandas</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="team-card">
        <h3>👥 Group 1 — AI/ML Advanced</h3>
        <div class="team-members">
            <div class="member-chip">🧑‍💻 Ilman Fadhil</div>
            <div class="member-chip">🧑‍💻 Abdu Rahman</div>
            <div class="member-chip">🧑‍💻 Azka Acuzio Raines Respati</div>
            <div class="member-chip">🧑‍💻 Balqies Hawa</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("""
<div class="footer">
    📡 Telco Churn Predictor v2.0 · Built with Streamlit<br>
    <b>Group 1</b> — AI/ML Advanced · GDGoC UIN Jakarta · 2026
</div>
""", unsafe_allow_html=True)
