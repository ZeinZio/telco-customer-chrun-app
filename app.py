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
    /* === Import Google Font & FontAwesome === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    /* === Global === */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    /* Fix for Streamlit icons (expander arrows, etc) breaking */
    .material-symbols-rounded, .material-icons, [class*="icon"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* === Hero Banner === */
    .hero-banner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 800px;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: 1px solid #bfdbfe;
    }

    /* === Metric Cards === */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: left;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #cbd5e1;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.3rem 0;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .delta-red { color: #ef4444; }
    .delta-green { color: #10b981; }
    .delta-blue { color: #3b82f6; }

    /* === Section Headers === */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* === Result Cards === */
    .result-churn {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .result-loyal {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .result-icon { 
        font-size: 3rem; 
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-churn .result-title { color: #b91c1c; }
    .result-loyal .result-title { color: #15803d; }
    .result-desc {
        font-size: 1rem;
        line-height: 1.5;
        font-weight: 400;
    }
    .result-churn .result-desc { color: #991b1b; }
    .result-loyal .result-desc { color: #166534; }

    /* === Probability Gauge === */
    .gauge-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .gauge-bar-bg {
        width: 100%;
        height: 10px;
        background: #f1f5f9;
        border-radius: 5px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .gauge-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease-out;
    }
    .gauge-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .gauge-value {
        font-size: 2rem;
        color: #0f172a;
        font-weight: 700;
    }

    /* === Recommendation Card === */
    .reco-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
    }
    .reco-card h4 {
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .reco-card ul {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
        padding-left: 1.2rem;
    }

    /* === Team Card === */
    .team-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .team-card h3 {
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    .team-members {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .member-chip {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        color: #334155;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.6rem;
    }
    .member-socials {
        display: flex;
        gap: 0.8rem;
        font-size: 1.1rem;
    }
    .member-socials a {
        text-decoration: none;
        color: #64748b;
        transition: transform 0.2s, color 0.2s;
    }
    .member-socials a:hover {
        transform: scale(1.1);
        color: #2563eb;
    }

    /* === Footer === */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 2rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    .footer a {
        color: #3b82f6;
        text-decoration: none;
    }

    /* === Tab styling === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0;
        font-weight: 500;
        color: #64748b;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        background: transparent !important;
        border-bottom: 2px solid #2563eb !important;
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

        # --- Custom Effects (Pure CSS, no iframe) ---
        import random as _rnd

        if prediction == 1:
            # === CHURN: Fire + Warning Triangle ===
            fire_emojis = ['🔥', '🔥', '🔥', '💥', '⚡']
            ember_colors = ['#ef4444', '#f97316', '#fbbf24', '#dc2626']

            # Generate fire particles HTML
            fire_particles = ""
            for i in range(45):
                emoji = _rnd.choice(fire_emojis)
                left = _rnd.uniform(0, 100)
                delay = _rnd.uniform(0, 1.5)
                duration = _rnd.uniform(2, 3.5)
                fire_particles += (
                    f'<div style="position:fixed;bottom:-40px;left:{left}vw;font-size:28px;'
                    f'pointer-events:none;z-index:9999;opacity:0;'
                    f'animation:churnFire {duration}s {delay}s ease-out forwards;">{emoji}</div>'
                )

            # Generate ember dots HTML
            for i in range(60):
                left = _rnd.uniform(0, 100)
                color = _rnd.choice(ember_colors)
                delay = _rnd.uniform(0, 2)
                duration = _rnd.uniform(2.5, 4)
                size = _rnd.uniform(3, 7)
                drift = _rnd.uniform(-100, 100)
                fire_particles += (
                    f'<div style="position:fixed;bottom:-10px;left:{left}vw;'
                    f'width:{size}px;height:{size}px;border-radius:50%;background:{color};'
                    f'pointer-events:none;z-index:9998;opacity:0;'
                    f'animation:churnEmber {duration}s {delay}s ease-out forwards;'
                    f'--drift:{drift}px;"></div>'
                )

            st.markdown(f"""
            <style>
                @keyframes churnFire {{
                    0%   {{ opacity:0; transform:translateY(0) scale(0.5) rotate(0deg); }}
                    15%  {{ opacity:1; }}
                    75%  {{ opacity:0.8; }}
                    100% {{ opacity:0; transform:translateY(-110vh) scale(1.3) rotate(35deg); }}
                }}
                @keyframes churnEmber {{
                    0%   {{ opacity:0; transform:translateY(0) scale(1); }}
                    15%  {{ opacity:1; }}
                    100% {{ opacity:0; transform:translateY(-110vh) translateX(var(--drift)) scale(0.3); }}
                }}
                @keyframes churnWarning {{
                    0%   {{ opacity:0; transform:scale(0.2) rotate(-10deg); }}
                    12%  {{ opacity:1; transform:scale(1.4) rotate(5deg); }}
                    25%  {{ transform:scale(0.9) rotate(-3deg); }}
                    40%  {{ transform:scale(1.15) rotate(2deg); }}
                    55%  {{ transform:scale(1.0) rotate(0deg); }}
                    75%  {{ opacity:1; }}
                    100% {{ opacity:0; transform:scale(1.6); }}
                }}
            </style>
            {fire_particles}
            <div style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;
                        z-index:10000;display:flex;align-items:center;justify-content:center;">
                <div style="font-size:130px;opacity:0;animation:churnWarning 2.5s ease-in-out forwards;
                            filter:drop-shadow(0 0 50px rgba(239,68,68,0.8));">⚠️</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # === LOYAL: Confetti + Fireworks ===
            st.balloons()

            confetti_emojis = ['🎊', '🎉', '✨', '⭐', '🌟', '🥳', '🎆']
            fw_colors = ['#34d399', '#6ee7b7', '#818cf8', '#a78bfa', '#fbbf24', '#f472b6', '#60a5fa']

            loyal_html = ""
            # Confetti falling from top
            for i in range(50):
                emoji = _rnd.choice(confetti_emojis)
                left = _rnd.uniform(0, 100)
                delay = _rnd.uniform(0, 2)
                duration = _rnd.uniform(2.5, 4.5)
                size = _rnd.uniform(16, 32)
                loyal_html += (
                    f'<div style="position:fixed;top:-40px;left:{left}vw;font-size:{size}px;'
                    f'pointer-events:none;z-index:9999;opacity:0;'
                    f'animation:loyalConfetti {duration}s {delay}s linear forwards;">{emoji}</div>'
                )

            # Firework burst dots
            import math
            burst_positions = [(20, 30), (75, 25), (50, 40), (35, 20), (65, 35)]
            for bx, by in burst_positions:
                b_delay = _rnd.uniform(0.2, 1.8)
                for j in range(16):
                    angle = (math.pi * 2 / 16) * j
                    dist = _rnd.uniform(50, 100)
                    fx = math.cos(angle) * dist
                    fy = math.sin(angle) * dist
                    color = _rnd.choice(fw_colors)
                    sz = _rnd.uniform(4, 7)
                    loyal_html += (
                        f'<div style="position:fixed;left:{bx}vw;top:{by}vh;'
                        f'width:{sz}px;height:{sz}px;border-radius:50%;background:{color};'
                        f'pointer-events:none;z-index:9998;opacity:0;'
                        f'animation:loyalBurst 1.5s {b_delay}s ease-out forwards;'
                        f'--bx:{fx}px;--by:{fy}px;"></div>'
                    )

            st.markdown(f"""
            <style>
                @keyframes loyalConfetti {{
                    0%   {{ opacity:1; transform:translateY(0) rotate(0deg) scale(1); }}
                    50%  {{ opacity:1; }}
                    100% {{ opacity:0; transform:translateY(110vh) rotate(720deg) scale(0.4); }}
                }}
                @keyframes loyalBurst {{
                    0%   {{ opacity:0; transform:translate(0,0) scale(0); }}
                    20%  {{ opacity:1; transform:translate(0,0) scale(1); }}
                    100% {{ opacity:0; transform:translate(var(--bx),var(--by)) scale(0.2); }}
                }}
                @keyframes loyalCeleb {{
                    0%   {{ opacity:0; transform:scale(0) rotate(-20deg); }}
                    15%  {{ opacity:1; transform:scale(1.5) rotate(10deg); }}
                    35%  {{ transform:scale(0.9) rotate(-5deg); }}
                    50%  {{ transform:scale(1.1) rotate(2deg); }}
                    70%  {{ opacity:1; transform:scale(1) rotate(0); }}
                    100% {{ opacity:0; transform:scale(2); }}
                }}
            </style>
            {loyal_html}
            <div style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;
                        z-index:10000;display:flex;align-items:center;justify-content:center;">
                <div style="font-size:120px;opacity:0;animation:loyalCeleb 2.5s ease-out forwards;
                            filter:drop-shadow(0 0 40px rgba(52,211,153,0.7));">🎉</div>
            </div>
            """, unsafe_allow_html=True)

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
    with st.expander("Lihat Raw Data (10 baris pertama)", icon="🔍"):
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
            <div class="member-chip">
                <span>🧑‍💻 Ilman Fadhil</span>
                <div class="member-socials">
                    <a href="https://linkedin.com/in/ilman-fadhil/" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
                    <a href="#" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>
                    <a href="https://instagram.com/man.fdhl" target="_blank" title="Instagram"><i class="fa-brands fa-instagram"></i></a>
                </div>
            </div>
            <div class="member-chip">
                <span>🧑‍💻 Abdu Rahman</span>
                <div class="member-socials">
                    <a href="#" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
                    <a href="#" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>
                    <a href="#" target="_blank" title="Instagram"><i class="fa-brands fa-instagram"></i></a>
                </div>
            </div>
            <div class="member-chip">
                <span>🧑‍💻 Azka Acuzio Raines Respati</span>
                <div class="member-socials">
                    <a href="https://www.linkedin.com/in/azka-acuzio-8a8a08322" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
                    <a href="https://github.com/ZeinZio" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>
                    <a href="https://www.instagram.com/dwnzzy" target="_blank" title="Instagram"><i class="fa-brands fa-instagram"></i></a>
                </div>
            </div>
            <div class="member-chip">
                <span>👩‍💻 Balqies Hawa</span>
                <div class="member-socials">
                    <a href="#" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
                    <a href="#" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>
                    <a href="#" target="_blank" title="Instagram"><i class="fa-brands fa-instagram"></i></a>
                </div>
            </div>
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
