# 📡 Telco Customer Churn Prediction

# Weekly Class AI/ML Advanced — Final Project

Aplikasi web berbasis **Streamlit** yang menggunakan model **Machine Learning (Klasifikasi)** untuk memprediksi apakah seorang pelanggan telekomunikasi akan **churn** (berhenti berlangganan) atau tetap loyal, berdasarkan profil demografi, layanan yang digunakan, dan informasi akun mereka.

---

## 😎 Group 1

| No | Nama |
|----|------|
| 1 | Ilman Fadhil |
| 2 | Abdu Rahman |
| 3 | Azka Acuzio Raines Respati |
| 4 | Balqies Hawa |

---

## 🧐 Latar Belakang

Industri telekomunikasi beroperasi di pasar yang sangat kompetitif. Biaya mengakuisisi pelanggan baru (**Customer Acquisition Cost**) jauh lebih mahal daripada mempertahankan pelanggan yang sudah ada.

Aplikasi ini dibangun untuk membantu **Tim Customer Service** mengidentifikasi pelanggan berisiko tinggi secara *real-time*, sehingga mereka dapat memberikan penawaran khusus tepat sebelum pelanggan memutuskan untuk berhenti berlangganan.

---

## ✨ Fitur Aplikasi

- **🔮 Prediksi Churn** — Input data pelanggan dan dapatkan prediksi churn beserta probabilitasnya
- **📊 Dashboard Data** — Visualisasi distribusi churn, tipe kontrak, monthly charges, dan internet service
- **📈 Metrik Ringkasan** — Total pelanggan, jumlah churn/loyal, dan churn rate

---

## 📚 Dataset

**[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)** — IBM Sample Data Sets (Owner: *BlastChar* @ Kaggle)

> *"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs."*

| Info | Detail |
|------|--------|
| Jumlah Data | 7.043 pelanggan |
| Jumlah Fitur | 21 kolom |
| Target | `Churn` (Yes / No) |

<details>
<summary><b>📍 Daftar Fitur Dataset (klik untuk expand)</b></summary>

| Fitur | Deskripsi |
|-------|-----------|
| `customerID` | Customer ID |
| `gender` | Male / Female |
| `SeniorCitizen` | Senior citizen atau bukan (1, 0) |
| `Partner` | Punya partner atau tidak (Yes, No) |
| `Dependents` | Punya tanggungan atau tidak (Yes, No) |
| `tenure` | Jumlah bulan berlangganan |
| `PhoneService` | Punya layanan telepon atau tidak |
| `MultipleLines` | Punya multiple lines atau tidak |
| `InternetService` | Provider internet (DSL, Fiber optic, No) |
| `OnlineSecurity` | Punya online security atau tidak |
| `OnlineBackup` | Punya online backup atau tidak |
| `DeviceProtection` | Punya device protection atau tidak |
| `TechSupport` | Punya tech support atau tidak |
| `StreamingTV` | Punya streaming TV atau tidak |
| `StreamingMovies` | Punya streaming movies atau tidak |
| `Contract` | Tipe kontrak (Month-to-month, One year, Two year) |
| `PaperlessBilling` | Paperless billing atau tidak |
| `PaymentMethod` | Metode pembayaran |
| `MonthlyCharges` | Tagihan bulanan |
| `TotalCharges` | Total tagihan |
| `Churn` | Churn atau tidak (Yes, No) — **Target** |

</details>

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/ZeinZio/streamlit-churn-app.git
cd streamlit-churn-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

---

## 🗂️ Struktur Project

```
streamlit-churn-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/
│   └── Telco-Customer-Churn.csv   # Dataset
├── model/
│   └── churn_model.pkl    # Trained ML model (Logistic Regression)
└── README.md
```

---

## 🛠️ Tech Stack

- **Python** — Bahasa pemrograman utama
- **Streamlit** — Framework web app
- **Scikit-learn** — Machine Learning (Logistic Regression + Pipeline)
- **Pandas** — Data manipulation
- **Joblib** — Model serialization

---

<p align="center">
  <b>AI/ML Advanced · GDGoC UIN Jakarta · 2026</b>
</p>
