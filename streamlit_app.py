import streamlit as st
import pickle
import pandas as pd

# Konfigurasi halaman utama (Harus di paling atas)
st.set_page_config(
    page_title="Prediksi Niat Pembelian Online",
    page_icon="🛍️",
    layout="wide"
)

# Load model
@st.cache_resource # Menggunakan cache agar model tidak di-load ulang setiap kali ada input berubah
def load_model():
    with open('src/random_forest_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header Aplikasi
st.title("🛍️ Prediksi Niat Pembelian Produk Online")
st.write("Aplikasi ini memprediksi apakah seorang pengunjung website e-commerce berpotensi melakukan pembelian berdasarkan perilaku browsing mereka.")
st.markdown("---")

# Membagi halaman menjadi 2 bagian: Input (Kiri) dan Hasil (Kanan)
col_input, col_result = st.columns([2, 1], gap="large")

with col_input:
    st.subheader("📋 Input Data Perilaku Pengguna")
    
    # Membuat Tab agar form input lebih rapi dan terorganisir
    tab1, tab2, tab3 = st.tabs(["📊 Aktivitas Halaman", "⏱️ Durasi Kunjungan", "👤 Profil & Waktu"])
    
    with tab1:
        col_t1_1, col_t1_2 = st.columns(2)
        with col_t1_1:
            Administrative = st.number_input("Jumlah Halaman Akun/Admin (Administrative)", min_value=0, step=1, value=0)
            Informational = st.number_input("Jumlah Halaman Informasi (Informational)", min_value=0, step=1, value=0)
            ProductRelated = st.number_input("Jumlah Halaman Produk (ProductRelated)", min_value=0, step=1, value=0)
        with col_t1_2:
            BounceRates = st.number_input("Bounce Rates", min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
            ExitRates = st.number_input("Exit Rates", min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
            PageValues = st.number_input("Page Values", min_value=0.0, step=0.1, value=0.0)

    with tab2:
        col_t2_1, col_t2_2 = st.columns(2)
        with col_t2_1:
            Administrative_Duration = st.number_input("Durasi di Halaman Admin (Detik)", min_value=0.0, step=1.0)
            Informational_Duration = st.number_input("Durasi di Halaman Informasi (Detik)", min_value=0.0, step=1.0)
        with col_t2_2:
            ProductRelated_Duration = st.number_input("Durasi di Halaman Produk (Detik)", min_value=0.0, step=1.0)
            SpecialDay = st.slider("Kedekatan dengan Hari Spesial (SpecialDay)", min_value=0.0, max_value=1.0, step=0.1)

    with tab3:
        col_t3_1, col_t3_2 = st.columns(2)
        with col_t3_1:
            # Mengubah angka encoding manual menjadi selectbox yang lebih user-friendly
            month_options = {
                "Februari": 2, "Maret": 3, "Mei": 5, "Juni": 6, 
                "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, 
                "November": 11, "Desember": 12
            }
            selected_month = st.selectbox("Bulan Kunjungan", list(month_options.keys()))
            Month = month_options[selected_month]
            
            OperatingSystems = st.number_input("Sistem Operasi (ID)", min_value=1, max_value=8, step=1)
            Browser = st.number_input("Browser (ID)", min_value=1, max_value=13, step=1)
            
        with col_t3_2:
            Region = st.number_input("Wilayah/Region (ID)", min_value=1, max_value=9, step=1)
            TrafficType = st.number_input("Jenis Trafik/Traffic Type (ID)", min_value=1, max_value=20, step=1)
            
            # Asumsi encoding VisitorType (misal: 0=New Visitor, 1=Returning Visitor, 2=Other)
            visitor_options = {"New Visitor": 0, "Returning Visitor": 1, "Other": 2}
            selected_visitor = st.selectbox("Tipe Pengunjung", list(visitor_options.keys()))
            VisitorType = visitor_options[selected_visitor]
            
            # Mengubah input biner 0/1 menjadi pilihan kata
            weekend_option = st.selectbox("Apakah Hari Libur/Weekend?", ["Tidak", "Ya"])
            Weekend = 1 if weekend_option == "Ya" else 0

# Bagian Kanan untuk Tombol Aksi dan Hasil Prediksi
with col_result:
    st.subheader("🔍 Hasil Analisis")
    st.write("Klik tombol di bawah ini untuk melihat hasil prediksi berdasarkan data yang telah dimasukkan.")
    
    # Membuat tombol prediksi yang terlihat dominan
    btn_predict = st.button("Mulai Prediksi 🚀", use_container_width=True, type="primary")
    
    st.markdown("---")
    
    if btn_predict:
        # Menyiapkan data untuk model
        data = pd.DataFrame({
            'Administrative': [Administrative],
            'Administrative_Duration': [Administrative_Duration],
            'Informational': [Informational],
            'Informational_Duration': [Informational_Duration],
            'ProductRelated': [ProductRelated],
            'ProductRelated_Duration': [ProductRelated_Duration],
            'BounceRates': [BounceRates],
            'ExitRates': [ExitRates],
            'PageValues': [PageValues],
            'SpecialDay': [SpecialDay],
            'Month': [Month],
            'OperatingSystems': [OperatingSystems],
            'Browser': [Browser],
            'Region': [Region],
            'TrafficType': [TrafficType],
            'VisitorType': [VisitorType],
            'Weekend': [Weekend]
        })

        # Melakukan prediksi
        prediction = model.predict(data)
        
        # Menampilkan hasil dengan visualisasi info box yang menarik
        if prediction[0] == 1:
            st.balloons() # Efek balon jika hasil positif/sukses
            st.success("### 🎉 Hasil: Pengguna Berpotensi Membeli!")
            st.info("💡 **Rekomendasi:** Berikan promo sensitif waktu atau tawarkan bantuan chat langsung (live chat) untuk mendorong konversi penjualan.")
        else:
            st.error("### 🛍️ Hasil: Pengguna Tidak Berpotensi Membeli")
            st.warning("💡 **Rekomendasi:** Tampilkan rekomendasi produk terpopuler atau berikan diskon khusus untuk menarik perhatian kembali.")