# Prediksi-Niat-Pembelian-Produk-Online

**LAPORAN PROYEK MACHINE LEARNING**
**Prediksi Niat Pembelian Produk Online Berdasarkan Perilaku Pengguna Menggunakan Algoritma Random Forest dengan Pendekatan CRISP-DM**

Dataset: Online Shoppers Purchasing Intention Dataset (UCI / Kaggle)
Pendekatan: CRISP-DM
2026

**1. Project Overview**
1.1 Latar Belakang
Perkembangan teknologi digital telah mendorong pertumbuhan industri e-commerce secara signifikan dalam beberapa tahun terakhir. Berbagai platform belanja online bersaing untuk meningkatkan pengalaman pengguna dan tingkat konversi penjualan. Namun, salah satu tantangan utama yang dihadapi perusahaan e-commerce adalah tingginya jumlah pengunjung situs yang tidak melakukan transaksi pembelian setelah mengakses berbagai halaman produk.

Perilaku pengguna selama menjelajahi situs, seperti jumlah halaman yang dikunjungi, durasi kunjungan, serta tingkat keluar dari halaman tertentu, dapat memberikan informasi yang berharga untuk memprediksi kemungkinan terjadinya pembelian. Dengan memanfaatkan teknik machine learning, perusahaan dapat mengidentifikasi calon pelanggan yang memiliki potensi tinggi untuk melakukan transaksi.

Pada proyek ini digunakan algoritma Random Forest untuk membangun model klasifikasi yang mampu memprediksi niat pembelian pengguna berdasarkan perilaku mereka selama mengakses situs e-commerce.

1.2 Manfaat Proyek

Bagi perusahaan e-commerce:

•	Membantu mengidentifikasi pelanggan potensial.

•	Mendukung strategi pemasaran yang lebih tepat sasaran.

•	Meningkatkan efektivitas promosi dan konversi penjualan.

Bagi peneliti:

•	Menambah pemahaman mengenai penerapan machine learning dalam bidang e-commerce.

•	Menjadi referensi untuk pengembangan model prediksi perilaku pelanggan.

2. Business Understanding
   
2.1 Problem Statements

Berdasarkan latar belakang yang telah dijelaskan, permasalahan yang akan diselesaikan dalam proyek ini adalah:

1.	Bagaimana membangun model machine learning yang mampu memprediksi niat pembelian pengguna situs e-commerce?
   
3.	Faktor-faktor apa saja yang memengaruhi keputusan pembelian pengguna?
   
5.	Seberapa baik performa algoritma Random Forest dalam memprediksi niat pembelian produk online?

2.2 Goals
Tujuan dari proyek ini adalah:
1.	Membangun model klasifikasi menggunakan algoritma Random Forest.
2.	Mengidentifikasi fitur yang paling berpengaruh terhadap keputusan pembelian pengguna.
3.	Mengevaluasi performa model menggunakan metrik klasifikasi.

2.3 Solution Statement
Untuk menyelesaikan permasalahan yang telah dirumuskan, digunakan algoritma Random Forest Classifier. Alasan pemilihan algoritma ini adalah:
•	Memiliki performa yang baik pada kasus klasifikasi.
•	Mampu menangani data dengan jumlah fitur yang cukup banyak.
•	Mengurangi risiko overfitting dibandingkan Decision Tree tunggal.
•	Dapat memberikan informasi mengenai tingkat kepentingan setiap fitur (feature importance).
3. Data Understanding
3.1 Dataset
Dataset yang digunakan adalah Online Shoppers Purchasing Intention Dataset yang diperoleh dari Kaggle dan berasal dari UCI Machine Learning Repository. Dataset ini berisi data aktivitas pengguna selama mengunjungi situs e-commerce yang digunakan untuk memprediksi kemungkinan terjadinya pembelian.
3.2 Informasi Dataset

df.info()

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/7bbc54ad-e295-4e41-a5da-9884d401559e" />

df.shape
<img width="230" height="250" alt="image" src="https://github.com/user-attachments/assets/14b0cab5-1037-4fac-92db-0cd41234890d" />

| Keterangan | Nilai |
|----------|----------|
| Jumlah data | 12.330 | 
| jumlah fitur | 18 | 
| Missing Value | Tidak Ada | 
| Target | revenue | 


3.3 Struktur Dataset
### Struktur Dataset

Berdasarkan hasil eksplorasi menggunakan fungsi `df.info()`, dataset terdiri dari beberapa tipe data sebagai berikut:

| Tipe Data | Jumlah Kolom |
|------------|------------|
| Integer | 7 |
| Float | 7 |
| Object | 2 |
| Boolean | 2 |

Kolom bertipe **Object** terdiri dari:

- Month
- VisitorType

Sedangkan kolom bertipe **Boolean** terdiri dari:

- Weekend
- Revenue

Karena algoritma Machine Learning tidak dapat memproses data kategorikal secara langsung, maka atribut tersebut akan dikonversi ke bentuk numerik pada tahap Data Preparation menggunakan Label Encoding.
3.4 Deskripsi Variabel
Variabel	Deskripsi
Administrative	Jumlah halaman administratif yang dikunjungi
Administrative_Duration	Lama kunjungan halaman administratif
Informational	Jumlah halaman informasi yang dikunjungi
Informational_Duration	Lama kunjungan halaman informasi
ProductRelated	Jumlah halaman produk yang dikunjungi
ProductRelated_Duration	Lama kunjungan halaman produk
BounceRates	Persentase pengguna keluar setelah membuka satu halaman
ExitRates	Persentase pengguna keluar dari situs
PageValues	Nilai halaman sebelum transaksi
SpecialDay	Kedekatan dengan hari spesial
Month	Bulan kunjungan
OperatingSystems	Sistem operasi pengguna
Browser	Browser yang digunakan
Region	Wilayah pengguna
TrafficType	Sumber trafik pengguna
VisitorType	Jenis pengunjung
Weekend	Kunjungan pada akhir pekan
Revenue	Status pembelian pengguna (target)
Tabel 3. Deskripsi variabel dalam dataset
3.5 Kondisi Dataset
Hasil eksplorasi menunjukkan bahwa seluruh atribut memiliki jumlah data yang sama, yaitu 12.330 observasi, sehingga tidak ditemukan missing value pada dataset. Selain itu, tidak ditemukan permasalahan kualitas data yang signifikan sehingga dataset dapat langsung digunakan pada tahap persiapan data.
3.6 Distribusi Target
Distribusi variabel target Revenue adalah sebagai berikut:
 
Revenue	Jumlah Data	Persentase
Tidak Membeli	10.297	84,37%
Membeli	1.908	15,63%
Tabel 4. Distribusi kelas pada variabel target Revenue
Berdasarkan hasil tersebut, dapat diketahui bahwa dataset bersifat tidak seimbang (imbalanced dataset), karena jumlah pengguna yang tidak melakukan pembelian jauh lebih besar dibandingkan pengguna yang melakukan pembelian.
Correlation Heatmap
 
Heatmap digunakan untuk melihat hubungan antar fitur numerik dalam dataset.
4.	Data Preparation
Tahapan data preparation dilakukan untuk mempersiapkan data sebelum digunakan dalam proses pemodelan.
4.1 Data Cleaning
Tahap ini dilakukan untuk memastikan kualitas data dengan menghapus data duplikat apabila ditemukan.
df = df.drop_duplicates()
4.2 Encoding Data Kategorikal
Beberapa atribut kategorikal dikonversi ke bentuk numerik menggunakan Label Encoding. Atribut yang dilakukan encoding adalah:
•	Month
•	VisitorType
 
Sebagian besar pengguna merupakan Returning Visitor dibandingkan New Visitor.
•	Weekend
•	Revenue
4.3 Feature Selection
Memisahkan variabel fitur (X) dan target (y).
X = df.drop('Revenue', axis=1)
y = df['Revenue']
4.4 Train-Test Split
Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
4.5 Alasan Tidak Menggunakan Normalisasi
Pada penelitian ini tidak dilakukan proses normalisasi karena algoritma Random Forest tidak sensitif terhadap perbedaan skala antar fitur.
5. Modeling
5.1 Algoritma Random Forest
Random Forest merupakan algoritma ensemble learning yang bekerja dengan membangun sejumlah pohon keputusan (decision tree) dan menggabungkan hasil prediksi dari seluruh pohon menggunakan mekanisme voting mayoritas.
5.2 Implementasi Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
5.3 Prediksi
y_pred = rf.predict(X_test)
6. Evaluation
Confusion Matrix digunakan untuk mengevaluasi performa model dalam mengklasifikasikan data pembelian dan tidak pembelian.
 
Evaluasi dilakukan menggunakan metrik accuracy, precision, recall, dan F1-score.
6.1 Hasil Evaluasi
Metrik	Nilai
Accuracy	90,41%
Precision	74,06%
Recall	54,42%
F1-Score	62,74%
Tabel 5. Hasil evaluasi model Random Forest
6.2 Accuracy
Nilai accuracy sebesar 90,41% menunjukkan bahwa model mampu melakukan klasifikasi dengan tingkat ketepatan yang tinggi terhadap keseluruhan data pengujian.
6.3 Precision
Nilai precision sebesar 74,06% menunjukkan bahwa sebagian besar prediksi pembelian yang dihasilkan model merupakan prediksi yang benar.
6.4 Recall
Nilai recall sebesar 54,42% menunjukkan bahwa model masih belum mampu mendeteksi seluruh pengguna yang benar-benar melakukan pembelian. Kondisi ini dipengaruhi oleh distribusi data yang tidak seimbang, di mana jumlah pengguna yang tidak membeli jauh lebih banyak dibandingkan pengguna yang membeli.
6.5 F1-Score
Nilai F1-score sebesar 62,74% menunjukkan bahwa model memiliki keseimbangan performa yang cukup baik antara precision dan recall.
7. Feature Importance
Salah satu keunggulan algoritma Random Forest adalah kemampuannya dalam mengukur tingkat kepentingan masing-masing fitur.
7.1 Top 10 Feature Importance
Peringkat	Fitur	Importance
1	PageValues	0,3816
2	ProductRelated_Duration	0,0889
3	ExitRates	0,0867
4	ProductRelated	0,0751
5	Administrative_Duration	0,0608
6	BounceRates	0,0567
7	Month	0,0418
8	Administrative	0,0410
9	TrafficType	0,0322
10	Region	0,0299
Tabel 6. Sepuluh fitur dengan tingkat kepentingan tertinggi
 
Grafik menunjukkan bahwa PageValues merupakan fitur yang paling berpengaruh terhadap keputusan pembelian pengguna.
7.2 Analisis
Fitur PageValues memiliki pengaruh terbesar terhadap keputusan pembelian pengguna. Selain itu, durasi kunjungan pada halaman produk, jumlah halaman produk yang dikunjungi, dan tingkat keluar halaman juga berkontribusi secara signifikan terhadap hasil prediksi.
8. Kesimpulan
Berdasarkan hasil penelitian yang telah dilakukan, algoritma Random Forest berhasil digunakan untuk memprediksi niat pembelian pengguna pada situs e-commerce menggunakan Online Shoppers Purchasing Intention Dataset.
Model yang dibangun menghasilkan nilai accuracy sebesar 90,41%, precision sebesar 74,06%, recall sebesar 54,42%, dan F1-score sebesar 62,74%. Hasil tersebut menunjukkan bahwa model memiliki kemampuan yang baik dalam mengklasifikasikan perilaku pengguna berdasarkan aktivitas mereka selama mengakses situs.
Analisis feature importance menunjukkan bahwa PageValues, ProductRelated_Duration, ExitRates, dan ProductRelated merupakan faktor yang paling berpengaruh terhadap keputusan pembelian pengguna.
Meskipun demikian, distribusi data yang tidak seimbang menyebabkan kemampuan model dalam mendeteksi seluruh pengguna yang melakukan pembelian masih dapat ditingkatkan. Pada penelitian selanjutnya dapat diterapkan teknik penanganan data tidak seimbang seperti SMOTE atau optimasi hyperparameter untuk meningkatkan performa model.
