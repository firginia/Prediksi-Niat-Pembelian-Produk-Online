# TUGAS MANCHINE LEARNING 

Nama         : Firginia Lahwa Sakira (2330511047)
Kelas        : 6C Teknik Informatika 
Mata Kuliah  : Manchine Learning
---
# Prediksi Niat Pembelian Produk Online Berdasarkan Perilaku Pengguna Menggunakan Algoritma Random Forest dengan Pendekatan CRISP-DM

## 1. Project Overview

### Latar Belakang

Perkembangan teknologi digital telah mendorong pertumbuhan industri e-commerce secara signifikan dalam beberapa tahun terakhir. Berbagai platform belanja online bersaing untuk meningkatkan pengalaman pengguna dan tingkat konversi penjualan. Namun, salah satu tantangan utama yang dihadapi perusahaan e-commerce adalah tingginya jumlah pengunjung situs yang tidak melakukan transaksi pembelian setelah mengakses berbagai halaman produk.

Perilaku pengguna selama menjelajahi situs, seperti jumlah halaman yang dikunjungi, durasi kunjungan, serta tingkat keluar dari halaman tertentu, dapat memberikan informasi yang berharga untuk memprediksi kemungkinan terjadinya pembelian. Dengan memanfaatkan teknik machine learning, perusahaan dapat mengidentifikasi calon pelanggan yang memiliki potensi tinggi untuk melakukan transaksi.

Pada proyek ini digunakan algoritma **Random Forest** untuk membangun model klasifikasi yang mampu memprediksi niat pembelian pengguna berdasarkan perilaku mereka selama mengakses situs e-commerce.

### Manfaat Proyek

**Bagi perusahaan e-commerce:**
- Membantu mengidentifikasi pelanggan potensial.
- Mendukung strategi pemasaran yang lebih tepat sasaran.
- Meningkatkan efektivitas promosi dan konversi penjualan.

**Bagi peneliti:**
- Menambah pemahaman mengenai penerapan machine learning dalam bidang e-commerce.
- Menjadi referensi untuk pengembangan model prediksi perilaku pelanggan.

---

## 2. Business Understanding

### Problem Statements

1. Bagaimana membangun model machine learning yang mampu memprediksi niat pembelian pengguna situs e-commerce?
2. Faktor-faktor apa saja yang memengaruhi keputusan pembelian pengguna?
3. Seberapa baik performa algoritma Random Forest dalam memprediksi niat pembelian produk online?

### Goals

1. Membangun model klasifikasi menggunakan algoritma Random Forest.
2. Mengidentifikasi fitur yang paling berpengaruh terhadap keputusan pembelian pengguna.
3. Mengevaluasi performa model menggunakan metrik klasifikasi.

### Solution Statement

Untuk menyelesaikan permasalahan yang telah dirumuskan, digunakan algoritma **Random Forest Classifier**. Alasan pemilihan algoritma ini adalah:

- Memiliki performa yang baik pada kasus klasifikasi.
- Mampu menangani data dengan jumlah fitur yang cukup banyak.
- Mengurangi risiko overfitting dibandingkan Decision Tree tunggal.
- Dapat memberikan informasi mengenai tingkat kepentingan setiap fitur (*feature importance*).

---

## 3. Data Understanding

### Dataset

Dataset yang digunakan adalah **Online Shoppers Purchasing Intention Dataset** yang diperoleh dari [Kaggle](https://www.kaggle.com/) dan berasal dari UCI Machine Learning Repository. Dataset ini berisi data aktivitas pengguna selama mengunjungi situs e-commerce yang digunakan untuk memprediksi kemungkinan terjadinya pembelian.

### Informasi Dataset

df.info()
df.info()

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/7bbc54ad-e295-4e41-a5da-9884d401559e" />

df.shape

<img width="256" height="67" alt="image" src="https://github.com/user-attachments/assets/0ba02a36-7bdc-4a5b-ac91-4aad2a8cd8d7" />

| Keterangan    | Nilai     |
|---------------|-----------|
| Jumlah Data   | 12.330    |
| Jumlah Fitur  | 18        |
| Missing Value | Tidak Ada |
| Target        | `Revenue` |

### Struktur Dataset

| Tipe Data | Jumlah Kolom |
|-----------|:------------:|
| Integer   | 7            |
| Float     | 7            |
| Object    | 2            |
| Boolean   | 2            |

### Deskripsi Variabel

| Variabel | Deskripsi |
|----------|-----------|
| `Administrative` | Jumlah halaman administratif yang dikunjungi |
| `Administrative_Duration` | Lama kunjungan halaman administratif |
| `Informational` | Jumlah halaman informasi yang dikunjungi |
| `Informational_Duration` | Lama kunjungan halaman informasi |
| `ProductRelated` | Jumlah halaman produk yang dikunjungi |
| `ProductRelated_Duration` | Lama kunjungan halaman produk |
| `BounceRates` | Persentase pengguna keluar setelah membuka satu halaman |
| `ExitRates` | Persentase pengguna keluar dari situs |
| `PageValues` | Nilai halaman sebelum transaksi |
| `SpecialDay` | Kedekatan dengan hari spesial |
| `Month` | Bulan kunjungan |
| `OperatingSystems` | Sistem operasi pengguna |
| `Browser` | Browser yang digunakan |
| `Region` | Wilayah pengguna |
| `TrafficType` | Sumber trafik pengguna |
| `VisitorType` | Jenis pengunjung |
| `Weekend` | Kunjungan pada akhir pekan |
| `Revenue` | Status pembelian pengguna (target) |

### Kondisi Dataset

Hasil eksplorasi menunjukkan bahwa seluruh atribut memiliki jumlah data yang sama, yaitu 12.330 observasi, sehingga tidak ditemukan *missing value* pada dataset. Selain itu, tidak ditemukan permasalahan kualitas data yang signifikan sehingga dataset dapat langsung digunakan pada tahap persiapan data.

### Distribusi Target

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/ea283c37-d8f6-40f2-9555-027bf9136b63" />


| Revenue       | Jumlah Data | Persentase |
|---------------|:-----------:|:----------:|
| Tidak Membeli | 10.297      | 84,37%     |
| Membeli       | 1.908       | 15,63%     |

> Dataset bersifat **tidak seimbang (imbalanced)**, karena jumlah pengguna yang tidak melakukan pembelian jauh lebih besar dibandingkan pengguna yang melakukan pembelian.

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/ba07e536-78e4-4583-8f07-0319c19f4765" />

Heatmap digunakan untuk melihat hubungan antar fitur numerik dalam dataset.

---

## 4. Data Preparation

### Data Cleaning

Menghapus data duplikat apabila ditemukan.

```python
df = df.drop_duplicates()
```

### Encoding Data Kategorikal

Atribut kategorikal dikonversi ke bentuk numerik menggunakan **Label Encoding**:

- `Month`
- `VisitorType`

  <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/9510c0c3-c4e2-425c-b19b-fd3d2acf9b48" />

- `Weekend`
- `Revenue`

### Feature Selection

```python
X = df.drop('Revenue', axis=1)
y = df['Revenue']
```

### Feature Selection

Memisahkan variabel fitur (X) dan target (y).

```python
X = df.drop('Revenue', axis=1)
y = df['Revenue']

### Train-Test Split

Dataset dibagi dengan rasio **80:20**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Alasan Tidak Menggunakan Normalisasi

Normalisasi tidak dilakukan karena algoritma Random Forest tidak sensitif terhadap perbedaan skala antar fitur.

---

## 5. Modeling

### Algoritma Random Forest

Random Forest merupakan algoritma *ensemble learning* yang bekerja dengan membangun sejumlah pohon keputusan (*decision tree*) dan menggabungkan hasil prediksi dari seluruh pohon menggunakan mekanisme *voting* mayoritas.

### Implementasi Model

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
```

### Prediksi

```python
y_pred = rf.predict(X_test)
```

---

## 6. Evaluation

Evaluasi dilakukan menggunakan metrik **Accuracy, Precision, Recall**, dan **F1-Score**.

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0cbd1331-05d9-4688-8e5b-ab543843b5d8" />

Evaluasi dilakukan menggunakan metrik accuracy, precision, recall, dan F1-score.
### Hasil Evaluasi

| Metrik    | Nilai  |
|-----------|:------:|
| Accuracy  | 90,41% |
| Precision | 74,06% |
| Recall    | 54,42% |
| F1-Score  | 62,74% |

**Accuracy** — Nilai sebesar 90,41% menunjukkan bahwa model mampu melakukan klasifikasi dengan tingkat ketepatan yang tinggi terhadap keseluruhan data pengujian.

**Precision** — Nilai sebesar 74,06% menunjukkan bahwa sebagian besar prediksi pembelian yang dihasilkan model merupakan prediksi yang benar.

**Recall** — Nilai sebesar 54,42% menunjukkan bahwa model masih belum mampu mendeteksi seluruh pengguna yang benar-benar melakukan pembelian. Kondisi ini dipengaruhi oleh distribusi data yang tidak seimbang.

**F1-Score** — Nilai sebesar 62,74% menunjukkan keseimbangan performa yang cukup baik antara precision dan recall.

---

## 7. Feature Importance

Salah satu keunggulan algoritma Random Forest adalah kemampuannya dalam mengukur tingkat kepentingan masing-masing fitur.

### Top 10 Feature Importance

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0569bd27-995e-4bb6-b20a-b2ed6790b18a" />

Grafik menunjukkan bahwa PageValues merupakan fitur yang paling berpengaruh terhadap keputusan pembelian pengguna.


| Peringkat | Fitur                      | Importance |
|:---------:|-----------------------------|:----------:|
| 1         | `PageValues`                | 0,3816     |
| 2         | `ProductRelated_Duration`   | 0,0889     |
| 3         | `ExitRates`                 | 0,0867     |
| 4         | `ProductRelated`            | 0,0751     |
| 5         | `Administrative_Duration`   | 0,0608     |
| 6         | `BounceRates`                | 0,0567     |
| 7         | `Month`                     | 0,0418     |
| 8         | `Administrative`            | 0,0410     |
| 9         | `TrafficType`               | 0,0322     |
| 10        | `Region`                    | 0,0299     |

### Analisis

Fitur `PageValues` memiliki pengaruh terbesar terhadap keputusan pembelian pengguna. Selain itu, durasi kunjungan pada halaman produk, jumlah halaman produk yang dikunjungi, dan tingkat keluar halaman juga berkontribusi secara signifikan terhadap hasil prediksi.

---

## 8. Kesimpulan

Algoritma Random Forest berhasil digunakan untuk memprediksi niat pembelian pengguna pada situs e-commerce menggunakan **Online Shoppers Purchasing Intention Dataset**.

Model yang dibangun menghasilkan nilai **Accuracy 90,41%**, **Precision 74,06%**, **Recall 54,42%**, dan **F1-Score 62,74%**. Hasil tersebut menunjukkan bahwa model memiliki kemampuan yang baik dalam mengklasifikasikan perilaku pengguna berdasarkan aktivitas mereka selama mengakses situs.

Analisis *feature importance* menunjukkan bahwa `PageValues`, `ProductRelated_Duration`, `ExitRates`, dan `ProductRelated` merupakan faktor yang paling berpengaruh terhadap keputusan pembelian pengguna.

Meskipun demikian, distribusi data yang tidak seimbang menyebabkan kemampuan model dalam mendeteksi seluruh pengguna yang melakukan pembelian masih dapat ditingkatkan. Pada penelitian selanjutnya dapat diterapkan teknik penanganan data tidak seimbang seperti **SMOTE** atau **optimasi hyperparameter** untuk meningkatkan performa model.
