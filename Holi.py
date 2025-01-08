import streamlit as st
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import io

# Fungsi untuk memuat data klasterisasi dari file .h5
def load_clusters(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data_scaled = h5_file['data_scaled'][:]
        clusters = h5_file['clusters'][:]
        centroids = h5_file['centroids'][:]
    return data_scaled, clusters, centroids

# Memuat model klasterisasi
file_path = "dessertnutrition_clusters.h5"  # Sesuaikan dengan lokasi file .h5
data_scaled, clusters, centroids = load_clusters(file_path)

# Latih model KMeans menggunakan data yang sudah ada
kmeans = KMeans(n_clusters=3, init=centroids, n_init=1, random_state=42)
kmeans.fit(data_scaled)  # Latih model KMeans dengan data yang sudah ada

# Deskripsi klaster
cluster_labels = ['Good Dessert', 'Moderate Dessert', 'Indulgent Dessert']

# Input dari pengguna
st.title("Dessert Recommendation")
st.write("Masukkan pilihan untuk mendapatkan rekomendasi dessert sehat yang sesuai dengan kebutuhan Anda!")

# Input slider untuk gula dan protein
sugar_level = st.slider("Pilih tingkat gula", min_value=0, max_value=100, value=20)
protein_level = st.slider("Pilih tingkat protein", min_value=0, max_value=100, value=5)

# Input checkbox untuk vitamin
vitamin_a = st.checkbox("Vitamin A")
vitamin_c = st.checkbox("Vitamin C")
vitamin_d = st.checkbox("Vitamin D")

# Siapkan input pengguna untuk prediksi klaster
# Pastikan input sesuai dengan urutan fitur yang digunakan oleh model
# Menyesuaikan urutan fitur sesuai dengan data yang digunakan dalam pelatihan
input_data = np.array([sugar_level, protein_level, int(vitamin_a), int(vitamin_c), int(vitamin_d)]).reshape(1, -1)

# Menstandarisasi input pengguna menggunakan scaler yang sama
scaler = StandardScaler()
# Gunakan scaler yang sudah digunakan sebelumnya untuk data_scaled
scaler.fit(data_scaled)  # Fitting scaler pada data yang sudah ada
input_data_scaled = scaler.transform(input_data)

# Prediksi klaster berdasarkan input pengguna
predicted_cluster = kmeans.predict(input_data_scaled)

# Menampilkan hasil prediksi
st.write(f"Klaster yang sesuai: {cluster_labels[predicted_cluster[0]]}")

# Menampilkan rekomendasi
st.write("Rekomendasi dessert berdasarkan klaster ini:")
# Filter dataset untuk hanya menampilkan dessert yang termasuk dalam klaster yang diprediksi
# (Pada aplikasi nyata, kita akan memilih dessert berdasarkan klaster dan preferensi)
recommended_desserts = pd.DataFrame(data_scaled, columns=["Caloric Value", "Protein", "Sugars", "Fat"])  # contohnya
st.dataframe(recommended_desserts)
