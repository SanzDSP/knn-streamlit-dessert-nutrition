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
        food = h5_file['food'][:]  # Mengambil nama-nama dessert dari kolom 'food'
    return data_scaled, clusters, centroids, food

# Memuat model klasterisasi
file_path = "dessertnutrition_clusters.h5"  # Sesuaikan dengan lokasi file .h5
data_scaled, clusters, centroids, food = load_clusters(file_path)

# Latih model KMeans menggunakan data yang sudah ada
kmeans = KMeans(n_clusters=3, init=centroids, n_init=1, random_state=42)
kmeans.fit(data_scaled)  # Latih model KMeans dengan data yang sudah ada

# Deskripsi klaster
cluster_labels = ['Good Dessert', 'Moderate Dessert', 'Indulgent Dessert']

# Input dari pengguna
st.title("Dessert Recommendation")
st.write("Masukkan pilihan untuk mendapatkan rekomendasi dessert sehat yang sesuai dengan kebutuhan Anda!")

# Input selection untuk tingkat gula dan protein
sugar_level = st.selectbox("Tingkat gula", ['Low', 'Medium', 'High'], index=1)
protein_level = st.selectbox("Tingkat protein", ['Low', 'Medium', 'High'], index=1)

# Input checkbox untuk vitamin
vitamin_a = st.checkbox("Vitamin A")
vitamin_c = st.checkbox("Vitamin C")
vitamin_d = st.checkbox("Vitamin D")

# Mengonversi pilihan slider menjadi nilai numerik
sugar_dict = {'Low': 10, 'Medium': 50, 'High': 90}
protein_dict = {'Low': 5, 'Medium': 15, 'High': 30}

# Siapkan input pengguna untuk prediksi klaster
input_data = np.array([sugar_dict[sugar_level], protein_dict[protein_level], 
                       int(vitamin_a), int(vitamin_c), int(vitamin_d)]).reshape(1, -1)

# Menstandarisasi input pengguna menggunakan scaler yang sama
scaler = StandardScaler()
scaler.fit(data_scaled)  # Fitting scaler pada data yang sudah ada
input_data_scaled = scaler.transform(input_data)

# Prediksi klaster berdasarkan input pengguna
predicted_cluster = kmeans.predict(input_data_scaled)

# Menampilkan hasil prediksi
st.write(f"Klaster yang sesuai: {cluster_labels[predicted_cluster[0]]}")

# Menampilkan rekomendasi dessert berdasarkan klaster yang diprediksi
st.write("Rekomendasi dessert berdasarkan klaster ini:")

# Filter dataset untuk hanya menampilkan dessert yang termasuk dalam klaster yang diprediksi
recommended_desserts = [food[i] for i in range(len(food)) if clusters[i] == predicted_cluster[0]]

# Menampilkan nama-nama dessert yang direkomendasikan
st.write(recommended_desserts)
