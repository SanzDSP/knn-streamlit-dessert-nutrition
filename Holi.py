import streamlit as st
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import io

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
file_path = "twodessertnutrition_clusters.h5"  # Sesuaikan dengan lokasi file .h5
data_scaled, clusters, centroids, food = load_clusters(file_path)

# Latih model KMeans menggunakan data yang sudah ada
kmeans = KMeans(n_clusters=3, init=centroids, n_init=1, random_state=42)
kmeans.fit(data_scaled)  # Latih model KMeans dengan data yang sudah ada

# Deskripsi klaster
cluster_labels = ['Good Dessert', 'Moderate Dessert', 'Indulgent Dessert']

# Input dari pengguna
st.title("Dessert Recommendation")
st.write("Masukkan pilihan untuk mendapatkan rekomendasi dessert sehat yang sesuai dengan kebutuhan Anda!")

# Input selection untuk setiap fitur dengan tingkat Low, Medium, dan High
sugar_level = st.selectbox("Tingkat gula", ['Low', 'Medium', 'High'], index=1)
protein_level = st.selectbox("Tingkat protein", ['Low', 'Medium', 'High'], index=1)
fat_level = st.selectbox("Tingkat lemak", ['Low', 'Medium', 'High'], index=1)
carbohydrates_level = st.selectbox("Tingkat karbohidrat", ['Low', 'Medium', 'High'], index=1)
fiber_level = st.selectbox("Tingkat serat", ['Low', 'Medium', 'High'], index=1)

# Input checkbox untuk vitamin
vitamin_a = st.checkbox("Vitamin A")
vitamin_c = st.checkbox("Vitamin C")
vitamin_d = st.checkbox("Vitamin D")

# Mengonversi pilihan slider menjadi nilai numerik
sugar_dict = {'Low': 10, 'Medium': 50, 'High': 90}
protein_dict = {'Low': 5, 'Medium': 15, 'High': 30}
fat_dict = {'Low': 5, 'Medium': 15, 'High': 30}
carbohydrates_dict = {'Low': 10, 'Medium': 50, 'High': 90}
fiber_dict = {'Low': 5, 'Medium': 15, 'High': 30}

# Siapkan input pengguna untuk prediksi klaster
input_data = np.array([sugar_dict[sugar_level], protein_dict[protein_level], fat_dict[fat_level], 
                       carbohydrates_dict[carbohydrates_level], fiber_dict[fiber_level], 
                       int(vitamin_a), int(vitamin_c), int(vitamin_d)]).reshape(1, -1)

# Definisikan semua fitur yang digunakan dalam klasterisasi (8 fitur yang diinputkan)
all_features = ['Sugars', 'Protein', 'Fat', 'Carbohydrates', 'Fiber', 'Vitamin A', 'Vitamin C', 'Vitamin D']

# Tambahkan nilai default untuk fitur lainnya yang tidak diinputkan oleh pengguna (misalnya, 0 atau nilai rata-rata)
default_values = np.zeros(len(data_scaled[0]) - len(all_features))  # Menyesuaikan dengan jumlah fitur yang ada
input_data_full = np.concatenate([input_data, default_values.reshape(1, -1)], axis=1)

# Menstandarisasi input pengguna menggunakan scaler yang sama
scaler = StandardScaler()
scaler.fit(data_scaled)  # Fitting scaler pada data yang sudah ada
input_data_scaled = scaler.transform(input_data_full)

# Prediksi klaster berdasarkan input pengguna
predicted_cluster = kmeans.predict(input_data_scaled)

# Menampilkan hasil prediksi
st.write(f"Klaster yang sesuai: {cluster_labels[predicted_cluster[0]]}")

# Menampilkan rekomendasi dessert berdasarkan klaster yang diprediksi
st.write("Rekomendasi dessert berdasarkan klaster ini:")

# Filter dataset untuk hanya menampilkan dessert yang termasuk dalam klaster yang diprediksi
recommended_desserts = [food[i] for i in range(len(food)) if clusters[i] == predicted_cluster[0]]

# Menampilkan nama-nama dessert yang direkomendasikan
st.write("3 Rekomendasi Dessert:")
st.write(recommended_desserts[:3])  # Menampilkan 3 rekomendasi teratas
