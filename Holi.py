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
        desserts = h5_file['desserts'][:]  # Nama-nama dessert
    return data_scaled, clusters, centroids, desserts

# Memuat model klasterisasi
file_path = "dessertnutrition_clusters.h5"  # Sesuaikan dengan lokasi file .h5
data_scaled, clusters, centroids, desserts = load_clusters(file_path)

# Latih model KMeans menggunakan data yang sudah ada
kmeans = KMeans(n_clusters=3, init=centroids, n_init=1, random_state=42)
kmeans.fit(data_scaled)  # Latih model KMeans dengan data yang sudah ada

# Deskripsi klaster
cluster_labels = ['Good Dessert', 'Moderate Dessert', 'Indulgent Dessert']

# Input dari pengguna
st.title("Dessert Recommendation")
st.write("Masukkan pilihan untuk mendapatkan rekomendasi dessert sehat yang sesuai dengan kebutuhan Anda!")

# Input selection untuk 34 fitur
features = [
    'Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats',
    'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
    'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2', 'Vitamin B3', 
    'Vitamin B5', 'Vitamin B6', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K', 
    'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese', 'Phosphorus', 'Potassium', 
    'Selenium', 'Zinc', 'Nutrition Density'
]

# Membuat selection input untuk masing-masing fitur dengan opsi 'Low', 'Medium', 'High'
input_data = []
for feature in features:
    option = st.selectbox(f"Pilih {feature}", ['Low', 'Medium', 'High'])
    
    # Menentukan nilai fitur berdasarkan pilihan
    if option == 'Low':
        input_value = 0
    elif option == 'Medium':
        input_value = 50
    elif option == 'High':
        input_value = 100
    
    input_data.append(input_value)

# Siapkan input pengguna untuk prediksi klaster
input_data = np.array(input_data).reshape(1, -1)

# Menstandarisasi input pengguna menggunakan scaler yang sama
scaler = StandardScaler()
scaler.fit(data_scaled)  # Fitting scaler pada data yang sudah ada
input_data_scaled = scaler.transform(input_data)

# Prediksi klaster berdasarkan input pengguna
predicted_cluster = kmeans.predict(input_data_scaled)

# Menampilkan hasil prediksi
st.write(f"Klaster yang sesuai: {cluster_labels[predicted_cluster[0]]}")

# Menampilkan rekomendasi dessert berupa nama-nama dessert
st.write("Rekomendasi dessert berdasarkan klaster ini:")

# Ambil nama-nama dessert yang termasuk dalam klaster yang diprediksi
recommended_desserts = [desserts[i] for i in range(len(desserts)) if clusters[i] == predicted_cluster[0]]

# Tampilkan daftar nama dessert
st.write(recommended_desserts)
