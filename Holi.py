import streamlit as st
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
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

# Fitur yang digunakan dalam klasterisasi
all_features = [
    'Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats', 
    'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
    'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2', 'Vitamin B3', 
    'Vitamin B5', 'Vitamin B6', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K', 
    'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese', 'Phosphorus', 'Potassium', 
    'Selenium', 'Zinc', 'Nutrition Density'
]

# Fungsi untuk menghitung nilai Low, Medium, dan High berdasarkan min dan max fitur
def calculate_feature_range(feature_values):
    min_value = np.min(feature_values)
    max_value = np.max(feature_values)
    medium_value = (min_value + max_value) / 2
    return min_value, medium_value, max_value

# Menentukan rentang nilai untuk setiap fitur
feature_ranges = {}
for i, feature in enumerate(all_features):
    feature_ranges[feature] = calculate_feature_range(data_scaled[:, i])

# Fungsi untuk mengonversi input pengguna menjadi nilai berdasarkan Low, Medium, dan High
def get_value_for_level(level, feature_range):
    min_value, medium_value, max_value = feature_range
    if level == 'Low':
        return min_value
    elif level == 'Medium':
        return medium_value
    elif level == 'High':
        return max_value

# Input selection untuk setiap fitur
user_inputs = {}
for feature in all_features:
    user_inputs[feature] = st.selectbox(f"Tingkat {feature}", ['Low', 'Medium', 'High'], index=1)

# Menyusun input pengguna berdasarkan pilihan untuk semua fitur
input_data = []
for feature in all_features:
    input_data.append(get_value_for_level(user_inputs[feature], feature_ranges[feature]))

# Menyusun input menjadi array
input_data = np.array(input_data).reshape(1, -1)

# Menstandarisasi input pengguna menggunakan scaler yang sama
scaler = StandardScaler()
scaler.fit(data_scaled)  # Fitting scaler pada data yang sudah ada
input_data_scaled = scaler.transform(input_data)

# Tombol untuk memulai klasifikasi
if st.button('Klasifikasi'):
    # Prediksi klaster berdasarkan input pengguna
    predicted_cluster = kmeans.predict(input_data_scaled)

    # Menampilkan hasil prediksi
    st.write(f"Klaster yang sesuai: {cluster_labels[predicted_cluster[0]]}")

    # Menggunakan NearestNeighbors untuk menemukan tetangga terdekat berdasarkan hasil klasifikasi
    neighbors = NearestNeighbors(n_neighbors=3, metric='euclidean')
    neighbors.fit(data_scaled)  # Latih NearestNeighbors dengan data yang sudah ada

    # Menemukan 3 tetangga terdekat berdasarkan input pengguna yang telah distandarisasi
    distances, indices = neighbors.kneighbors(input_data_scaled)

    # Menampilkan rekomendasi dessert berdasarkan tetangga terdekat
    st.write("3 Rekomendasi Dessert berdasarkan klaster ini:")
    recommended_desserts = [food[i] for i in indices[0]]
    st.write(recommended_desserts)


