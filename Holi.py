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

# Input slider untuk 34 fitur
features = [
    'Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats',
    'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
    'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2', 'Vitamin B3', 
    'Vitamin B5', 'Vitamin B6', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K', 
    'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese', 'Phosphorus', 'Potassium', 
    'Selenium', 'Zinc', 'Nutrition Density'
]

# Membuat slider input untuk masing-masing fitur
input_data = []
for feature in features:
    if feature in ['Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats', 
                   'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water']:
        # Asumsikan nilai fitur ini dalam rentang yang sesuai
        min_value, max_value = 0, 100
    elif feature in ['Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2', 'Vitamin B3', 
                     'Vitamin B5', 'Vitamin B6', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K']:
        # Vitamin biasanya memiliki rentang nilai antara 0 dan 100
        min_value, max_value = 0, 100
    else:
        # Minerals biasanya memiliki rentang yang lebih kecil
        min_value, max_value = 0, 50

    input_value = st.slider(f"Pilih {feature}", min_value=min_value, max_value=max_value, value=(min_value + max_value) // 2)
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

# Menampilkan rekomendasi
st.write("Rekomendasi dessert berdasarkan klaster ini:")
# Filter dataset untuk hanya menampilkan dessert yang termasuk dalam klaster yang diprediksi
# (Pada aplikasi nyata, kita akan memilih dessert berdasarkan klaster dan preferensi)
recommended_desserts = pd.DataFrame(data_scaled, columns=features)  # Menggunakan nama fitur yang sama
st.dataframe(recommended_desserts)
