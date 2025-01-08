import streamlit as st
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Fungsi untuk memuat data klasterisasi dari file .h5
def load_clusters(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data_scaled = h5_file['data_scaled'][:]
        clusters = h5_file['clusters'][:]
        centroids = h5_file['centroids'][:]
    return data_scaled, clusters, centroids

# Memuat model klasterisasi
file_path = "dessertnutrition_clusters.h5"  
data_scaled, clusters, centroids = load_clusters(file_path)

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
input_data = np.array([sugar_level, protein_level, vitamin_a, vitamin_c, vitamin_d]).reshape(1, -1)

# Standarisasi input pengguna
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Prediksi klaster berdasarkan input pengguna
kmeans = KMeans(n_clusters=3, random_state=42)
predicted_cluster = kmeans.predict(input_data_scaled)

# Menampilkan hasil prediksi
st.write(f"Klaster yang sesuai: {cluster_labels[predicted_cluster[0]]}")

# Menampilkan rekomendasi
st.write("Rekomendasi dessert berdasarkan klaster ini:")
# Filter dataset untuk hanya menampilkan dessert yang termasuk dalam klaster yang diprediksi
# (Pada aplikasi nyata, kita akan memilih dessert berdasarkan klaster dan preferensi)
recommended_desserts = pd.DataFrame(data_scaled, columns=["Caloric Value", "Protein", "Sugars", "Fat"])  # contohnya
st.dataframe(recommended_desserts)
