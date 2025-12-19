# =====================================================
# PRAKTIK K-MEANS CLUSTERING
# Dataset: Breast Cancer Wisconsin (Original)
# Nama : Femmy Marisa Nurjanah
# NPM  : 406222000015
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, silhouette_score

# =========================
# 1. LOAD DATASET
# =========================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

columns = [
    "ID",
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "Class"
]

df = pd.read_csv(url, names=columns)

# Ganti '?' dengan NaN lalu drop
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Konversi tipe data
for col in columns[1:]:
    df[col] = df[col].astype(int)

print("Nama  : Femmy Marisa Nurjanah")
print("NPM   : 406222000015")
print("Jumlah data:", df.shape[0])

# =========================
# 2. PREPROCESSING
# =========================

X = df.iloc[:, 1:10]
y = df["Class"].replace({2: 0, 4: 1})  # 0=Benign, 1=Malignant

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. K-MEANS k = 2
# =========================

kmeans_2 = KMeans(n_clusters=2, n_init=10, random_state=42)
cluster_2 = kmeans_2.fit_predict(X_scaled)

print("\nK=2")
print("Inertia:", kmeans_2.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, cluster_2))

# Mapping cluster ke label asli
mapping = {}
for c in np.unique(cluster_2):
    majority = y[cluster_2 == c].mode()[0]
    mapping[c] = majority

pred_2 = np.array([mapping[c] for c in cluster_2])

cm_2 = confusion_matrix(y, pred_2)
print("Confusion Matrix (k=2):\n", cm_2)

purity_2 = np.sum(np.diag(cm_2)) / np.sum(cm_2)
print("Purity (k=2):", purity_2)

# =========================
# 4. VISUALISASI k = 2
# =========================

plt.figure()
plt.scatter(
    df["Clump Thickness"],
    df["Uniformity of Cell Size"],
    c=cluster_2
)

centroids_2 = scaler.inverse_transform(kmeans_2.cluster_centers_)
plt.scatter(
    centroids_2[:, 0],
    centroids_2[:, 1],
    marker='X',
    s=200
)

plt.xlabel("Clump Thickness")
plt.ylabel("Uniformity of Cell Size")
plt.title("K-Means Clustering (k=2)\nFemmy Marisa Nurjanah - 406222000015")
plt.show()

# =========================
# 5. EKSPERIMEN k = 3 dan k = 4
# =========================

for k in [3, 4]:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster = kmeans.fit_predict(X_scaled)

    print(f"\nK={k}")
    print("Inertia:", kmeans.inertia_)
    print("Silhouette Score:", silhouette_score(X_scaled, cluster))

    plt.figure()
    plt.scatter(
        df["Clump Thickness"],
        df["Uniformity of Cell Size"],
        c=cluster
    )

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='X',
        s=200
    )

    plt.xlabel("Clump Thickness")
    plt.ylabel("Uniformity of Cell Size")
    plt.title(f"K-Means Clustering (k={k})\nFemmy Marisa Nurjanah - 406222000015")
    plt.show()


# =========================
# 6. RUN K-MEANS 5 KALI (n_init=1)
# =========================

print("\n=== EKSPERIMEN 5 KALI RUN (n_init=1) ===")

inertia_list = []
centroid_list = []

for i in range(5):
    kmeans = KMeans(n_clusters=2, n_init=1, random_state=i)
    labels = kmeans.fit_predict(X_scaled)
    inertia_list.append(kmeans.inertia_)
    centroid_list.append(kmeans.cluster_centers_)

    print(f"Run ke-{i+1}")
    print("Inertia:", kmeans.inertia_)
    print("Centroid (scaled):")
    print(kmeans.cluster_centers_)
    print("----------------------")

# =========================
# 7. VISUALISASI HASIL TERBAIK (INERTIA TERKECIL)
# =========================

best_index = np.argmin(inertia_list)
print("Run terbaik (inertia terkecil): Run ke-", best_index + 1)

best_centroids = scaler.inverse_transform(centroid_list[best_index])
best_labels = KMeans(n_clusters=2, n_init=1, random_state=best_index).fit_predict(X_scaled)

plt.figure()
plt.scatter(
    df["Clump Thickness"],
    df["Uniformity of Cell Size"],
    c=best_labels
)

plt.scatter(
    best_centroids[:, 0],
    best_centroids[:, 1],
    marker='X',
    s=200
)

plt.xlabel("Clump Thickness")
plt.ylabel("Uniformity of Cell Size")
plt.title(
    f"K-Means k=2 (Run Terbaik dari 5x Percobaan)\n"
    f"Femmy Marisa Nurjanah - 406222000015"
)
plt.show()

# =========================
# 8. HASIL AKHIR (n_init = 10)
# =========================

print("\n=== HASIL AKHIR K-MEANS (n_init = 10) ===")

kmeans_final = KMeans(n_clusters=2, n_init=10, random_state=42)
final_labels = kmeans_final.fit_predict(X_scaled)

print("Inertia akhir:", kmeans_final.inertia_)
print("Centroid akhir (scaled):")
print(kmeans_final.cluster_centers_)

final_centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)

plt.figure()
plt.scatter(
    df["Clump Thickness"],
    df["Uniformity of Cell Size"],
    c=final_labels
)

plt.scatter(
    final_centroids[:, 0],
    final_centroids[:, 1],
    marker='X',
    s=200
)

plt.xlabel("Clump Thickness")
plt.ylabel("Uniformity of Cell Size")
plt.title(
    "K-Means k=2 (Hasil Akhir n_init=10)\n"
    "Femmy Marisa Nurjanah - 406222000015"
)
plt.show()