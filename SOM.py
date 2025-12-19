import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# Data dari soal (SOM manual)
X = np.array([
    [0.10, 0.10],
    [0.20, 0.20],
    [0.30, 0.10],
    [0.50, 0.30],
    [0.40, 0.40],
    [0.20, 0.40]
])

# Inisialisasi bobot: gunakan sedikit noise agar tidak simetris
np.random.seed(0)
weights = np.array([
    [0.5, 0.5],
    [0.5, 0.5]
]) + 0.01 * np.random.randn(2, 2)

# Parameter
initial_lr = 0.5        # simpan learning rate awal
epoch_max = 10

# simpan salinan bobot awal (untuk keperluan perhitungan/banding)
initial_weights = weights.copy()

def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

print("Bobot awal:\n", weights, "\n")

for epoch in range(epoch_max):
    # learning rate terbaru: α_epoch = initial_lr * (0.5 ** epoch)
    epoch_lr = initial_lr * (0.5 ** epoch)
    print(f"=== Epoch {epoch+1} | learning rate = {epoch_lr} ===")
    # acak urutan data tiap epoch untuk menghindari pola tetap
    order = np.random.permutation(len(X))
    for idx in order:
        x = X[idx]
        # Hitung jarak setiap neuron terhadap data x
        distances = np.array([euclidean(x, w) for w in weights])

        # Winner neuron: jika ada tie, pilih secara acak di antara minima
        minima = np.flatnonzero(distances == distances.min())
        winner = np.random.choice(minima)

        # Update bobot winner menggunakan learning rate pada epoch ini
        weights[winner] = weights[winner] + epoch_lr * (x - weights[winner])

        # Tampilkan bobot_awal * learning_rate_terbaru seperti yang diminta
        scaled_initial = initial_weights[winner] * epoch_lr
        print(f"Data {idx+1} {x} → Winner: C{winner+1}")
        print("Bobot awal (C{0}): {1}".format(winner+1, initial_weights[winner]))
        print("Bobot awal dikali learning rate terbaru:", scaled_initial)
        print("Bobot baru:\n", weights, "\n")

    # Jangan break; jalankan hingga epoch_max (decay sudah diatur lewat epoch_lr)

print("=== Bobot akhir ===")
print(weights)


# ====== MiniSom example (optional, but safe-guarded) ======
try:
    from minisom import MiniSom
    # Load a sample dataset (Iris) and scale
    iris = datasets.load_iris()
    data = iris.data
    scaled_data = scale(data)

    # Inisialisasi SOM 10x10
    som = MiniSom(x=10, y=10, input_len=scaled_data.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(scaled_data)
    som.train_random(scaled_data, num_iteration=100)

    # Visualize U-matrix
    plt.figure(figsize=(8, 8))
    plt.pcolor(som.distance_map().T, cmap='bone_r')   # U-Matrix
    plt.colorbar()

    # Map data points to their BMUs and visualize them
    for i, x in enumerate(scaled_data):
        w = som.winner(x)   # BMU coordinates (row, col)
        plt.plot(w[0] + 0.5, w[1] + 0.5, 'o',
                 markerfacecolor='None',
                 markeredgecolor=plt.cm.get_cmap('viridis')(iris.target[i] / 3.0),
                 markersize=8, markeredgewidth=1.5)
    plt.title('SOM U-Matrix with data mapped to BMUs')
    plt.show()

except ImportError:
    print("MiniSom tidak terpasang. Untuk menggunakan MiniSom, jalankan di terminal:")
    print("    pip install MiniSom")
    print("    pip install MiniSom")
