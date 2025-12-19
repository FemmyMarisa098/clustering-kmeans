from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y, title):
    # Buat grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Prediksi setiap titik grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Gambar contour dan titik data
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1 (normalized)')
    plt.ylabel('Feature 2 (normalized)')
    plt.show()

# === 1. Load dataset ===
iris = datasets.load_iris()
X = iris.data[:, :2]  # hanya 2 fitur pertama (sepal length, sepal width)
y = iris.target

# === 2. Normalisasi data ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 3. Split data ===
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, train_size=0.80, test_size=0.20, random_state=101
)

# === 4. Train model ===
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

# === 5. Prediksi ===
rbf_pred = rbf.predict(X_test)
poly_pred = poly.predict(X_test)

# === 6. Evaluasi ===
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')

print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy * 100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1 * 100))
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy * 100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1 * 100))

# Visualisasi hasil
plot_decision_boundary(poly, X, y, f'Polynomial Kernel (Acc: {poly_accuracy * 100:.2f}%)')
plot_decision_boundary(rbf, X, y, f'RBF Kernel (Acc: {rbf_accuracy * 100:.2f}%)')




