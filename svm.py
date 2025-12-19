import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load dataset dari file lokal
df = pd.read_csv(r'C:\Users\femmy\OneDrive\Documents\6. Semester 6\Machine Learning\praktek\Social_Network_Ads.csv')

# Split train data --> features= X & target= y
X = df.drop(['User ID', 'Gender'], axis=1)
y = df['Purchased']

# Split X & Y --> data-train & data-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Perform Feature Scaling to normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit SVM to the Training set
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predict the Test Set Results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(cm)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")


#svm binary dan svm multiclass