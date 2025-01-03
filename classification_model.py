
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Hedef ve bağımsız değişkenlerin ayrılması
df = pd.read_csv('selected_volatility.csv')
X = df.drop("Volatility_24h", axis=1)
y = df["Volatility_24h"]

# 1. Volatility_24h değerlerini kategorilere ayırarak sınıflandırma
threshold = y.quantile(0.75)  # %75'lik eşik değeri
print(f"Volatility_24h için %75'lik eşik değer: {threshold}")

y_classified = np.where(y > threshold, 1, 0)  # 1: Riskli, 0: Risksiz

# 3. SMOTE ile Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_classified)

# 4. Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. Verilerin Ölçeklendirilmesi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Makine Öğrenmesi Modelleri
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = {}
roc_curves = {}

for model_name, model in models.items():
    # Modeli eğit
    model.fit(X_train_scaled, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Pozitif sınıf olasılıkları
    else:
        y_prob = model.decision_function(X_test_scaled)

    # Değerlendirme metrikleri
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_prob)
    results[model_name] = {
        "Accuracy": report["accuracy"],
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"],
        "ROC AUC": auc_score
    }

    # ROC Eğrisi
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[model_name] = (fpr, tpr, auc_score)

# Sonuçları Analiz Et
print("\nModel Performans Karşılaştırması:")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy = {metrics['Accuracy']:.4f}, Precision = {metrics['Precision']:.4f}, Recall = {metrics['Recall']:.4f}, F1-Score = {metrics['F1-Score']:.4f}, ROC AUC = {metrics['ROC AUC']:.4f}")

# En İyi Modeli Kaydetme
best_model_name = max(results, key=lambda k: results[k]["ROC AUC"])
print(f"\nEn iyi model: {best_model_name}")

# ROC AUC Grafiği
plt.figure(figsize=(10, 6))
for model_name, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve")
plt.legend()
plt.grid()
plt.show()

# # modellerin yorumlanması
# Volatilite tahmini için:


# Decision Tree ve Random Forest modelleriniz çok iyi çalışıyor
# SVR tahmin modelli için uygun değildir
# Linear Regression ile doğru veri volatility_24h tahmini yapılamiyor
# Ağaç bazlı modeller (Decision Tree, Random Forest) bu veri seti için en uygun görünüyor


# Risk sınıflandırması için:


# Random Forest açık ara en iyi performansı gösteriyor
# Tüm modeller kabul edilebilir seviyede çalışıyor