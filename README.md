# Volatility24h Tahmin ve Sınıflandırma Modelleri

Bu proje, finansal piyasalardaki bir varlığın 24 saatlik volatilitesini tahmin etmek ve ardından risk durumunu sınıflandırmak amacıyla iki makine öğrenmesi modeli içerir. Proje, hem regresyon hem de sınıflandırma problemlerini çözmeyi hedeflemektedir.

## Projenin Genel Yapısı

### Verinin İşlenmesi

1. **Verinin Yüklenmesi**
   - CSV formatındaki Bitcoin veri seti, `pandas` kullanılarak yüklenir ve ilk özelliklere genel bakış `df.info()` ile sağlanır.

2. **Özelliklerin Ölçeklendirilmesi**
   - Min-Max Scaling: `Volatility_24h` sütunu 0-1 arasında normalleştirildi.

3. **Tarih ve Veri Türü Formatı Dönüşümü**
   - `Open Time` sütunu datetime formatına dönüştürüldü.
   - `Volatility_24h` sütunu float formatına getirildi.

4. **Veri Görselleştirme**
   - `Volatility_24h` değişkeni zaman serisi grafiği olarak çizildi.

### Korelasyon Analizi

1. **Korelasyon Matrisi**
   - `Volatility_24h` ile en yüksek korelasyona sahip sütunlar belirlendi.

2. **Yüksek Korelasyonlu Sütunların Seçilmesi**
   - Korelasyon eşik değeri olarak 0.2 belirlendi ve bu eşik üzerindeki sütunlar öne çıkarıldı.

3. **Null (Eksik) Değerlerin Doldurulması**
   - Ortalama: Bazı sütunlar verilerin ortalama değerleriyle dolduruldu.
   - Medyan : Bazı sutunlar sutundaki en çok tekrar eden değerle dolduruldu.
   - Forward Fill: Önceki bilinen değerle doldurma.
   - Interpolasyon: Doğrusal interpolasyon işlemi uygulandı.

4. **Hareketli Ortalamalar ve Standart Sapma**
   - Bazı özellikler için hareketli ortalamalar ve standart sapmalar yeniden hesaplandı.

### Özellik Seçimi

1. **Random Forest Özellik Önem Skoru**
   - Random Forest algoritması, özelliklerin göreceli önem seviyelerini hesaplamak için kullanıldı.
   - Önem seviyesi 0.006'nin altındaki sütunlar çıkarıldı.

2. **Varyans Analizi (F-Test)**
   - F-Test analizine göre anlamsız özellikler çıkarıldı.

3. **Fazla Benzerlik Gösteren Özelliklerin Çıkarılması**
   - Aşırı benzerlik (BB_Lower_4h ve Volatility_8h gibi) gösteren sütunlar elendi.

### Uç Değerlerin Temizlenmesi

- Z-Skoru (eşik: 3) kullanılarak uç değerler temizlendi.

## Makine Öğrenmesi Modelleri

### Regresyon Modelleri

1. Aşağıdaki regresyon modelleri kullanıldı:
   - Linear Regression
   - Random Forest Regressor
   - Support Vector Regressor (SVR)
   - Decision Tree Regressor
   - KNN Regressor

2. **Değerlendirme Metrikleri**
   - Mean Squared Error (MSE): Hata oranının küçük olması beklenir.
   - R² Skoru: Modelin açıklayıcılık oranı.

Sonuç olarak, en iyi regresyon modeli Random Forest Regressor oldu.

### Sınıflandırma Modelleri

1. **Hedef Değişkenin Dönüştürülmesi**
   - `Volatility_24h`, %75 eşik değerine göre riskli (1) ve risksiz (0) olarak sınıflandırıldı.

2. **SMOTE Kullanılarak Oversampling**
   - Dengeli bir veri seti oluşturmak için SMOTE (Synthetic Minority Oversampling Technique) uygulandı.

3. **Sınıflandırma Modelleri**
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - Decision Tree Classifier
   - KNN Classifier

4. **ROC AUC Analizi**
   - Her modelin ROC AUC skorları hesaplandı ve en iyi performansı Random Forest Classifier modeli sağladı.

## Görselleştirmeler

1. **Korelasyon Görselleştirmesi**
   - Bağımsız değişkenler ile hedef değişken arasındaki korelasyon görselleştirildi.

2. **ROC AUC Grafiği**
   - Her modelin ROC eğrisi çizildi.

## Sonuçlar

- **Regresyon Modelleri**: Random Forest Regressor, en iyi performansı sağladı.
- **Sınıflandırma Modelleri**: Random Forest Classifier, en yüksek ROC AUC skoruna ulaştı.

## Dosyalar

- `high_correlation_with_volatility.csv`: Yüksek korelasyona sahip sütunları içeren dosya.
- `selected_volatility.csv`: Nihai özellik seti ve hedef değişkeni.

---

## Gereksinimler

Proje aşağıdaki yazılım ve kütüphaneleri gerektirir:

### Veri Manipülasyonu ve Analizi

- `pandas`: Veri çerçevelerinin oluşturulması ve işlenmesi.
- `numpy`: Sayısal hesaplamalar ve matris işlemleri.

### Görselleştirme

- `matplotlib`: Grafikler ve veri görselleştirmeleri oluşturmak için.
- `seaborn`: İleri düzey grafikler ve korelasyon analizleri.

### Makine Öğrenmesi

- `scikit-learn`: Makine öğrenmesi algoritmaları, veri ön işleme ve metrikler.
- `imblearn`: SMOTE ve SMOTEENN gibi dengesiz veri setlerini dengeleme yöntemleri.

### Modelleme

- `RandomForestClassifier`: Sınıflandırma problemleri için rastgele orman algoritması.
- `LogisticRegression`: Lojistik regresyon modeli.
- `SVC`: Destek vektör makineleri (sınıflandırma ve regresyon).
- `RandomForestRegressor`: Regresyon problemleri için rastgele orman algoritması.
- `LinearRegression`: Basit doğrusal regresyon modeli.
- `DecisionTreeRegressor`: Karar ağaçları regresyon modeli.
- `KNeighborsRegressor`: K-en yakın komşu algoritması (regresyon).

### Ölçeklendirme ve Özellik Seçimi

- `StandardScaler`: Özelliklerin ölçeklendirilmesi ve standartlaştırılması.
- `f_regression`: Özelliklerin hedef değişkenle ilişkisini ölçmek için F-istatistiği tabanlı yöntem.

### İstatistik ve Veri Dönüşümü

- `zscore`: Uç değer analizi için Z-skoru.
- `scipy.stats`: İstatistiksel testler ve analizler.

---

## Kurulum

1. Bu projeyi klonlayın:
   ```bash
   git clone https://github.com/Kenanyilmaz55/volatility24h.git
   cd volatility24h
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Verilerinizi `btc_veri.csv` dosyasına uygun formatta sağlayın.

---

## Kullanım

### 1. Volatility24h Tahmini (Regresyon)

Regresyon modelini çalıştırmak için aşağıdaki adımları izleyin:

- btc_veri.csv dosyasini indirin
- Veri ön işleme: Eksik değerlerin doldurulması, uç değerlerin temizlenmesi ve özellik seçimi gerçekleştirilir.
- Modellerin eğitilmesi:
  ```bash
  python regression_model.py
  ```
- Çıktılar: Her model için Mean Squared Error (MSE) ve R² skorları hesaplanır.
- bu kod çalıştıktan sonra selected_volatility.csv ve high_correlation_with_volatility.csv dosyaları oluşturulur
- iki modelde de eğtilen veriler selected_volatlity.csv dosyasındaki veriler kullanılarak yapılır

### 2. Risk Durumu Sınıflandırması (Sınıflandırma)

Sınıflandırma modelini çalıştırmak için aşağıdaki adımları izleyin:

- Veriler, riskli (1) ve risksiz (0) olarak kategorilere ayrılır.
- SMOTE kullanılarak dengesiz sınıflar dengelenir.
- Modeller eğitilir:
  ```bash
  python classification_model.py
  ```
- Çıktılar: Her model için Accuracy, Precision, Recall, F1-Score ve ROC AUC değerleri hesaplanır.

---

### Model Performansının Değerlendirilmesi

- En iyi model regresyon ve sınıflandırma görevleri için ayrı ayrı seçilir.
- ROC eğrisi ve diğer metrikler görselleştirilir.

---

## Sonuçların Kaydedilmesi

- Yüksek korelasyona sahip özellikler ve seçilen nihai özellikler şu dosyalara kaydedilir:
  - `high_correlation_with_volatility.csv`
  - `selected_volatility.csv`

