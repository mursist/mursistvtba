import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_style('whitegrid')

# ----------------------------------------------------------------------------
# ÖRNEK 1: ZAMAN SERİSİ ANALİZİ VE SATIŞ TAHMİNİ
# ----------------------------------------------------------------------------

def create_sample_sales_data(n_days=1095):
    """3 yıllık yapay satış verisi oluşturur"""
    
    # Tarih aralığı oluştur (3 yıl)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    # Trend bileşeni
    trend = np.linspace(100, 300, n_days)
    
    # Mevsimsellik bileşeni (haftalık ve yıllık)
    weekly_seasonality = 20 * np.sin(np.arange(n_days) * (2 * np.pi / 7))
    yearly_seasonality = 100 * np.sin(np.arange(n_days) * (2 * np.pi / 365))
    
    # Tatil etkisi
    holidays = pd.Series(0, index=dates)
    # Yılbaşı, Ramazan ve Kurban Bayramı
    holidays['2022-01-01'] = 100
    holidays['2022-05-02'] = 120
    holidays['2022-05-03'] = 150
    holidays['2022-05-04'] = 120
    holidays['2022-07-09'] = 120
    holidays['2022-07-10'] = 150
    holidays['2022-07-11'] = 140
    holidays['2022-07-12'] = 110
    
    holidays['2023-01-01'] = 110
    holidays['2023-04-21'] = 130
    holidays['2023-04-22'] = 160
    holidays['2023-04-23'] = 130
    holidays['2023-06-28'] = 130
    holidays['2023-06-29'] = 160
    holidays['2023-06-30'] = 150
    holidays['2023-07-01'] = 120
    
    holidays['2024-01-01'] = 120
    holidays['2024-04-10'] = 140
    holidays['2024-04-11'] = 170
    holidays['2024-04-12'] = 140
    holidays['2024-06-16'] = 140
    holidays['2024-06-17'] = 170
    holidays['2024-06-18'] = 160
    holidays['2024-06-19'] = 130
    
    # Promosyon etkisi
    promotions = pd.Series(0, index=dates)
    # Yılda 4 kez büyük promosyon (her 3 ayda bir)
    for year in [2022, 2023, 2024]:
        for month in [3, 6, 9, 12]:
            start_date = pd.Timestamp(f"{year}-{month:02d}-01")
            end_date = start_date + pd.Timedelta(days=6)
            mask = (dates >= start_date) & (dates <= end_date)
            promotions[mask] = 80
    
    # Rastgele gürültü
    noise = np.random.normal(0, 20, n_days)
    
    # Tüm bileşenleri birleştir
    sales = trend + weekly_seasonality + yearly_seasonality + holidays.values + promotions.values + noise
    
    # Veri çerçevesi oluştur
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'weekday': dates.dayofweek,
        'month': dates.month,
        'year': dates.year,
        'is_weekend': dates.dayofweek >= 5,
        'is_holiday': holidays > 0,
        'is_promotion': promotions > 0,
        'day_of_year': dates.dayofyear
    })
    
    # Satış değerlerini pozitif yap
    df['sales'] = df['sales'].clip(lower=0)
    
    return df

def analyze_time_series(df):
    """Zaman serisi analizi yapar ve sonuçları döndürür"""
    
    # Tarih sütununu dizin olarak ayarla
    df_ts = df.copy()
    df_ts.set_index('date', inplace=True)
    
    # Mevsimsel ayrıştırma
    try:
        result = seasonal_decompose(df_ts['sales'], model='additive', period=30)
        return result
    except Exception as e:
        print(f"Zaman serisi analizi sırasında hata oluştu: {e}")
        return None

def forecast_sales(df, forecast_days=30):
    """ARIMA modeli ile satış tahmini yapar"""

    # Tarihi dizin olarak ayarla
    df_forecast = df.copy()
    df_forecast.set_index('date', inplace=True)

    # ARIMA modeli için sadece satış verilerini al
    sales_series = df_forecast['sales']

    try:
        # ARIMA parametreleri (p, d, q)
        model = ARIMA(sales_series, order=(5, 1, 2))
        model_fit = model.fit()

        # Tahmin ve güven aralığı
        forecast_object = model_fit.get_forecast(steps=forecast_days)
        forecast = forecast_object.predicted_mean

        forecast_index = pd.date_range(start=sales_series.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_series = pd.Series(forecast.values, index=forecast_index)

        return forecast_series
    except Exception as e:
        print(f"Satış tahmini sırasında hata oluştu: {e}")
        return None

def train_ml_sales_model(df):
    """Makine öğrenmesi modeli ile satış tahmini"""
    
    try:
        # Özellikler ve hedef değişken
        X = df.drop(['sales', 'date'], axis=1)
        y = df['sales']
        
        # One-hot encoding için kategorik değişkenler
        categorical_features = []
        numeric_features = ['weekday', 'month', 'year', 'day_of_year']
        binary_features = ['is_weekend', 'is_holiday', 'is_promotion']
        
        # Veri ön işleme pipeline'ı
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('bin', 'passthrough', binary_features)
            ])
        
        # Eğitim ve test verileri
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # RandomForest modeli
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # XGBoost modeli
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42))
        ])
        
        # Modelleri eğit
        rf_pipeline.fit(X_train, y_train)
        xgb_pipeline.fit(X_train, y_train)
        
        return rf_pipeline, xgb_pipeline
    except Exception as e:
        print(f"ML modeli eğitimi sırasında hata oluştu: {e}")
        return None, None

# ----------------------------------------------------------------------------
# ÖRNEK 2: ANOMALİ TESPİTİ VE MÜŞTERİ SEGMENTASYONU
# ----------------------------------------------------------------------------

def create_customer_data(n_customers=1000):
    """Müşteri segmentasyonu için örnek veri oluşturur"""
    
    # Ana müşteri segmentleri için merkezler
    centers = [
        [5000, 15, 0.3],  # Yüksek harcama, orta sıklık, düşük iade
        [1000, 30, 0.1],  # Düşük harcama, yüksek sıklık, çok düşük iade
        [8000, 5, 0.05],  # Çok yüksek harcama, düşük sıklık, çok düşük iade
        [500, 2, 0.5],    # Çok düşük harcama, çok düşük sıklık, yüksek iade (potansiyel anomali)
        [3000, 12, 0.2]   # Orta harcama, orta sıklık, düşük iade
    ]
    
    # Cluster büyüklükleri (toplam 1000 müşteri)
    sizes = [300, 250, 50, 100, 300]
    
    # Veri özellikler
    features = ['avg_purchase_value', 'purchase_frequency', 'return_rate']
    
    # Her segment için veri oluştur
    segments = []
    segment_labels = []
    
    for i, (center, size) in enumerate(zip(centers, sizes)):
        # Her özellik için rastgele dağılım
        std_devs = [center[0] * 0.2, center[1] * 0.3, center[2] * 0.1]
        
        # Rastgele nokta oluştur
        segment_data = np.random.normal(loc=center, scale=std_devs, size=(size, len(features)))
        
        # Negatif değerleri düzelt
        segment_data = np.abs(segment_data)
        
        # Return rate'i 0-1 arasına sınırla
        segment_data[:, 2] = np.clip(segment_data[:, 2], 0, 1)
        
        segments.append(segment_data)
        segment_labels.extend([i] * size)
    
    # Tüm segmentleri birleştir
    data = np.vstack(segments)
    
    # Müşteri ID'leri oluştur
    customer_ids = [f'CUST_{i:05d}' for i in range(1, n_customers + 1)]
    
    # Diğer özellikler ekle
    loyalty_years = np.random.uniform(0, 10, size=n_customers)
    avg_basket_size = np.random.uniform(1, 15, size=n_customers)
    pct_discount_used = np.random.uniform(0, 0.7, size=n_customers)
    
    # Veri çerçevesi oluştur
    df = pd.DataFrame(data, columns=features)
    df['customer_id'] = customer_ids
    df['loyalty_years'] = loyalty_years
    df['avg_basket_size'] = avg_basket_size
    df['pct_discount_used'] = pct_discount_used
    df['true_segment'] = segment_labels
    
    # İleride analiz için Kümülatif Değer oluştur
    df['customer_value'] = df['avg_purchase_value'] * df['purchase_frequency'] * (1 - df['return_rate']) * (1 + df['loyalty_years'] * 0.1)
    
    return df

def detect_customer_anomalies(df):
    """Müşteri verilerinde anomali tespiti yapar"""
    
    try:
        # Anomali tespiti için kullanılacak özellikler
        features = ['avg_purchase_value', 'purchase_frequency', 'return_rate', 
                    'loyalty_years', 'avg_basket_size', 'pct_discount_used']
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        
        # Isolation Forest modeli
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(X_scaled)
        df['anomaly_score'] = iso_forest.score_samples(X_scaled)
        
        # Anomalileri -1, normal gözlemleri 1 olarak işaretler
        # Bunu 0 ve 1 olarak değiştirelim (1: anomali, 0: normal)
        df['anomaly'] = [1 if x == -1 else 0 for x in df['anomaly']]
        
        return df
    except Exception as e:
        print(f"Anomali tespiti sırasında hata oluştu: {e}")
        return df

def segment_customers(df, n_clusters=4):
    """K-means ile müşteri segmentasyonu yapar"""
    
    try:
        # Segmentasyon için kullanılacak özellikler
        features = ['avg_purchase_value', 'purchase_frequency', 'return_rate', 
                    'loyalty_years', 'customer_value']
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        
        # K-means modeli
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        return df, kmeans, scaler
    except Exception as e:
        print(f"Müşteri segmentasyonu sırasında hata oluştu: {e}")
        return df, None, None

# ----------------------------------------------------------------------------
# ÖRNEK 3: TEKNOLOJİK ÜRÜNLER İÇİN ÖNERİ MOTORU (İÇERİK TABANLI)
# ----------------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def create_tech_product_data():
    """Farklı kategorilerde teknolojik ürün verisi"""
    return pd.DataFrame({
        'product_id': [101, 102, 103, 104, 105, 106, 107],
        'product_name': [
            'Gaming Laptop', 'Ultrabook', 'Akıllı Telefon',
            'Tablet', 'Masaüstü Bilgisayar', 'Kulaklık', 'Akıllı Saat'
        ],
        'description': [
            'Intel i7 işlemci, 16GB RAM, NVIDIA RTX 3060 ekran kartı, 1TB SSD ile yüksek performanslı oyun deneyimi.',
            'Intel i5 işlemci, 8GB RAM, hafif tasarım, 512GB SSD, uzun pil ömrü ile taşınabilirlik odaklı.',
            'Snapdragon işlemci, 128GB hafıza, 6.5 inç ekran, Android 13, 5000mAh batarya ile güçlü akıllı telefon.',
            '10.1 inç ekran, 4GB RAM, 64GB depolama, hafif ve taşınabilir, Android tabanlı tablet.',
            'Ryzen 5 işlemci, 32GB RAM, 2TB SSD, 4K destekli ekran kartı ile ofis ve oyun için masaüstü bilgisayar.',
            'Kablosuz bluetooth kulaklık, aktif gürültü engelleme, 40 saat pil ömrü.',
            '1.43 inç AMOLED ekran, kalp ritmi takibi, adım sayar, 7 gün pil ömrü ile akıllı saat.'
        ]
    })

def recommend_similar_tech_products(df, product_id, top_n=3):
    """İçerik tabanlı öneri üretir"""
    try:
        tfidf = TfidfVectorizer(stop_words='turkish')
        tfidf_matrix = tfidf.fit_transform(df['description'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        idx = df.index[df['product_id'] == product_id][0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        product_indices = [i[0] for i in sim_scores]
        return df.iloc[product_indices][['product_name', 'description']]
    except Exception as e:
        print(f"Öneri motoru hatası: {e}")
        return pd.DataFrame()


# Uygulama başlatıldığında çalışacak ana fonksiyon
if __name__ == "__main__":
    print("Veri analizi modülü başarıyla yüklendi!")
    print("\nÖRNEK 3: Ürün Öneri Motoru Çalışıyor...\n")
    product_df = create_tech_product_data()
    selected_product_id = 103  # Örneğin Akıllı Telefon
    print(f"Seçilen Ürün: {product_df[product_df['product_id'] == selected_product_id]['product_name'].values[0]}")
    recommendations = recommend_similar_tech_products(product_df, selected_product_id)
    print("\nBenzer Ürün Önerileri:")
    print(recommendations.to_string(index=False))
