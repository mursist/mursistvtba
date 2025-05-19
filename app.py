import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import veri_analizi as va
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Modülleri içe aktar
from modules.dashboard import add_dashboard
from modules.sales_analysis import seasonal_analysis, price_analysis
from modules.customer_analysis import rfm_analysis, sentiment_analysis
from modules.advanced_analytics import profitability_analysis, trend_analysis
from modules.feedback_module import add_feedback_tab, init_db
from modules.database_utils import save_dataframe, read_table, init_database


st.set_page_config(page_title="Yapay Zeka ile Veri Analizi", layout="wide")

st.title("Yapay Zeka ile Veri Analizi")

# Sekmeleri oluşturma - Ana modüller ve yeni modüller eklendi
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Ana Sayfa", 
    "Satış Tahmini", 
    "Müşteri Analizi", 
    "Gelişmiş Analizler",
    "Trendler",
    "Kullanım Kılavuzu",
    "Geri Bildirim"
])

# Ana Sayfa Sekmesi
with tab1:
    st.header("Yapay Zeka ile Veri Analizi Uygulamasına Hoş Geldiniz")
    
    st.info("Bu uygulama, Python'da geliştirilmiş veri analizi ve yapay zeka fonksiyonlarını kullanıcı dostu bir arayüz üzerinden erişilebilir hale getirmek için tasarlanmıştır.")
    
    # Dashboard ekle
    add_dashboard()
    
    # Modülleri görsel kutularda göster
    st.write("### Analiz Modülleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px;">
            <h4 style="color:#1E88E5;">📈 Satış Tahmini</h4>
            <p>ARIMA ve makine öğrenmesi modelleri ile gelecek satışları tahmin edin.</p>
            <p>Mevsimsel analizler ve trend analizleri yapın.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px;">
            <h4 style="color:#43A047;">👥 Müşteri Analizi</h4>
            <p>K-means ile müşteri segmentasyonu yapın.</p>
            <p>RFM analizi ile değerli müşterilerinizi tanımlayın.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px;">
            <h4 style="color:#E53935;">🔍 Gelişmiş Analizler</h4>
            <p>Duygu analizi, sepet analizi ve karlılık analizi gibi gelişmiş analizler yapın.</p>
        </div>
        """, unsafe_allow_html=True)

# Satış Tahmini Sekmesi
with tab2:
    st.header("Zaman Serisi Analizi ve Satış Tahmini")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Temel Tahmin", "Dönemsel Analiz", "Fiyat Analizi"])
    
    with sub_tab1:
        # Mevcut satış tahmini kodu
        sales_file = st.file_uploader("CSV Dosyası Yükleyin (veya örnek veri kullanın)", type="csv")
        if sales_file:
            sales_data = pd.read_csv(sales_file)
            st.session_state['sales_data'] = sales_data

            # 🔽🔽🔽 Veritabanına kaydet (tablo adı: sales_data)
            from modules.database_utils import save_dataframe
            save_dataframe(sales_data, "sales_data", mode='replace')
            st.success("Satış verisi veritabanına kaydedildi.")
        else:
            if st.button("Örnek Veri Oluştur"):
                st.info("Örnek veri oluşturuluyor...")
                sales_data = va.create_sample_sales_data()
                st.success("Örnek veri oluşturuldu!")
                st.session_state['sales_data'] = sales_data
        
        if 'sales_data' in st.session_state:
            sales_data = st.session_state['sales_data']
            st.write("Veri Önizleme:")
            st.dataframe(sales_data.head())
            
            forecast_days = st.slider("Tahmin Günü Sayısı", 7, 90, 30)
            
            if st.button("Analizi Başlat"):
                st.info("Analiz yapılıyor...")
                try:
                    # Zaman serisi analizi
                    with st.spinner("Zaman serisi analizi yapılıyor..."):
                        result = va.analyze_time_series(sales_data)
                    
                    # ARIMA tahmin
                    with st.spinner(f"{forecast_days} günlük tahmin yapılıyor..."):
                        forecast = va.forecast_sales(sales_data, forecast_days)
                    
                    st.success("Analiz tamamlandı!")
                    
                    # Sonuçları göster
                    st.subheader("Zaman Serisi Ayrıştırma")
                    
                    # Gözlemlenen satışlar
                    st.write("#### Gözlemlenen Satışlar")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.observed.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # Trend bileşeni
                    st.write("#### Trend Bileşeni")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.trend.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # Mevsimsel bileşen
                    st.write("#### Mevsimsel Bileşen")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.seasonal.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # Artık bileşen
                    st.write("#### Artık Bileşeni")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.resid.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # ARIMA tahmin sonuçları
                    st.subheader("ARIMA Tahmin Sonuçları")
                    # ARIMA tahmin sonucunu veritabanına kaydet
                    forecast_df = pd.DataFrame({
                    "date": forecast.index,
                    "predicted_sales": forecast.values
                    })

from modules.database_utils import save_dataframe
save_dataframe(forecast_df, "arima_forecast", mode='replace')
st.success("ARIMA tahmin verisi veritabanına kaydedildi.")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Son 90 gün + tahmin
                    ax.plot(sales_data.set_index('date')['sales'][-90:].index, 
                            sales_data.set_index('date')['sales'][-90:].values, 
                            label='Geçmiş Veriler')
                    ax.plot(forecast.index, forecast.values, color='red', label='Tahmin')
                    ax.set_title(f'{forecast_days} Günlük Tahmin')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Machine Learning modeli sonuçları
                    st.subheader("Makine Öğrenmesi Model Sonuçları")
                    with st.spinner("Makine öğrenmesi modelleri eğitiliyor..."):
                        rf_model, xgb_model = va.train_ml_sales_model(sales_data)
                    
                    # Model sonuçlarını göster
                    st.success("Modeller başarıyla eğitildi!")
                    
                except Exception as e:
                    st.error(f"Analiz sırasında bir hata oluştu: {e}")
    
    with sub_tab2:
        # Dönemsel analiz
        seasonal_analysis()
    
    with sub_tab3:
        # Fiyat elastikiyeti analizi
        price_analysis()

# Müşteri Analizi Sekmesi
with tab3:
    st.header("Müşteri Analizi")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Segmentasyon", "RFM Analizi", "Duygu Analizi"])
    
    with sub_tab1:
        # Müşteri segmentasyonu
        customer_file = st.file_uploader("Müşteri CSV Dosyası Yükleyin (veya örnek veri kullanın)", type="csv")
        if customer_file:
            customer_data = pd.read_csv(customer_file)
            save_dataframe(customer_data, "customer_data", mode='replace')
        else:
            if st.button("Örnek Müşteri Verisi Oluştur"):
                st.info("Örnek müşteri verisi oluşturuluyor...")
                customer_data = va.create_customer_data()
                st.success("Örnek müşteri verisi oluşturuldu!")
                st.session_state['customer_data'] = customer_data
        
        if 'customer_data' in st.session_state:
            customer_data = st.session_state['customer_data']
            st.write("Veri Önizleme:")
            st.dataframe(customer_data.head())
            
            cluster_count = st.slider("Küme Sayısı", 2, 8, 4)
            
            if st.button("Segmentasyon Analizini Başlat"):
                st.info("Segmentasyon analizi yapılıyor...")
                try:
                    with st.spinner("Müşteriler segmentlere ayrılıyor..."):
                        segmented_data, kmeans_model, scaler = va.segment_customers(customer_data, cluster_count)
                    
                    st.success("Segmentasyon tamamlandı!")
                    
                    # Sonuçları göster
                    st.subheader("Segmentasyon Sonuçları")
                    
                    # Küme görselleştirme
                    st.write("#### Küme Görselleştirmesi")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(customer_data['avg_purchase_value'], 
                                        customer_data['purchase_frequency'],
                                        c=segmented_data['cluster'], 
                                        cmap='viridis', 
                                        alpha=0.6)
                    ax.set_xlabel('Ortalama Satın Alma Değeri')
                    ax.set_ylabel('Satın Alma Sıklığı')
                    ax.set_title('Müşteri Segmentasyonu')
                    legend1 = ax.legend(*scatter.legend_elements(),
                                      title="Kümeler")
                    ax.add_artist(legend1)
                    st.pyplot(fig)
                    
                    # Küme istatistikleri
                    st.write("#### Küme İstatistikleri")
                    cluster_stats = segmented_data.groupby('cluster').agg({
                        'customer_id': 'count',
                        'avg_purchase_value': 'mean',
                        'purchase_frequency': 'mean',
                        'return_rate': 'mean',
                        'customer_value': 'mean'
                    }).reset_index()
                    
                    cluster_stats.columns = ['Küme', 'Müşteri Sayısı', 'Ort. Satın Alma', 'Satın Alma Sıklığı', 'İade Oranı', 'Müşteri Değeri']
                    st.dataframe(cluster_stats)
                    
                except Exception as e:
                    st.error(f"Segmentasyon sırasında bir hata oluştu: {e}")
    
    with sub_tab2:
        # RFM analizi
        rfm_analysis()
    
    with sub_tab3:
        # Duygu analizi
        sentiment_analysis()

# Gelişmiş Analizler Sekmesi
with tab6:
    st.header("Gelişmiş Analizler")
    
    sub_tab1, sub_tab2 = st.tabs(["Karlılık Analizi", "Ürün Öneri Motoru"])
    
    with sub_tab1:
        # Karlılık analizi
        profitability_analysis()
    
    with sub_tab2:
        # Öneri motoru
        st.write("Ürün öneri motoru yakında eklenecek...")

# Trendler Sekmesi
with tab5:
    # Trend analizi
    trend_analysis()

# Kullanım Kılavuzu sekmesi
with tab4:
    st.header("Kullanım Kılavuzu ve Teknik Detaylar")
    
    # Genel Bakış
    st.subheader("1. Genel Bakış")
    st.write("Bu uygulama, veri analizi ve yapay zeka yöntemlerini kullanarak satış tahmini, müşteri segmentasyonu ve anomali tespiti yapmanızı sağlayan etkileşimli bir araçtır.")
    
    with st.expander("Uygulamanın Amacı", expanded=False):
        st.write("""
        Bu uygulama, karmaşık veri analizi ve yapay zeka işlemlerini kod yazmadan gerçekleştirmenize olanak tanır. Başlıca kullanım alanları:
        
        - **İş Analitiği**: Satış tahminleri yaparak envanter yönetimi ve finansal planlamayı optimize edin
        - **Pazarlama Stratejisi**: Müşterileri segmentlere ayırarak hedefli pazarlama kampanyaları geliştirin
        - **Risk Yönetimi**: Anormal müşteri davranışlarını tespit ederek potansiyel sahtekarlık veya kayıpları önleyin
        - **Karar Destek**: Veri odaklı iş kararları vermek için güvenilir analizler elde edin
        
        Bu araçla, veri bilimcilerin ve analistlerin günlük olarak kullandığı gelişmiş tekniklere kolayca erişebilirsiniz.
        """)
    
    st.markdown("""
    Bu uygulamada üç ana analiz modülü bulunmaktadır:
    - **Zaman Serisi Analizi ve Satış Tahmini**: Geçmiş verileri analiz ederek gelecek satışlarını tahmin eder
    - **Müşteri Segmentasyonu**: Benzer davranış gösteren müşterileri gruplandırır
    - **Anomali Tespiti**: Normal müşteri davranışından sapan anormal desenleri tespit eder
    """)
    
    # Zaman Serisi Analizi 
    st.subheader("2. Zaman Serisi Analizi ve Satış Tahmini")
    st.write("Bu modül, geçmiş satış verilerini analiz ederek gelecekteki satışları tahmin etmek için kullanılır.")
    
    # Veri Formatı
    st.write("#### 2.1. Veri Formatı")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        CSV dosyanızın aşağıdaki **zorunlu sütunları** içermesi gerekir:
        - `date`: YYYY-MM-DD formatında tarih (ör. 2022-01-01)
        - `sales`: Sayısal satış değeri
        """)
    
    with col2:
        st.markdown("""
        İsteğe bağlı olarak aşağıdaki sütunları da ekleyebilirsiniz:
        - `is_holiday`: Tatil günü olup olmadığını belirten 1/0 değeri
        - `is_promotion`: Promosyon dönemi olup olmadığını belirten 1/0 değeri
        - `weekday`: Haftanın günü (0-6, 0=Pazartesi)
        - `month`: Ay (1-12)
        - `year`: Yıl
        - `is_weekend`: Hafta sonu olup olmadığını belirten 1/0 değeri
        """)
    
    # CSV örneği
    with st.expander("CSV Dosya Örneği", expanded=False):
        st.code("""date,sales,is_holiday,is_promotion,weekday,month,year
2022-01-01,350,1,0,5,1,2022
2022-01-02,280,0,0,6,1,2022
2022-01-03,320,0,0,0,1,2022
2022-01-04,310,0,0,1,1,2022
2022-01-05,340,0,0,2,1,2022""", language="csv")
        
        st.info("Kendi CSV dosyanızı bu formatta hazırlayabilir veya 'Örnek Veri Oluştur' butonunu kullanarak örnek veri oluşturabilirsiniz.")
    
    # Hesaplama Adımları
    st.write("#### 2.2. Hesaplama Adımları")
    
    st.write("##### 2.2.1. Zaman Serisi Ayrıştırma (Seasonal Decomposition)")
    st.markdown("""
    Zaman serisi ayrıştırma, satış verilerinin içindeki farklı bileşenleri ayrıştırmak için kullanılır:
    
    1. **Gözlemlenen Satışlar**: Orijinal zaman serisi verisi
    2. **Trend Bileşeni**: Uzun vadeli artış veya azalış trendi
       - Hareketli ortalama (moving average) yöntemi ile hesaplanır
       - Formül: Belirli bir periyot boyunca verilerin ortalaması alınır
    3. **Mevsimsel Bileşen**: Tekrarlanan, periyodik dalgalanmalar
       - Trendsiz verilerin mevsimsel periyotlarına göre ortalaması alınarak hesaplanır
       - Günlük, haftalık, aylık ve yıllık desenler içerebilir
    4. **Artık (Residual) Bileşen**: Trend ve mevsimsellikle açıklanamayan değişimler
       - Formül: Gözlemlenen Veri - (Trend + Mevsimsellik)
    
    Bu ayrıştırma için statsmodels kütüphanesinin `seasonal_decompose` fonksiyonunu kullanıyoruz.
    """)
    
    # İlgili görseli göster
    sample_decompose = Image.open("https://via.placeholder.com/800x400?text=Seasonal+Decomposition+Example") if 'Image' in globals() else None
    if sample_decompose:
        st.image(sample_decompose, caption="Zaman Serisi Ayrıştırma Örneği", use_column_width=True)
    
    st.write("##### 2.2.2. ARIMA Modeli ile Satış Tahmini")
    st.markdown("""
    ARIMA (AutoRegressive Integrated Moving Average) modelini kullanarak gelecek satışlarını tahmin ediyoruz:
    
    1. **Otoregresif Bileşen (AR - p)**: 
       - Geçmiş değerler kullanılarak gelecek değerlerin tahmini
       - Formül: Yt = c + φ1*Y(t-1) + φ2*Y(t-2) + ... + φp*Y(t-p) + εt
       - Modelimizde p=5 kullanılıyor (5 gecikmeli değer)
    
    2. **Entegrasyon Derecesi (I - d)**:
       - Zaman serisini durağanlaştırmak için kullanılan fark alma işlemi
       - Modelimizde d=1 kullanılıyor (birinci dereceden fark alma)
    
    3. **Hareketli Ortalama Bileşeni (MA - q)**:
       - Geçmiş hata terimlerini kullanarak gelecek değerleri tahmin etme
       - Formül: Yt = c + εt + θ1*ε(t-1) + θ2*ε(t-2) + ... + θq*ε(t-q)
       - Modelimizde q=2 kullanılıyor (2 gecikmeli hata terimi)
    
    4. **Tahmin ve Güven Aralığı**:
       - Model ile gelecek için nokta tahminleri yapılır
       - %95 güven aralığı ile tahmin belirsizliği gösterilir
    
    Bu tahmin için statsmodels kütüphanesinin `ARIMA` modelini kullanıyoruz.
    """)
    
    with st.expander("ARIMA Parametreleri Hakkında Detaylı Bilgi", expanded=False):
        st.markdown("""
        **ARIMA Parametrelerinin Seçimi**
        
        ARIMA modeli üç parametreye sahiptir: p, d ve q. Bu parametrelerin seçimi önemlidir ve verilerin yapısına bağlıdır.
        
        - **p (AR terimi)**: Otoregresif terim, bir gözlemin geçmiş gözlemlere bağlılığını belirtir. Yüksek p değerleri, daha uzun geçmiş bağımlılıkları yakalar ancak aşırı uyum (overfitting) riski taşır.
        
        - **d (Fark alma)**: Serinin durağanlaştırılması için gereken fark alma sayısı. d=1, her gözlemden bir önceki gözlemi çıkarır. d=2, birinci farkların farkını alır.
        
        - **q (MA terimi)**: Hareketli ortalama terimi, bir gözlemin geçmiş hata terimleriyle ilişkisini gösterir.
        
        Parametrelerin optimal değerleri genellikle şu yöntemlerle belirlenir:
        
        1. **ACF ve PACF Grafikleri**: Otokorelasyon ve kısmi otokorelasyon fonksiyonları
        2. **Bilgi Kriterleri**: AIC (Akaike Information Criterion) veya BIC (Bayesian Information Criterion)
        3. **Grid Search**: Farklı p, d, q kombinasyonlarını deneyerek en iyi performansı veren kombinasyonu bulma
        
        Bizim varsayılan modelimiz ARIMA(5,1,2) olarak seçilmiştir, ancak verileriniz için farklı parametreler daha iyi sonuç verebilir.
        """)
    
    st.write("##### 2.2.3. Makine Öğrenmesi Modelleri ile Satış Tahmini")
    st.markdown("""
    İki farklı makine öğrenmesi algoritması kullanarak alternatif tahminler yapıyoruz:
    
    1. **RandomForest Regressor**:
       - Çok sayıda karar ağacının ortalamasını alarak çalışır
       - Aşırı öğrenmeye (overfitting) karşı dirençlidir
       - Parametreler:
         - n_estimators=100 (100 farklı ağaç)
         - random_state=42 (tekrarlanabilirlik için)
    
    2. **XGBoost Regressor**:
       - Gradient boosting tekniğini kullanır
       - Her adımda bir önceki modelin hatalarını düzeltmeye çalışır
       - Parametreler:
         - n_estimators=100 (100 iterasyon)
         - learning_rate=0.1 (öğrenme hızı)
         - max_depth=7 (ağaç derinliği)
    
    3. **Özellik Önemliliği**:
       - Hangi faktörlerin satışları en çok etkilediğini gösterir
       - RandomForest modelinin feature_importances_ özelliği kullanılır
    
    4. **Çapraz Doğrulama**:
       - Zaman serisi verilerinde özel bir çapraz doğrulama olan TimeSeriesSplit kullanılır
       - Model performansını değerlendirmek için RMSE (Root Mean Squared Error) kullanılır
    """)
    
    # Adım Adım Kullanım
    st.write("#### 2.3. Adım Adım Kullanım")
    
    st.markdown("""
    1. **Veri Hazırlama**:
       - CSV dosyanızı uygun formata getirin veya "Örnek Veri Oluştur" butonunu kullanın
    
    2. **Veri Yükleme**:
       - "CSV Dosyası Yükleyin" bölümünden dosyanızı seçin veya 
       - "Örnek Veri Oluştur" butonuna tıklayın
    
    3. **Parametre Ayarlama**:
       - "Tahmin Günü Sayısı" ayarını değiştirerek kaç gün ilerisini tahmin etmek istediğinizi belirleyin
    
    4. **Analizi Başlatma**:
       - "Analizi Başlat" butonuna tıklayın ve işlemin tamamlanmasını bekleyin
    
    5. **Sonuçları İnceleme**:
       - Zaman Serisi Ayrıştırma: Veriyi bileşenlerine ayırır
       - ARIMA Tahmin Sonuçları: Gelecek satışları tahmin eder
       
    6. **Grafikleri Büyütme**:
       - Her grafiğin yanındaki "Büyüt" butonuna tıklayarak detaylı görünümü açabilirsiniz
       - Detaylı görünümü kapatmak için "Kapat" butonuna tıklayın
    """)
    
    # Yorumlama Rehberi
    with st.expander("Sonuçları Yorumlama Rehberi", expanded=False):
        st.markdown("""
        **Zaman Serisi Analizi Sonuçlarını Yorumlama**
        
        1. **Trend Bileşeni**:
           - Yukarı yönlü bir trend, büyüyen bir pazarı veya artan talebi gösterebilir
           - Aşağı yönlü bir trend, azalan ilgiyi veya pazar daralmasını gösterebilir
           - Düz bir trend, olgun ve stabil bir pazarı gösterebilir
        
        2. **Mevsimsel Bileşen**:
           - Güçlü mevsimsellik, belirli dönemlerde tekrarlanan desenler olduğunu gösterir
           - Mevsimsellik desenini anlamak, envanter planlaması ve pazarlama zamanlaması için önemlidir
        
        3. **Artık Bileşen**:
           - Büyük artıklar, açıklanamayan değişkenliği gösterir ve tahminleri zorlaştırabilir
           - Ardışık artıklar arasında ilişki olmaması iyidir (white noise)
        
        **ARIMA Tahmin Sonuçlarını Yorumlama**
        
        1. **Tahmin Eğrisi**:
           - Tahminin genel yönü ve şekli, gelecekteki beklenen eğilimleri gösterir
        
        2. **Güven Aralığı**:
           - Geniş güven aralıkları, yüksek belirsizlik gösterir
           - Dar güven aralıkları, daha güvenilir tahminler anlamına gelir
        
        3. **Tahmin Doğruluğu**:
           - RMSE (Root Mean Squared Error): Daha düşük değerler daha iyi tahmin demektir
           - MAE (Mean Absolute Error): Daha düşük değerler daha iyi tahmin demektir
        """)
    
    # İpuçları ve En İyi Uygulamalar
    with st.expander("İpuçları ve En İyi Uygulamalar", expanded=False):
        st.markdown("""
        **Satış Tahmini için İpuçları**
        
        1. **Veri Kalitesi**:
           - Tutarlı ve düzenli aralıklarla toplanmış veri kullanın
           - Eksik değerleri doldurun veya ilgili satırları kaldırın
           - Aykırı değerleri tespit edin ve gerekirse düzeltin
        
        2. **Zaman Aralığı**:
           - En az 1-2 yıllık veri kullanın (mevsimselliği yakalamak için)
           - Çok eski veriler güncel trendleri yansıtmayabilir, dikkatli kullanın
        
        3. **Ek Faktörler**:
           - Tatil günleri ve promosyonlar gibi özel faktörleri modelinize dahil edin
           - Ekonomik göstergeler veya sektör trendleri gibi dış faktörleri düşünün
        
        4. **Model Değerlendirmesi**:
           - Modeli geçmiş verilerde test edin (örn. son 30 günü tahmin edin ve gerçek değerlerle karşılaştırın)
           - Birden fazla modeli karşılaştırın (ARIMA, XGBoost, vb.)
        
        5. **Sürekli İyileştirme**:
           - Tahminleri düzenli olarak güncelle ve yeni verilerle modeli eğitin
           - Tahmin hatalarından öğrenin ve modeli iyileştirin
        """)
    
    # Müşteri Segmentasyonu
    st.subheader("3. Müşteri Segmentasyonu")
    st.write("Bu modül, müşterilerinizi benzer davranış özelliklerine göre gruplara ayırmak için kullanılır.")
    
    # Veri Formatı
    st.write("#### 3.1. Veri Formatı")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        CSV dosyanızın aşağıdaki **zorunlu sütunları** içermesi gerekir:
        - `customer_id`: Müşteri kimliği (ör. CUST_00001)
        - `avg_purchase_value`: Ortalama satın alma değeri (ör. 5000)
        - `purchase_frequency`: Satın alma sıklığı (ör. 12 - yıllık satın alma sayısı)
        - `return_rate`: İade oranı (0-1 arası, ör. 0.05 = %5)
        """)
    
    with col2:
        st.markdown("""
        İsteğe bağlı olarak aşağıdaki sütunları da ekleyebilirsiniz:
        - `loyalty_years`: Müşteri sadakat yılı
        - `avg_basket_size`: Ortalama sepet büyüklüğü (ürün sayısı)
        - `pct_discount_used`: İndirim kullanım oranı (0-1 arası)
        - `customer_value`: Müşteri yaşam boyu değeri
        """)
    
    # CSV örneği
    with st.expander("CSV Dosya Örneği", expanded=False):
        st.code("""customer_id,avg_purchase_value,purchase_frequency,return_rate,loyalty_years
CUST_00001,2500,12,0.05,3.5
CUST_00002,1800,3,0.12,1.2
CUST_00003,5400,6,0.02,4.7
CUST_00004,980,24,0.08,2.1
CUST_00005,12500,2,0.01,5.3""", language="csv")
        
        st.info("Kendi CSV dosyanızı bu formatta hazırlayabilir veya 'Örnek Müşteri Verisi Oluştur' butonunu kullanarak örnek veri oluşturabilirsiniz.")
    
    # Hesaplama Adımları
    st.write("#### 3.2. Hesaplama Adımları")
    
    st.write("##### 3.2.1. Veri Ön İşleme")
    st.markdown("""
    Segmentasyon öncesi veri hazırlığı:
    
    1. **Veri Normalizasyonu**:
       - Farklı ölçeklerdeki özellikleri 0-1 arasına getirme
       - StandardScaler kullanılır: z = (x - μ) / σ
       - Burada x: orijinal değer, μ: ortalama, σ: standart sapma
    
    2. **Özellik Seçimi**:
       - Segmentasyon için en bilgilendirici özellikler seçilir
       - Kullanılan özellikler: 'avg_purchase_value', 'purchase_frequency', 'return_rate', 'loyalty_years', 'customer_value'
    """)
    
    st.write("##### 3.2.2. K-means Kümeleme")
    st.markdown("""
    K-means algoritması ile müşteri segmentasyonu:
    
    1. **Optimal Küme Sayısı Belirleme**:
       - Silhouette skoru kullanılır: -1 (kötü) ile 1 (mükemmel) arası bir değer
       - Formül: s(i) = (b(i) - a(i)) / max{a(i), b(i)}
         - a(i): Bir noktanın kendi kümesindeki diğer noktalara olan ortalama mesafesi
         - b(i): Bir noktanın en yakın komşu kümedeki noktalara olan ortalama mesafesi
       - 2'den 8'e kadar her küme sayısı için hesaplanır ve en yüksek skora sahip küme sayısı seçilir
    
    2. **K-means Algoritması**:
       - 1) Rastgele k adet merkez nokta seçilir (başlangıç noktaları)
       - 2) Her veri noktası en yakın merkeze atanır
       - 3) Her küme için yeni merkez hesaplanır (kümedeki noktaların ortalaması)
       - 4) Merkezler değişmeyene kadar adım 2 ve 3 tekrarlanır
       - Uzaklık ölçümü için Öklid mesafesi kullanılır: d(x,y) = √Σ(xi-yi)²
    
    3. **Küme Analizi**:
       - Her kümenin merkezi özelliklerini belirlemek
       - Her kümede kaç müşteri olduğunu hesaplamak
       - Kümeleri görselleştirmek (2B ve 3B grafikler)
    """)
    
    # Adım Adım Kullanım
    st.write("#### 3.3. Adım Adım Kullanım")
    
    st.markdown("""
    1. **Veri Hazırlama**:
       - CSV dosyanızı uygun formata getirin veya "Örnek Müşteri Verisi Oluştur" butonunu kullanın
    
    2. **Veri Yükleme**:
       - "Müşteri CSV Dosyası Yükleyin" bölümünden dosyanızı seçin veya 
       - "Örnek Müşteri Verisi Oluştur" butonuna tıklayın
    
    3. **Parametre Ayarlama**:
       - "Küme Sayısı" ayarını değiştirerek müşterilerinizi kaç segmente ayırmak istediğinizi belirleyin
    
    4. **Analizi Başlatma**:
       - "Segmentasyon Analizi Başlat" butonuna tıklayın ve işlemin tamamlanmasını bekleyin
    
    5. **Sonuçları İnceleme**:
       - Müşteri Segmentasyonu (2B): Müşterilerin segmentlere göre dağılımını gösterir
       - Küme İstatistikleri: Her segmentin özelliklerini gösterir (ortalama değerler)
       
    6. **Grafikleri Büyütme**:
       - Her grafiğin yanındaki "Büyüt" butonuna tıklayarak detaylı görünümü açabilirsiniz
       - Detaylı görünümü kapatmak için "Kapat" butonuna tıklayın
    """)
    
    # Yorumlama Rehberi
    with st.expander("Segmentasyon Sonuçlarını Yorumlama", expanded=False):
        st.markdown("""
        **Müşteri Segmentlerini Anlamak**
        
        K-means algoritması müşterilerinizi benzer davranış özelliklerine sahip gruplara ayırır. Bu segmentleri şu şekilde yorumlayabilirsiniz:
        
        1. **Yüksek Değerli Müşteriler**:
           - Yüksek ortalama satın alma değeri
           - Orta-yüksek satın alma sıklığı
           - Düşük iade oranı
           - Yüksek sadakat
           - **Strateji**: Özel VIP programları, kişiselleştirilmiş hizmetler
        
        2. **Sık Alışveriş Yapan Müşteriler**:
           - Düşük-orta ortalama satın alma değeri
           - Yüksek satın alma sıklığı
           - Düşük iade oranı
           - **Strateji**: Sadakat programları, çapraz satış teklifleri
        
        3. **Büyük Alışveriş Yapan Nadir Müşteriler**:
           - Çok yüksek ortalama satın alma değeri
           - Düşük satın alma sıklığı
           - Çok düşük iade oranı
           - **Strateji**: Düzenli hatırlatmalar, özel teklifler
        
        4. **Risk Altındaki Müşteriler**:
           - Düşük ortalama satın alma değeri
           - Düşük satın alma sıklığı
           - Yüksek iade oranı
           - **Strateji**: Geri kazanma kampanyaları, müşteri memnuniyeti anketleri
        
        5. **Ortalama Müşteriler**:
           - Orta seviye ortalama satın alma değeri
           - Orta seviye satın alma sıklığı
           - Orta seviye iade oranı
           - **Strateji**: Genel pazarlama kampanyaları, ürün önerileri
        
        Bu segment tanımları genel bir çerçeve sunar, ancak sizin verilerinize göre farklı segment özellikleri ortaya çıkabilir.
        """)
    
    # Anomali Tespiti
    st.subheader("4. Anomali Tespiti")
    st.write("Bu modül, normal müşteri davranışından sapan anormal desenleri tespit etmek için kullanılır.")
    
    # Hesaplama Adımları
    st.write("#### 4.1. Hesaplama Adımları")
    
    st.write("##### 4.1.1. Isolation Forest Algoritması")
    st.markdown("""
    Isolation Forest, anormallikleri diğer noktalardan "izole etme" kolaylığına göre tespit eder:
    
    1. **Çalışma Prensibi**:
       - Normal noktaları izole etmek daha fazla bölme gerektirir (daha derin ağaç)
       - Anormal noktalar daha az bölme ile izole edilebilir (daha sığ ağaç)
       - Temel varsayım: Anormal noktalar, daha az sayıda ve normal noktalardan daha uzaktadır
    
    2. **Algoritma Adımları**:
       - 1) Veri kümesinden rastgele bir alt küme alınır
       - 2) Rastgele bir özellik seçilir
       - 3) Seçilen özellik için rastgele bir bölme değeri belirlenir
       - 4) Veri iki alt kümeye bölünür
       - 5) İzolasyon tamamlanana veya maksimum ağaç derinliğine ulaşılana kadar tekrarlanır
       - 6) Çoklu ağaçlar oluşturularak ortalama yol uzunluğu hesaplanır
    
    3. **Anomali Skoru Hesaplama**:
       - s(x,n) = 2^(-E(h(x))/c(n))
       - E(h(x)): Noktanın ortalama yol uzunluğu
       - c(n): Normal dağılımlı veride ortalama yol uzunluğu
       - Skor 0'a yaklaştıkça daha anormal, 0.5'e yaklaştıkça daha normal
    
    4. **Contamination (Kirlilik) Parametresi**:
       - Verinin ne kadarının anormal olarak işaretleneceğini belirler
       - Uygulamada bu değer, "Anomali Eşiği (%)" ayarı ile belirlenir
    """)
    
    with st.expander("Isolation Forest Algoritması Hakkında Detaylı Bilgi", expanded=False):
        st.markdown("""
        **Isolation Forest Algoritması Detayları**
        
        Isolation Forest, geleneksel anomali tespit yöntemlerinden farklı olarak, anormal noktaları izole etme kolaylığına göre tanımlar. Bu yaklaşımın avantajları:
        
        - **Verimlilik**: Lineer zaman karmaşıklığı ve düşük bellek kullanımı
        - **Etkililik**: Yüksek boyutlu verilerde bile iyi performans gösterir
        - **Scalabilirlik**: Herhangi bir dağılım veya yoğunluğu öğrenmeye gerek duymaz
        
        Isolation Forest algoritması, anormal noktalar için daha kısa ortalama yol uzunlukları ürettiğinden, bu özelliği anormallikleri tespit etmek için kullanır.
        
        **Hiperparametreler**:
        
        - **n_estimators**: Oluşturulacak ağaç sayısı. Daha fazla ağaç, daha kararlı sonuçlar sağlar ancak hesaplama maliyetini artırır.
        - **max_samples**: Her ağaç için kullanılacak örnek sayısı. Tam veri setini kullanmak yerine alt örnekleme yaparak verimliliği artırır.
        - **contamination**: Verideki tahmini anormallik oranı. Bu değer, anormallik eşiğini belirlemek için kullanılır.
        - **max_features**: Her ağaç için kullanılacak özellik sayısı. Varsayılan olarak tüm özellikler kullanılır.
        - **bootstrap**: Alt örneklemenin iade ile mi (True) yoksa iadesiz mi (False) yapılacağını belirler.
        
        **Kısıtlamalar**:
        
        - Kategorik değişkenlerle doğrudan çalışamaz, önceden dönüştürülmeleri gerekir.
        - Çok düşük kirlilik oranlarında performans düşebilir.
        - Veri setinin boyutu çok küçükse güvenilirliği azalabilir.
        """)
    
    st.write("##### 4.1.2. Local Outlier Factor (LOF) Algoritması")
    st.markdown("""
    LOF, bir noktanın yerel yoğunluğunu komşularının yerel yoğunluğu ile karşılaştırır:
    
    1. **Çalışma Prensibi**:
       - Bir noktanın yerel yoğunluğu, komşularının yerel yoğunluğundan önemli ölçüde düşükse, bu nokta bir anormallik olarak kabul edilir
       - Temel varsayım: Normal noktalar benzer yoğunluktaki bölgelerde kümelenirken, anormal noktalar daha düşük yoğunluklu bölgelerde bulunur
    
    2. **Algoritma Adımları**:
       - 1) Her nokta için k en yakın komşu (k-NN) bulunur
       - 2) Her nokta için erişilebilirlik mesafesi hesaplanır
       - 3) Her nokta için yerel erişilebilirlik yoğunluğu (LRD) hesaplanır
       - 4) Her nokta için LOF skoru hesaplanır
    
    3. **LOF Skoru Hesaplama**:
       - LOF(A) = (Σ LRD(N) / LRD(A)) / k
       - N: A noktasının k en yakın komşusu
       - LRD: Yerel erişilebilirlik yoğunluğu
       - LOF değeri 1'e yakınsa normal, 1'den büyükse anormal
    
    4. **Contamination (Kirlilik) Parametresi**:
       - Verinin ne kadarının anormal olarak işaretleneceğini belirler
       - Uygulamada bu değer, "Anomali Eşiği (%)" ayarı ile belirlenir
    """)
    
    st.write("##### 4.1.3. One-Class SVM Algoritması")
    st.markdown("""
    One-Class SVM, normal veriyi kapsayan bir sınır çizmeye çalışır:
    
    1. **Çalışma Prensibi**:
       - Veri noktalarını içeren minimum hacimli bir hiper-küre veya hiper-düzlem bulunur
       - Bu sınırın dışında kalan noktalar anormal kabul edilir
       - Temel varsayım: Normal veriler belirli bir bölgede yoğunlaşır
    
    2. **Algoritma Detayları**:
       - Bir çekirdek fonksiyonu (genellikle RBF) kullanarak veriyi daha yüksek boyutlu bir uzaya taşır
       - Bu uzayda, verileri orijinden ayıran bir hiper-düzlem bulunur
       - Hiper-düzlemden uzaklık, anormallik ölçüsü olarak kullanılır
    
    3. **Parametreler**:
       - **nu**: Eğitim hatasının üst sınırı ve destek vektörlerinin alt sınırı (0,1]
       - **kernel**: Çekirdek fonksiyonu (rbf, linear, poly)
       - **gamma**: RBF çekirdeğinin genişlik parametresi
    """)
    
    # Adım Adım Kullanım
    st.write("#### 4.2. Adım Adım Kullanım")
    
    st.markdown("""
    1. **Veri Yükleme**:
       - Müşteri segmentasyonu için kullandığınız aynı veri seti anomali tespiti için de kullanılır
       - Eğer henüz yüklemediyseniz, "Müşteri CSV Dosyası Yükleyin" bölümünden dosyanızı seçin veya 
       - "Örnek Müşteri Verisi Oluştur" butonuna tıklayın
    
    2. **Parametre Ayarlama**:
       - "Anomali Eşiği (%)" ayarını değiştirerek verinizin yüzde kaçının anormal olarak işaretlenmesini istediğinizi belirleyin
       - "Algoritma Seçimi" ile hangi anomali tespit algoritmasını kullanmak istediğinizi seçin
    
    3. **Analizi Başlatma**:
       - "Anomali Tespiti Başlat" butonuna tıklayın ve işlemin tamamlanmasını bekleyin
    
    4. **Sonuçları İnceleme**:
       - Anomali Görseli: Tespit edilen anormal müşterileri gösterir
       - Anormal Müşteriler Tablosu: Anormal olarak işaretlenen müşterilerin detaylarını listeler
       
    5. **Grafikleri Büyütme**:
       - Her grafiğin yanındaki "Büyüt" butonuna tıklayarak detaylı görünümü açabilirsiniz
       - Detaylı görünümü kapatmak için "Kapat" butonuna tıklayın
    """)
    
    # Yorumlama Rehberi
    with st.expander("Anomali Tespiti Sonuçlarını Yorumlama", expanded=False):
        st.markdown("""
        **Anomali Sonuçlarını Anlamak**
        
        Anomali tespit algoritmaları normal müşteri davranışından sapan desenleri tespit eder. Bu sonuçları şu şekilde yorumlayabilirsiniz:
        
        1. **Potansiyel Sahtecilik**:
           - Normalden çok daha yüksek satın alma değerleri + düşük sadakat
           - Normalden çok daha düşük iade oranları + yüksek satın alma sıklığı
           - **Eylem**: Bu hesapları detaylı inceleme, güvenlik kontrolleri
        
        2. **Tatminsiz Müşteriler**:
           - Düşük satın alma değeri + çok yüksek iade oranı
           - Düşük satın alma sıklığı + uzun sadakat süresi
           - **Eylem**: Müşteri hizmetleri takibi, anket veya geri bildirim isteme
        
        3. **Davranış Değişikliği**:
           - Geçmiş alışveriş davranışından ani sapmalar
           - Alışılmadık sepet bileşimleri veya ürün kategorileri
           - **Eylem**: Kişiselleştirilmiş teklifler, davranış değişikliğinin nedenini anlama
        
        4. **Sistem Hataları**:
           - Gerçekçi olmayan değerler (ör. negatif satın alma değeri)
           - Olanaksız kombinasyonlar (ör. %100'den fazla iade oranı)
           - **Eylem**: Veri kalitesi kontrolleri, sistem güvenilirliği iyileştirmeleri
        
        5. **Nadir Segmentler**:
           - Çok nadir bir müşteri profiline sahip geçerli müşteriler
           - Az sayıda kişinin gösterdiği alışılmadık ancak değerli davranışlar
           - **Eylem**: Özel müşteri deneyimi, yeni pazar fırsatlarının keşfi
        
        Anomali tespiti, doğrudan "iyi" veya "kötü" müşterileri belirlemez, sadece "farklı" olanları belirler. Bu nedenle, tespit edilen her anormalliğin bağlamda değerlendirilmesi önemlidir.
        """)
    
    st.markdown("""
    ### 5. Teknik Detaylar
    
    Bu uygulama, aşağıdaki Python kütüphaneleri ve teknolojileri kullanılarak geliştirilmiştir:
    
    - **Streamlit**: Web arayüzü ve etkileşimli bileşenler
    - **Pandas**: Veri manipülasyonu ve analizi
    - **NumPy**: Sayısal hesaplamalar
    - **Matplotlib ve Plotly**: Veri görselleştirme
    - **Scikit-learn**: Makine öğrenmesi algoritmaları (K-means, Isolation Forest, LOF, One-Class SVM)
    - **Statsmodels**: Zaman serisi analizi ve ARIMA modelleri
    - **XGBoost**: Gradyan artırma modelleri
    
    Uygulama, orta büyüklükte veri setleri (10.000-100.000 satır) için optimize edilmiştir. Çok büyük veri setleri için işlem süresi uzayabilir.
    
    Tüm hesaplamalar tarayıcınızda ve sunucu tarafında gerçekleştirilir; verileriniz üçüncü taraf hizmetlerle paylaşılmaz.
    """)
    
    # FAQ
    st.subheader("6. Sıkça Sorulan Sorular (SSS)")
    
    faq_items = [
        ("Hangi veri formatları destekleniyor?", 
         "Şu anda sadece CSV formatı desteklenmektedir. Excel (XLSX) dosyalarınızı önce CSV formatına dönüştürmeniz gerekir."),
        
        ("Verilerim güvende mi?", 
         "Evet, verileriniz sadece analiz için kullanılır ve saklanmaz. Tüm işlemler tarayıcınızda ve sunucu tarafında gerçekleştirilir."),
        
        ("Ne kadar veri yükleyebilirim?", 
         "Teknik olarak 200 MB'a kadar dosya yükleyebilirsiniz, ancak optimal performans için 50 MB'dan küçük dosyaların kullanılması önerilir."),
        
        ("Analiz sonuçlarını indirebilir miyim?", 
         "Evet, her grafik ve tablo için 'İndir' butonu bulunmaktadır. Ayrıca, tüm analiz sonuçlarını tek bir PDF raporu olarak da indirebilirsiniz."),
        
        ("Özelleştirilmiş analizler yaptırabilir miyim?", 
         "Evet, özel analiz ihtiyaçlarınız için 'İletişim' sekmesindeki formu doldurarak bizimle iletişime geçebilirsiniz."),
        
        ("Zaman serisi analizi için minimum ne kadar veri gerekiyor?", 
         "İyi sonuçlar için en az 30 veri noktası (30 gün) öneriyoruz, ancak mevsimselliği yakalamak için en az 1 yıllık veri ideal olacaktır."),
        
        ("Müşteri segmentasyonu için minimum kaç müşteri verisi gerekiyor?", 
         "İstatistiksel olarak anlamlı sonuçlar için en az 50 müşteri verisi öneriyoruz, ancak 200+ müşteri verisi daha güvenilir segmentler oluşturmanızı sağlayacaktır."),
        
        ("Eksik verileri nasıl işlemeliyim?", 
         "Uygulama, eksik verileri temel yöntemlerle işleyebilir, ancak en iyi sonuçlar için eksik verileri doldurmak veya ilgili satırları kaldırmak önerilir."),
        
        ("Hangi tarayıcılar destekleniyor?", 
         "Chrome, Firefox, Safari ve Edge'in güncel sürümleri tam olarak desteklenmektedir. Internet Explorer desteklenmemektedir."),
        
        ("Analiz sonuçlarını API üzerinden alabilir miyim?", 
         "Şu anda API hizmeti sunmuyoruz, ancak kurumsal müşterilerimiz için özel API çözümleri geliştirilebilir. Detaylar için iletişime geçiniz.")
    ]
    
    for question, answer in faq_items:
        with st.expander(question, expanded=False):
            st.write(answer)
    
   
    
    # İletişim Bilgileri
    st.subheader("7. İletişim ve Destek")
    
    st.markdown("""
    Bu uygulama hakkında sorularınız veya geri bildirimleriniz için aşağıdaki kanallardan bizimle iletişime geçebilirsiniz:
    
    - **E-posta**: mursist@gmail.com

    """)
    
    # Lisanslama Bilgileri
    with st.expander("Lisanslama Bilgileri", expanded=False):
        st.markdown("""
        **Mursist Yapay Zeka ile Veri Analiz Uygulaması**
        
        © 2025 Mursist Tüm hakları saklıdır.
        
        Bu yazılım lisanslı bir üründür ve şu lisans koşulları altında dağıtılmaktadır:
        
        1. **Kullanım Hakkı**: Bu yazılımı, lisans süreniz boyunca kurumunuz içinde sınırsız sayıda kullanıcı için kullanabilirsiniz.
        
        2. **Sınırlamalar**: Yazılımı kiralayamaz, ödünç veremez, yeniden dağıtamaz veya alt lisanslayamazsınız. Yazılımın kaynak kodunu tersine mühendislik yöntemleriyle çıkaramaz veya erişmeye çalışamazsınız.
        
        3. **Mülkiyet**: Yazılım ve tüm fikri mülkiyet hakları Veri Analitik Teknolojileri A.Ş.'ye aittir.
        
        4. **Garanti Reddi**: Yazılım "olduğu gibi" sağlanmaktadır ve herhangi bir garanti içermez.
        
        5. **Sorumluluk Sınırlaması**: Veri Analitik Teknolojileri A.Ş., yazılımın kullanımından doğabilecek herhangi bir zarardan sorumlu değildir.
        
        Bu yazılımda kullanılan açık kaynak bileşenleri hakkında bilgi için 'Açık Kaynak Bildirimleri' dökümanına bakınız.
        """)
with tab7:
    add_feedback_tab()

init_db()
init_database()  # Uygulama başlarken çalışır, tabloyu oluşturur

if __name__ == "__main__":
    # Burada ihtiyaç duyulabilecek başlangıç işlemleri yapılabilir
    pass
