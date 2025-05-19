import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ana dizini Python yolu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import veri_analizi as va
except:
    st.error("veri_analizi.py dosyası bulunamadı! Ana dizinde bulunduğundan emin olun.")

def rfm_analysis(customer_data=None):
    st.subheader("RFM Analizi")
    
    if customer_data is None:
        # Örnek veri
        try:
            customer_data = va.create_customer_data()
        except:
            st.error("Örnek veri oluşturulamadı. veri_analizi.py dosyasının doğru konumda olduğundan emin olun.")
            return
    
    st.write("""
    RFM, müşteri segmentasyonunda kullanılan önemli bir yöntemdir:
    - **R (Recency)**: Müşterinin en son ne zaman alışveriş yaptığı 
    - **F (Frequency)**: Müşterinin ne sıklıkla alışveriş yaptığı
    - **M (Monetary)**: Müşterinin toplam harcama miktarı
    """)
    
    # RFM Skorlama Ayarları
    col1, col2, col3 = st.columns(3)
    with col1:
        r_weight = st.slider("Recency Ağırlığı", 0.1, 1.0, 0.5, 0.1)
    with col2:
        f_weight = st.slider("Frequency Ağırlığı", 0.1, 1.0, 0.3, 0.1)
    with col3:
        m_weight = st.slider("Monetary Ağırlığı", 0.1, 1.0, 0.2, 0.1)
    
    # RFM skorlarını hesapla (örnek)
    # Gerçek bir uygulamada, müşteri verilerinden RFM skorları hesaplanır
    
    # Recency değeri için 'loyalty_years' kullanılabilir (gerçek recency bunun tersi olacaktır)
    if 'loyalty_years' in customer_data.columns:
        customer_data['recency_score'] = 5 - pd.qcut(customer_data['loyalty_years'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    else:
        # Örnek bir recency skoru oluştur
        customer_data['recency_score'] = np.random.randint(1, 6, size=len(customer_data))
    
    # Frequency için 'purchase_frequency' kullanılabilir
    if 'purchase_frequency' in customer_data.columns:
        customer_data['frequency_score'] = pd.qcut(customer_data['purchase_frequency'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    else:
        # Örnek bir frequency skoru oluştur
        customer_data['frequency_score'] = np.random.randint(1, 6, size=len(customer_data))
    
    # Monetary için 'avg_purchase_value' veya 'customer_value' kullanılabilir
    if 'customer_value' in customer_data.columns:
        customer_data['monetary_score'] = pd.qcut(customer_data['customer_value'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    elif 'avg_purchase_value' in customer_data.columns:
        customer_data['monetary_score'] = pd.qcut(customer_data['avg_purchase_value'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    else:
        # Örnek bir monetary skoru oluştur
        customer_data['monetary_score'] = np.random.randint(1, 6, size=len(customer_data))
    
    # Ağırlıklı RFM skoru hesapla
    customer_data['rfm_score'] = (
        customer_data['recency_score'] * r_weight + 
        customer_data['frequency_score'] * f_weight + 
        customer_data['monetary_score'] * m_weight
    ) / (r_weight + f_weight + m_weight)
    
    # RFM segmentlerini oluştur
    def rfm_segment(row):
        if row['rfm_score'] >= 4.5:
            return 'Champions'
        elif row['rfm_score'] >= 4:
            return 'Loyal Customers'
        elif row['rfm_score'] >= 3:
            return 'Potential Loyalists'
        elif row['rfm_score'] >= 2:
            return 'At Risk'
        else:
            return 'Hibernating'
    
    customer_data['segment'] = customer_data.apply(rfm_segment, axis=1)
    
    # Müşteri grupları
    st.write("### Müşteri Segmentleri")
    
    # Segmentlerin dağılımını göster
    segment_counts = customer_data['segment'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(segment_counts.index, segment_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(segment_counts))))
    
    # Değerleri göster
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height}',
               ha='center', va='bottom')
    
    ax.set_title('RFM Segmentleri Dağılımı')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Müşteri Sayısı')
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Segment açıklamaları
    segments = [
        {"segment": "Champions", "description": "Yakın zamanda alışveriş yapmış, sık alışveriş yapan ve yüksek harcama yapan müşteriler."},
        {"segment": "Loyal Customers", "description": "Ortalama recency, yüksek frequency ve monetary değerlere sahip müşteriler."},
        {"segment": "Potential Loyalists", "description": "Yakın zamanda alışveriş yapmış, orta sıklıkta alışveriş yapan müşteriler."},
        {"segment": "At Risk", "description": "Daha önce düzenli alışveriş yapan ancak uzun süredir alışveriş yapmayan müşteriler."},
        {"segment": "Hibernating", "description": "Uzun süredir alışveriş yapmayan, değer ve sıklık değerleri düşük müşteriler."}
    ]
    
    for segment in segments:
        with st.expander(f"{segment['segment']} ({segment_counts.get(segment['segment'], 0)} müşteri)"):
            st.write(segment['description'])
            
            # İlgili segment için örnek müşterileri göster
            if segment['segment'] in customer_data['segment'].values:
                sample_customers = customer_data[customer_data['segment'] == segment['segment']].head(5)
                st.write("Örnek Müşteriler:")
                st.dataframe(sample_customers[['customer_id', 'avg_purchase_value', 'purchase_frequency', 'customer_value', 'rfm_score']])

def sentiment_analysis():
    st.subheader("Duygu Analizi ve Müşteri Geri Bildirim Analizi")
    
    st.write("""
    Müşteri yorumlarını ve geri bildirimlerini analiz ederek ürün geliştirme ve müşteri memnuniyeti stratejileri oluşturun.
    """)
    
    # Örnek müşteri yorumları
    sample_reviews = [
        {"text": "Ürün beklediğimden çok daha iyi çıktı. Hızlı kargo için teşekkürler!", "rating": 5, "sentiment": "Olumlu"},
        {"text": "Fiyatına göre iyi bir ürün, ancak iki hafta gecikmeli geldi.", "rating": 3, "sentiment": "Nötr"},
        {"text": "Kalitesi çok düşük, dayanıklı değil ve müşteri hizmetleri hiç yardımcı olmadı. Paranızı boşa harcamayın.", "rating": 1, "sentiment": "Olumsuz"},
        {"text": "Tam beklediğim gibi, hızlı teslimat ve kaliteli ürün.", "rating": 5, "sentiment": "Olumlu"},
        {"text": "Ürün işimi görüyor ama tasarımı biraz daha iyi olabilirdi.", "rating": 4, "sentiment": "Olumlu"},
        {"text": "Aldıktan bir ay sonra bozuldu. Garanti süreci çok yorucu.", "rating": 2, "sentiment": "Olumsuz"},
    ]
    
    # Kendi yorumunu girmek için alan
    user_review = st.text_area("Bir müşteri yorumu girin (veya örnek yorumları inceleyin)", 
                              "Bu ürünü çok beğendim, herkese tavsiye ederim!")
    analyze_btn = st.button("Yorumu Analiz Et")
    
    if analyze_btn:
        # Gerçek uygulamada burada doğal dil işleme yapılır
        # Örnek bir sonuç gösteriyoruz
        sentiment_score = 0.8  # 0-1 arası bir değer, 1=çok olumlu
        
        # Duygu skoru gösterimi
        st.write("### Duygu Analizi Sonucu")
        
        # Progress bar ile duygu skoru göster
        st.progress(sentiment_score)
        
        # Duygu skoru metni
        if sentiment_score > 0.7:
            st.success(f"Olumlu Yorum (Skor: {sentiment_score:.2f})")
        elif sentiment_score > 0.4:
            st.info(f"Nötr Yorum (Skor: {sentiment_score:.2f})")
        else:
            st.error(f"Olumsuz Yorum (Skor: {sentiment_score:.2f})")
        
        # Duygusal tepkiyi yorumlama
        if sentiment_score > 0.7:
            st.write("Bu yorum oldukça olumlu! Müşteri ürünü beğenmiş görünüyor.")
        elif sentiment_score > 0.4:
            st.write("Bu yorum nötr/hafif olumlu. Müşteri ürünle ilgili karışık duygulara sahip olabilir.")
        else:
            st.write("Bu yorum olumsuz. Müşteri deneyimini iyileştirmek için nedenleri araştırılmalı.")
    
    # Mevcut yorumların analizi
    st.write("### Yorum Dağılımı")
    
    # Duygu dağılımını göster
    sentiment_counts = {"Olumlu": 0, "Nötr": 0, "Olumsuz": 0}
    for review in sample_reviews:
        sentiment_counts[review["sentiment"]] += 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_colors = {'Olumlu': '#28a745', 'Nötr': '#ffc107', 'Olumsuz': '#dc3545'}
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=[sentiment_colors[s] for s in sentiment_counts.keys()])
    
    ax.set_title('Müşteri Yorumlarının Duygu Dağılımı')
    ax.set_xlabel('Duygu')
    ax.set_ylabel('Yorum Sayısı')
    
    for i, v in enumerate(sentiment_counts.values()):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    st.pyplot(fig)
    
    # Örnek yorumları göster
    st.write("### Örnek Yorumlar")
    
    for i, review in enumerate(sample_reviews):
        col_color = "success" if review["sentiment"] == "Olumlu" else "warning" if review["sentiment"] == "Nötr" else "danger"
        with st.container():
            st.markdown(f"""
            <div style="border-left: 4px solid {'green' if review['sentiment'] == 'Olumlu' else 'orange' if review['sentiment'] == 'Nötr' else 'red'}; padding-left: 10px; margin-bottom: 10px;">
                <p>{review['text']}</p>
                <p style="color: {'green' if review['sentiment'] == 'Olumlu' else 'orange' if review['sentiment'] == 'Nötr' else 'red'}; font-weight: bold;">
                    Değerlendirme: {review['rating']}/5 - {review['sentiment']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Anahtar kelime analizi
    st.write("### Popüler Anahtar Kelimeler")
    
    # Örnek anahtar kelimeler ve frekansları
    keywords = {
        "hızlı teslimat": 15, "kaliteli": 12, "ucuz": 8, "dayanıklı": 7, "kullanışlı": 6,
        "pahalı": 5, "gecikmeli": 4, "kötü paketleme": 3, "müşteri hizmetleri": 3, "bozuk": 2
    }
    
    try:
        from wordcloud import WordCloud
        
        # Kelime bulutu oluşturma
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(keywords)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Müşteri Yorumlarında Popüler Kelimeler')
        
        st.pyplot(fig)
    except:
        # WordCloud kütüphanesi yoksa basit bir bar grafik göster
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(list(keywords.keys()), list(keywords.values()), color='skyblue')
        ax.set_title('Müşteri Yorumlarında Popüler Kelimeler')
        ax.set_xlabel('Frekans')
        
        # Değerleri göster
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width}', 
                   ha='left', va='center')
        
        st.pyplot(fig)