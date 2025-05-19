import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ana dizini Python yolu ekle - veri_analizi.py dosyasını import etmek için gerekli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import veri_analizi as va
except:
    st.error("veri_analizi.py dosyası bulunamadı! Ana dizinde bulunduğundan emin olun.")

def seasonal_analysis(sales_df=None):
    if sales_df is None:
        # Örnek veri oluştur
        try:
            sales_df = va.create_sample_sales_data()
        except:
            st.error("Örnek veri oluşturulamadı. veri_analizi.py dosyasının doğru konumda olduğundan emin olun.")
            return
        
    st.subheader("Mevsimsel ve Dönemsel Analiz")
    
    tab_daily, tab_weekly, tab_monthly, tab_yearly = st.tabs(["Günlük", "Haftalık", "Aylık", "Yıllık"])
    
    with tab_weekly:
        st.write("Haftanın günlerine göre satış dağılımı")
        
        # Haftanın günlerine göre analiz
        if 'weekday' in sales_df.columns:
            weekday_counts = sales_df.groupby('weekday')['sales'].mean()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            weekdays = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            ax.bar(weekdays[:len(weekday_counts)], weekday_counts, color='royalblue')
            ax.set_title("Haftanın Günlerine Göre Ortalama Satışlar")
            ax.set_ylabel("Ortalama Satış Miktarı")
            st.pyplot(fig)
        else:
            st.warning("Veri setinde 'weekday' sütunu bulunamadı.")
    
    with tab_monthly:
        st.write("Aylara göre satış dağılımı")
        
        # Aylara göre analiz
        if 'month' in sales_df.columns and 'date' in sales_df.columns:
            # Tarih sütununu datetime formatına çevir
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            
            # Aylık satışları hesapla
            monthly_sales = sales_df.groupby(sales_df['date'].dt.month)['sales'].mean()
            
            # Ay isimlerini oluştur
            month_names = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", 
                           "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(1, 13), [monthly_sales.get(i, 0) for i in range(1, 13)], color='seagreen')
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)
            ax.set_title("Aylara Göre Ortalama Satışlar")
            ax.set_ylabel("Ortalama Satış Miktarı")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Veri setinde 'month' veya 'date' sütunu bulunamadı.")
    
    with tab_yearly:
        st.write("Yıllara göre satış trendi")
        
        # Yıllara göre analiz
        if 'year' in sales_df.columns and 'date' in sales_df.columns:
            # Tarih sütununu datetime formatına çevir
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            
            # Yıllık satışları hesapla
            yearly_sales = sales_df.groupby(sales_df['date'].dt.year)['sales'].sum()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_sales.index, yearly_sales.values, marker='o', linewidth=2, color='purple')
            ax.set_title("Yıllık Toplam Satışlar")
            ax.set_xlabel("Yıl")
            ax.set_ylabel("Toplam Satış Miktarı")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("Veri setinde 'year' veya 'date' sütunu bulunamadı.")
    
    with tab_daily:
        st.write("Günlük satış dağılımı")
        
        # Günlük analiz - 30 günlük hareketli ortalama
        if 'date' in sales_df.columns:
            # Tarih sütununu datetime formatına çevir ve indeks olarak ayarla
            sales_df_ts = sales_df.copy()
            sales_df_ts['date'] = pd.to_datetime(sales_df_ts['date'])
            sales_df_ts.set_index('date', inplace=True)
            
            # 30 günlük hareketli ortalama
            rolling_mean = sales_df_ts['sales'].rolling(window=30).mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sales_df_ts.index, sales_df_ts['sales'], alpha=0.5, color='gray', label='Günlük Satışlar')
            ax.plot(rolling_mean.index, rolling_mean, color='red', linewidth=2, label='30 Günlük Hareketli Ortalama')
            ax.set_title("Günlük Satışlar ve Hareketli Ortalama")
            ax.set_xlabel("Tarih")
            ax.set_ylabel("Satış Miktarı")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("Veri setinde 'date' sütunu bulunamadı.")

def price_analysis():
    st.subheader("Fiyat Elastikiyeti ve Promosyon Etki Analizi")
    
    st.write("""
    Fiyat elastikiyeti, fiyat değişikliklerinin talep üzerindeki etkisini ölçer.
    Elastikiyet katsayısı > 1 ise "elastik" (fiyat değişimlerine duyarlı) demektir.
    """)
    
    # Ürün seçimi
    products = ["Akıllı Telefon", "Laptop", "Tablet", "Kulaklık", "Akıllı Saat"]
    product = st.selectbox("Ürün Seçin", products)
    
    # Elastikiyet grafiği
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(80, 120, 100)  # fiyat değişim yüzdesi
    
    # Farklı ürünler için farklı elastikiyet değerleri
    elasticity = -1.5  # varsayılan
    if product == "Akıllı Telefon":
        elasticity = -0.8
    elif product == "Laptop":
        elasticity = -1.1
    elif product == "Tablet":
        elasticity = -1.5
    elif product == "Kulaklık":
        elasticity = -2.2
    elif product == "Akıllı Saat":
        elasticity = -1.8
    
    y = 100 + elasticity * (x - 100)  # talep değişimi
    ax.plot(x, y)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.3)
    ax.axvline(x=100, color='r', linestyle='--', alpha=0.3)
    ax.set_title(f"{product} için Fiyat Elastikiyeti")
    ax.set_xlabel("Fiyat Değişimi (%)")
    ax.set_ylabel("Talep Değişimi (%)")
    ax.grid(True, alpha=0.3)
    
    ax.fill_between(x[x < 100], y[x < 100], 100, alpha=0.2, color='green')
    ax.fill_between(x[x > 100], y[x > 100], 100, alpha=0.2, color='red')
    
    st.pyplot(fig)
    
    st.write(f"**{product}** için elastikiyet katsayısı: **{elasticity}**")
    st.write(f"Bu, fiyatı %10 artırmanın talebi yaklaşık %{-elasticity * 10:.1f} düşüreceği anlamına gelir.")
    
# Promosyon etki analizi
    st.write("### Promosyon Etki Analizi")
    
    promo_data = {
        "İndirim": [5, 10, 15, 20, 25, 30],
        "Satış Artışı (%)": [8, 18, 25, 40, 48, 55],
        "Karlılık Etkisi (%)": [3, 6, 8, 5, -2, -10]
    }
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('İndirim Oranı (%)')
    ax1.set_ylabel('Satış Artışı (%)', color='tab:blue')
    ax1.plot(promo_data["İndirim"], promo_data["Satış Artışı (%)"], 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Karlılık Etkisi (%)', color='tab:red')
    ax2.plot(promo_data["İndirim"], promo_data["Karlılık Etkisi (%)"], 'o-', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    ax1.set_title(f"{product} için Promosyon Etki Analizi")
    ax1.grid(True, alpha=0.3)
    
    # Optimal noktayı göster
    optimal_idx = promo_data["Karlılık Etkisi (%)"].index(max(promo_data["Karlılık Etkisi (%)"]))
    optimal_discount = promo_data["İndirim"][optimal_idx]
    ax1.axvline(x=optimal_discount, color='green', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    st.write(f"**{product}** için optimal indirim oranı: **%{optimal_discount}**")
    st.write(f"Bu indirim oranı, satışları %{promo_data['Satış Artışı (%)'][optimal_idx]} artırırken karlılığı %{promo_data['Karlılık Etkisi (%)'][optimal_idx]} yükseltir.")
    
    