import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_dashboard():
    st.subheader("İnteraktif Gösterge Paneli")
    
    # İki sütuna bölünmüş dinamik metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Satış", "₺1.25M", "+12.5%")
    with col2:
        st.metric("Ortalama Sepet Değeri", "₺480", "+8.2%")
    with col3:
        st.metric("Müşteri Sayısı", "2,845", "-2.1%")
    with col4:
        st.metric("Tahmin Doğruluğu", "%92.5", "+3.4%")
    
    # Tarih aralığı seçimi
    date_range = st.date_input("Tarih Aralığı Seçin", 
                              [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-12-31")])
    
    # Grafikler
    col1, col2 = st.columns(2)
    
    with col1:
        # Örnek satış trendi grafiği
        st.write("#### Satış Trendi")
        
        # Örnek veri
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        sales_values = [120000, 118000, 125000, 135000, 140000, 150000, 148000, 152000, 149000, 155000, 160000, 170000]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, sales_values, marker='o')
        ax.set_title('Aylık Satış Trendi')
        ax.set_xlabel('Tarih')
        ax.set_ylabel('Satış (₺)')
        ax.grid(True, alpha=0.3)
        
        # X ekseni formatlaması
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: pd.to_datetime(x).strftime('%b %Y')))
        plt.xticks(rotation=45)
        
        # Y ekseni formatlaması
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x/1000:.0f}K'))
        
        st.pyplot(fig)
    
    with col2:
        # Örnek kategori dağılımı grafiği
        st.write("#### Kategori Satış Dağılımı")
        
        # Örnek veri
        categories = ['Elektronik', 'Giyim', 'Mobilya', 'Kitap', 'Kozmetik']
        category_sales = [450000, 320000, 280000, 150000, 170000]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(categories, category_sales, color=plt.cm.tab10.colors)
        
        # Değerleri göster
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                   f'₺{height/1000:.0f}K',
                   ha='center', va='bottom')
        
        ax.set_title('Kategori Bazlı Satışlar')
        ax.set_xlabel('Kategori')
        ax.set_ylabel('Satış (₺)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Y ekseni formatlaması
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x/1000:.0f}K'))
        
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
    
    # Günlük satış tablosu
    st.write("#### Son 7 Günün Satış Verileri")
    
    # Örnek veri
    recent_dates = pd.date_range(end=pd.Timestamp.now(), periods=7, freq='D')
    recent_sales = [15800, 14200, 16500, 17800, 18900, 12400, 14700]
    recent_orders = [124, 98, 115, 132, 145, 87, 102]
    
    recent_data = pd.DataFrame({
        'Tarih': recent_dates,
        'Satış': recent_sales,
        'Sipariş Sayısı': recent_orders,
        'Ortalama Sepet': [s/o for s, o in zip(recent_sales, recent_orders)]
    })
    
    # Tarih formatını düzenle
    recent_data['Tarih'] = recent_data['Tarih'].dt.strftime('%d %b %Y')
    
    # Sayı formatlarını düzenle
    recent_data['Satış'] = recent_data['Satış'].apply(lambda x: f'₺{x:,.2f}')
    recent_data['Ortalama Sepet'] = recent_data['Ortalama Sepet'].apply(lambda x: f'₺{x:,.2f}')
    
    st.dataframe(recent_data)