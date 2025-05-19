import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def profitability_analysis():
    st.subheader("Maliyet ve Karlılık Analizi")
    
    st.write("""
    Ürün ve hizmetlerinizin maliyet yapısını analiz edin ve karlılığı artırmak için stratejiler geliştirin.
    """)
    
    # Örnek maliyet verileri
    cost_data = {
        "Satış Geliri": 1000000,
        "Malzeme Maliyeti": -420000,
        "İşçilik Maliyeti": -180000,
        "Brüt Kar": 400000,
        "Pazarlama Giderleri": -120000,
        "Operasyonel Giderler": -95000,
        "Faaliyet Karı": 185000,
        "Vergiler": -37000,
        "Net Kar": 148000
    }
    
    # Şelale (Waterfall) grafiği
    def waterfall_chart(data, title):
        # Renkleri belirleme
        colors = []
        for key, value in data.items():
            if key in ["Brüt Kar", "Faaliyet Karı", "Net Kar"]:
                colors.append('#28a745')  # yeşil
            elif value >= 0:
                colors.append('#17a2b8')  # mavi
            else:
                colors.append('#dc3545')  # kırmızı
        
        # Toplamlar için ara değerler hesaplama
        cumsum = 0
        bottoms = []
        for value in data.values():
            if value >= 0:
                bottoms.append(cumsum)
                cumsum += value
            else:
                bottoms.append(cumsum + value)
                cumsum += value
        
        # Grafiği oluşturma
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(data.keys(), data.values(), bottom=bottoms, color=colors)
        
        # Değerleri gösterme
        for i, (key, value) in enumerate(data.items()):
            if value >= 0:
                label_position = bottoms[i] + value / 2
                color = 'black'
            else:
                label_position = bottoms[i] + value / 2
                color = 'white'
            
            ax.text(i, label_position, f'{abs(value):,.0f}', ha='center', va='center', 
                   color=color, fontweight='bold')
        
        ax.set_title(title)
        ax.set_ylabel('Tutar (₺)')
        
        # Y eksenini düzenleme
        ax.ticklabel_format(axis='y', style='plain')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x/1000:.0f}K'))
        
        # X eksenini döndürme
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    # Şelale grafiğini çiz
    fig = waterfall_chart(cost_data, "Karlılık Analizi (₺)")
    st.pyplot(fig)
    
    # Kar marjları
    st.write("### Kar Marjları")
    
    gross_margin = cost_data["Brüt Kar"] / cost_data["Satış Geliri"] * 100
    operating_margin = cost_data["Faaliyet Karı"] / cost_data["Satış Geliri"] * 100
    net_margin = cost_data["Net Kar"] / cost_data["Satış Geliri"] * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Brüt Kar Marjı", f"%{gross_margin:.1f}")
    with col2:
        st.metric("Faaliyet Kar Marjı", f"%{operating_margin:.1f}")
    with col3:
        st.metric("Net Kar Marjı", f"%{net_margin:.1f}")
    
    # Maliyet kırılımı pasta grafiği
    st.write("### Maliyet Kırılımı")
    
    cost_breakdown = {
        "Malzeme Maliyeti": abs(cost_data["Malzeme Maliyeti"]),
        "İşçilik Maliyeti": abs(cost_data["İşçilik Maliyeti"]),
        "Pazarlama Giderleri": abs(cost_data["Pazarlama Giderleri"]),
        "Operasyonel Giderler": abs(cost_data["Operasyonel Giderler"]),
        "Vergiler": abs(cost_data["Vergiler"])
    }
    
    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(cost_breakdown.values(), 
                                     autopct='%1.1f%%',
                                     textprops={'color': "w", 'fontweight': 'bold'},
                                     colors=plt.cm.tab10.colors)
    
    ax.legend(wedges, cost_breakdown.keys(),
             title="Maliyet Kalemleri",
             loc="center left",
             bbox_to_anchor=(1, 0, 0.5, 1))
    
    ax.set_title("Maliyet Kırılımı")
    
    st.pyplot(fig)
    
    # Kârlılık iyileştirme önerileri
    st.write("### Kârlılık İyileştirme Senaryoları")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price_increase = st.slider("Fiyat Artışı (%)", 0, 30, 5)
        cost_reduction = st.slider("Maliyet Azaltma (%)", 0, 30, 10)
    
    with col2:
        volume_increase = st.slider("Satış Hacmi Artışı (%)", 0, 30, 15)
        mix_improvement = st.slider("Ürün Karması İyileştirme (%)", 0, 30, 8)
    
    # İyileştirme senaryosu hesaplamaları
    scenario_results = []
    
    # Baz senaryo
    base_revenue = cost_data["Satış Geliri"]
    base_costs = abs(cost_data["Malzeme Maliyeti"]) + abs(cost_data["İşçilik Maliyeti"]) + abs(cost_data["Pazarlama Giderleri"]) + abs(cost_data["Operasyonel Giderler"]) + abs(cost_data["Vergiler"])
    base_profit = cost_data["Net Kar"]
    base_margin = base_profit / base_revenue * 100
    
    scenario_results.append({
        "Senaryo": "Mevcut Durum",
        "Gelir": base_revenue,
        "Maliyet": base_costs,
        "Kâr": base_profit,
        "Kâr Marjı": base_margin
    })
    
    # Senaryo 1: Fiyat Artışı
    s1_revenue = base_revenue * (1 + price_increase/100)
    s1_costs = base_costs  # Maliyetler sabit
    s1_profit = s1_revenue - s1_costs
    s1_margin = s1_profit / s1_revenue * 100
    
    scenario_results.append({
        "Senaryo": "Fiyat Artışı",
        "Gelir": s1_revenue,
        "Maliyet": s1_costs,
        "Kâr": s1_profit,
        "Kâr Marjı": s1_margin
    })
    
    # Senaryo 2: Maliyet Azaltma
    s2_revenue = base_revenue
    s2_costs = base_costs * (1 - cost_reduction/100)
    s2_profit = s2_revenue - s2_costs
    s2_margin = s2_profit / s2_revenue * 100
    
    scenario_results.append({
        "Senaryo": "Maliyet Azaltma",
        "Gelir": s2_revenue,
        "Maliyet": s2_costs,
        "Kâr": s2_profit,
        "Kâr Marjı": s2_margin
    })
    
    # Senaryo 3: Hacim Artışı
    s3_revenue = base_revenue * (1 + volume_increase/100)
    s3_costs = base_costs * (1 + volume_increase/100 * 0.8)  # Hacim arttıkça maliyetler de artar ama ölçek ekonomisi nedeniyle daha az
    s3_profit = s3_revenue - s3_costs
    s3_margin = s3_profit / s3_revenue * 100
    
    scenario_results.append({
        "Senaryo": "Hacim Artışı",
        "Gelir": s3_revenue,
        "Maliyet": s3_costs,
        "Kâr": s3_profit,
        "Kâr Marjı": s3_margin
    })
    
    # Senaryo 4: Kombine Strateji
    s4_revenue = base_revenue * (1 + price_increase/100) * (1 + volume_increase/100 * 0.7)  # Fiyat artışı hacmi biraz düşürebilir
    s4_costs = base_costs * (1 - cost_reduction/100) * (1 + volume_increase/100 * 0.7 * 0.8)
    s4_profit = s4_revenue - s4_costs
    s4_margin = s4_profit / s4_revenue * 100
    
    scenario_results.append({
        "Senaryo": "Kombine Strateji",
        "Gelir": s4_revenue,
        "Maliyet": s4_costs,
        "Kâr": s4_profit,
        "Kâr Marjı": s4_margin
    })
    
    # Sonuçları gösterme
    scenario_df = pd.DataFrame(scenario_results)
    scenario_df["Kâr Artışı"] = scenario_df["Kâr"] - base_profit
    scenario_df["Kâr Artışı (%)"] = (scenario_df["Kâr"] - base_profit) / base_profit * 100
    
    # Formatla
    formatted_df = scenario_df.copy()
    for col in ["Gelir", "Maliyet", "Kâr", "Kâr Artışı"]:
        formatted_df[col] = formatted_df[col].map(lambda x: f"₺{x:,.0f}")
    
    for col in ["Kâr Marjı", "Kâr Artışı (%)"]:
        formatted_df[col] = formatted_df[col].map(lambda x: f"%{x:.1f}")
    
    st.table(formatted_df)
    
    # Kârlılık iyileştirme önerileri
    st.write("### Kârlılık İyileştirme Önerileri")
    
    recommendations = [
        "**Fiyatlandırma Stratejisi:** Ürün değerini daha iyi yansıtmak için fiyatlandırma stratejilerini gözden geçirin. Değer bazlı fiyatlandırma ve farklılaştırılmış fiyatlandırma teknikleri uygulayın.",
        "**Tedarik Zinciri Optimizasyonu:** Tedarikçilerle daha iyi anlaşmalar yaparak ve daha büyük hacimli siparişler vererek malzeme maliyetlerini düşürün.",
        "**Operasyonel Verimlilik:** İş süreçlerini otomatikleştirerek ve optimizasyon teknikleri uygulayarak operasyonel maliyetleri azaltın.",
"**Pazarlama ROI'sini Artırma:** Pazarlama harcamalarının etkinliğini artırın ve düşük performans gösteren kampanyaları sonlandırın.",
        "**Ürün Karması Optimizasyonu:** Daha yüksek kâr marjına sahip ürünlere odaklanarak genel kârlılığı artırın."
    ]
    
    for i, rec in enumerate(recommendations):
        st.markdown(f"{i+1}. {rec}")

def trend_analysis():
    st.subheader("Trend Analizi ve Pazar Tahminlemesi")
    
    st.write("""
    Veri bilimi ve derin öğrenme ile pazar trendlerini belirleyin ve gelecek eğilimleri tahmin edin.
    """)
    
    # Trend ölçütleri
    trend_metrics = {
        "Arama Hacmi Artışı": 78,
        "Sosyal Medya Paylaşımları": 42,
        "Haber Makaleleri": 156,
        "Yatırım Aktivitesi": 23,
        "Patentler ve Ar-Ge": 17
    }
    
    # Trend skor kartı
    st.write("### Trend Skorları")
    
    cols = st.columns(len(trend_metrics))
    for i, (metric, value) in enumerate(trend_metrics.items()):
        with cols[i]:
            st.metric(
                label=metric,
                value=value,
                delta=f"{np.random.randint(-15, 30)}%"
            )
    
    # Anahtar kelime trend grafiği
    st.write("### Anahtar Kelime Trend Analizi")
    
    # Örnek anahtar kelimeler ve trenleri
    keywords = ["Yapay Zeka", "Bulut Bilişim", "Nesnelerin İnterneti", "Blok Zinciri", "Sanal Gerçeklik"]
    
    # 24 aylık örnek trend verileri (son 2 yıl)
    num_months = 24
    months = pd.date_range(start=pd.Timestamp.now() - pd.DateOffset(months=num_months-1), periods=num_months, freq='M')
    
    # Her anahtar kelime için farklı trend desenleri
    trend_data = {}
    np.random.seed(42)
    
    # Yapay Zeka - sürekli artan
    trend_data[keywords[0]] = [100 + i*5 + np.random.normal(0, 10) for i in range(num_months)]
    
    # Bulut Bilişim - yavaş artan sonra plato
    trend_data[keywords[1]] = [150 + 30 * np.log(i+1) + np.random.normal(0, 8) for i in range(num_months)]
    
    # Nesnelerin İnterneti - mevsimsel
    trend_data[keywords[2]] = [120 + 20 * np.sin(i/12 * 2 * np.pi) + i*2 + np.random.normal(0, 5) for i in range(num_months)]
    
    # Blok Zinciri - önce artış sonra düşüş
    peak = num_months // 2
    trend_data[keywords[3]] = [50 + 100 * (1 - abs(i - peak) / peak) + np.random.normal(0, 10) for i in range(num_months)]
    
    # Sanal Gerçeklik - ani artış sonra yavaşlama
    trend_data[keywords[4]] = [70 + 60 * (1 - np.exp(-i/5)) + np.random.normal(0, 7) for i in range(num_months)]
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for keyword in keywords:
        ax.plot(months, trend_data[keyword], marker='o', markersize=3, linewidth=2, label=keyword)
    
    ax.set_title("Anahtar Kelime Arama Trendleri (2 Yıllık)")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Trend Değeri (100=Başlangıç)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # X ekseni formatı
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Trend tahminleme
    st.write("### Trend Tahminleme")
    
    forecast_months = st.slider("Tahmin Ayı", 1, 12, 6)
    
    # Tahmin verilerini oluştur
    future_months = pd.date_range(start=months[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='M')
    all_months = months.append(future_months)
    
    # Her anahtar kelime için tahmin
    forecast_data = {}
    
    for keyword in keywords:
        # Mevcut verilere basit bir model uydur (polinom regresyon)
        x = np.arange(len(months))
        y = trend_data[keyword]
        
        degree = 2  # 2. dereceden polinom
        model = np.poly1d(np.polyfit(x, y, degree))
        
        # Geleceği tahmin et
        future_x = np.arange(len(months), len(months) + len(future_months))
        future_y = model(future_x)
        
        # Gerçek veriler + tahmin
        forecast_data[keyword] = np.concatenate([trend_data[keyword], future_y])
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for keyword in keywords:
        # Gerçek veriler
        ax.plot(months, trend_data[keyword], marker='o', markersize=3, linewidth=2, label=keyword)
        
        # Tahmin (noktalı çizgi)
        ax.plot(future_months, forecast_data[keyword][-len(future_months):], marker='o', markersize=3, 
               linestyle='--', linewidth=2, alpha=0.7, color=ax.get_lines()[-1].get_color())
    
    # Gerçek vs tahmin sınırını göster
    ax.axvline(x=months[-1], color='gray', linestyle='-', alpha=0.5)
    ax.annotate('Tahmin', xy=(future_months[0], ax.get_ylim()[0]), xytext=(10, 10), 
               textcoords='offset points', color='gray', fontsize=10)
    
    ax.set_title(f"Anahtar Kelime Trend Tahmini (Gelecek {forecast_months} Ay)")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Trend Değeri")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # X ekseni formatı
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Trend analizi özeti
    st.write("### Trend Analizi Özeti")
    
    # Kategorik trendler (Yükseliyor, Düşüyor, Sabit, vb.)
    trend_summary = []
    
    for keyword in keywords:
        # Son 6 ay ve önceki 6 ay karşılaştırması ile trend yönünü belirle
        last_6_months = np.mean(trend_data[keyword][-6:])
        previous_6_months = np.mean(trend_data[keyword][-12:-6])
        percent_change = (last_6_months - previous_6_months) / previous_6_months * 100
        
        # Tahmini büyüme/düşüş
        forecast_value = forecast_data[keyword][-1]
        current_value = trend_data[keyword][-1]
        forecast_growth = (forecast_value - current_value) / current_value * 100
        
        # Trend yönü belirle
        if percent_change > 20:
            trend_direction = "Hızlı Yükseliş"
            trend_color = "success"
        elif percent_change > 5:
            trend_direction = "Yükseliş"
            trend_color = "success"
        elif percent_change > -5:
            trend_direction = "Sabit"
            trend_color = "warning"
        elif percent_change > -20:
            trend_direction = "Düşüş"
            trend_color = "danger"
        else:
            trend_direction = "Hızlı Düşüş"
            trend_color = "danger"
        
        trend_summary.append({
            "Anahtar Kelime": keyword,
            "Mevcut Trend": trend_direction,
            "Değişim (%)": percent_change,
            "Tahmini Büyüme (%)": forecast_growth
        })
    
    # DataFrame oluştur
    trend_df = pd.DataFrame(trend_summary)
    
    # Renklendirme için HTML formatında göster
    html_table = "<table width='100%' style='text-align: left;'>"
    html_table += "<tr><th>Anahtar Kelime</th><th>Mevcut Trend</th><th>Değişim (%)</th><th>Tahmini Büyüme (%)</th></tr>"
    
    for _, row in trend_df.iterrows():
        trend_color = "green" if row["Mevcut Trend"] in ["Yükseliş", "Hızlı Yükseliş"] else "red" if row["Mevcut Trend"] in ["Düşüş", "Hızlı Düşüş"] else "orange"
        forecast_color = "green" if row["Tahmini Büyüme (%)"] > 0 else "red"
        
        html_table += "<tr>"
        html_table += f"<td>{row['Anahtar Kelime']}</td>"
        html_table += f"<td style='color: {trend_color};'>{row['Mevcut Trend']}</td>"
        html_table += f"<td>{row['Değişim (%)']:.1f}%</td>"
        html_table += f"<td style='color: {forecast_color};'>{row['Tahmini Büyüme (%)']:.1f}%</td>"
        html_table += "</tr>"
    
    html_table += "</table>"
    
    st.markdown(html_table, unsafe_allow_html=True)