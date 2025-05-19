import streamlit as st
import matplotlib.pyplot as plt
import io
import base64

def create_clickable_plot(fig, title, key=None):
    """Tıklandığında büyüyen basit bir grafik oluşturur"""
    
    # Benzersiz bir anahtar oluştur
    plot_key = key if key else title.replace(" ", "_").lower()
    
    # Grafik görüntüsünü kaydet
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Başlık ve buton
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{title}**")
    with col2:
        expand_button = st.button("Büyüt", key=f"btn_{plot_key}")
    
    # Küçük görünüm
    st.image(buf, width=400)
    
    # Büyütme kontrolü
    if expand_button:
        st.session_state[f"show_{plot_key}"] = True
    
    # Büyütülmüş görünüm 
    if f"show_{plot_key}" in st.session_state and st.session_state[f"show_{plot_key}"]:
        with st.expander("Büyütülmüş Görünüm (Kapatmak için tıklayın)", expanded=True):
            st.image(buf, use_column_width=True)
            if st.button("Kapat", key=f"close_{plot_key}"):
                st.session_state[f"show_{plot_key}"] = False
    
    # Grafiği serbest bırak
    plt.close(fig)