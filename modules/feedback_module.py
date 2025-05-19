# modules/feedback_module.py

import streamlit as st
import pandas as pd
import datetime
from modules.database_utils import append_dataframe, read_table

def init_db():
    # Gerekirse buraya başka tablo kurulumları eklenebilir
    pass  # Çünkü database_utils modülü zaten tabloyu oluşturuyor

def add_feedback_tab():
    st.subheader("📝 Geri Bildirim Formu")
    st.write("Uygulama hakkında düşüncelerinizi bizimle paylaşın.")

    name = st.text_input("Adınız")
    email = st.text_input("E-posta Adresiniz")
    message = st.text_area("Mesajınız")

    if st.button("Gönder"):
        if name and email and message:
            new_row = pd.DataFrame([{
                "name": name,
                "email": email,
                "message": message,
                "timestamp": datetime.datetime.now()
            }])
            append_dataframe(new_row, "feedback")
            st.success("Geri bildiriminiz kaydedildi. Teşekkür ederiz!")
        else:
            st.error("Lütfen tüm alanları doldurunuz.")

    if st.checkbox("Gönderilen Mesajları Göster"):
        try:
            df = read_table("feedback")
            st.dataframe(df)
        except Exception as e:
            st.warning(f"Veri alınamadı: {e}")
