# modules/feedback_module.py

import streamlit as st
import sqlite3
import datetime
import pandas as pd 

DB_PATH = "mursistva.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_feedback_tab():
    st.subheader("📝 Geri Bildirim Formu")
    st.write("Uygulama hakkında düşüncelerinizi bizimle paylaşın.")

    name = st.text_input("Adınız")
    email = st.text_input("E-posta Adresiniz")
    message = st.text_area("Mesajınız")

    if st.button("Gönder"):
        if name and email and message:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO feedback (name, email, message) VALUES (?, ?, ?)", 
                      (name, email, message))
            conn.commit()
            conn.close()
            st.success("Geri bildiriminiz kaydedildi. Teşekkür ederiz!")
        else:
            st.error("Lütfen tüm alanları doldurunuz.")

    if st.checkbox("Gönderilen Mesajları Göster"):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
        st.dataframe(df)
        conn.close()
