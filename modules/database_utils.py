# modules/database_utils.py
import sqlite3
import pandas as pd
import os

DB_PATH = "mursistva.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_database():
    conn = get_connection()
    conn.execute('''
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

def save_dataframe(df: pd.DataFrame, table_name: str, mode='replace'):
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=mode, index=False)
    conn.close()

def read_table(table_name: str):
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df
