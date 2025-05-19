import sqlite3
import pandas as pd
import datetime

DB_PATH = "mursistva.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def save_dataframe(df: pd.DataFrame, table_name: str):
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def append_dataframe(df: pd.DataFrame, table_name: str):
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

def read_table(table_name: str):
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df
