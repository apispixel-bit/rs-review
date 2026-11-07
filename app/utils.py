import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='db_hospital_sentiment'
    )

def get_all_reviews():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM reviews ORDER BY id DESC")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results