import sqlite3
import sys


def get_user_info_safe(username: str):
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # ✅ 安全做法：使用 ? 作为占位符 (SQLite)
    sql = "SELECT * FROM users WHERE username = ?"

    # 将参数作为元组传递给 execute 方法
    cursor.execute(sql, (username,))

    results = cursor.fetchall()
    conn.close()
    return results


def main():
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = "admin"

    print("--- Testing Unsafe Query ---")
    get_user_info_safe(user_input)


if __name__ == "__main__":
    main()