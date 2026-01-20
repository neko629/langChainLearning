import sqlite3
import sys


def get_user_info_unsafe(username: str):
    """
    这是一个故意包含 SQL 注入漏洞的函数。
    该函数直接将用户输入拼接到 SQL 查询字符串中。
    """
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # 🚨 严重漏洞：直接使用 f-string 拼接 SQL 语句
    # 如果 username 输入为: "' OR '1'='1"
    # SQL 就会变成: SELECT * FROM users WHERE username = '' OR '1'='1'
    sql = f"SELECT * FROM users WHERE username = '{username}'"

    print(f"Executing SQL: {sql}")

    try:
        # 执行拼接后的 SQL
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            print(row)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


def main():
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = "admin"

    print("--- Testing Unsafe Query ---")
    get_user_info_unsafe(user_input)


if __name__ == "__main__":
    main()