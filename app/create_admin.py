"""Tạo hoặc đặt lại tài khoản quản trị CineMatch."""
from getpass import getpass
import os
import sys

from werkzeug.security import generate_password_hash

import database


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = database.get_database_config()
    if not config["password"]:
        os.environ["DB_PASSWORD"] = getpass("Mật khẩu MySQL của cinematch_app: ")

    database.ensure_schema()
    username = input("Tên đăng nhập admin: ").strip()
    password = getpass("Mật khẩu admin: ")
    confirmation = getpass("Nhập lại mật khẩu admin: ")
    if not username or not password:
        raise SystemExit("Tên đăng nhập và mật khẩu không được để trống.")
    if password != confirmation:
        raise SystemExit("Hai mật khẩu không khớp.")

    with database.cursor(commit=True) as db_cursor:
        db_cursor.execute(
            """
            INSERT INTO admins (username, password_hash)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE password_hash = VALUES(password_hash)
            """,
            (username, generate_password_hash(password)),
        )
    print(f"Đã tạo/cập nhật tài khoản admin: {username}")


if __name__ == "__main__":
    main()
