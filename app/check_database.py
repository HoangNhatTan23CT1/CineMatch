"""Kiểm tra kết nối và schema MySQL của CineMatch."""
from getpass import getpass
import os
import sys

import database


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not database.get_database_config()["password"]:
        os.environ["DB_PASSWORD"] = getpass("Mật khẩu MySQL của cinematch_app: ")
    database.ensure_schema()
    status = database.healthcheck()
    print(f"Kết nối MySQL thành công: {status['database_name']} · {status['version']}")
    with database.cursor() as db_cursor:
        db_cursor.execute("SELECT COUNT(*) FROM admins")
        admins = db_cursor.fetchone()[0]
        db_cursor.execute("SELECT COUNT(*) FROM ai_videos")
        movies = db_cursor.fetchone()[0]
    print(f"Tài khoản admin: {admins}")
    print(f"Phim nội bộ: {movies}")


if __name__ == "__main__":
    main()
