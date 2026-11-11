"""Kết nối MySQL và khởi tạo schema cho khu vực quản trị CineMatch."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import mysql.connector
from mysql.connector import MySQLConnection


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
SCHEMA_PATH = ROOT / "schema.sql"


class DatabaseUnavailable(RuntimeError):
    """Lỗi cấu hình hoặc kết nối MySQL có thể hiển thị an toàn cho quản trị viên."""


def _read_env_file() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_env_value(name: str, default: str = "") -> str:
    """Đọc một cấu hình từ biến môi trường hoặc file .env của dự án."""
    return os.environ.get(name, _read_env_file().get(name, default)).strip()


def get_database_config() -> Dict[str, Any]:
    def value(name: str, default: str = "") -> str:
        return get_env_value(name, default)

    return {
        "host": value("DB_HOST", "127.0.0.1"),
        "port": int(value("DB_PORT", "3306")),
        "user": value("DB_USER", "cinematch_app"),
        "password": value("DB_PASSWORD"),
        "database": value("DB_NAME", "cinematch"),
        "charset": "utf8mb4",
        "collation": "utf8mb4_unicode_ci",
        "connection_timeout": 5,
    }


def connect() -> MySQLConnection:
    config = get_database_config()
    if not config["password"]:
        raise DatabaseUnavailable(
            "Chưa cấu hình DB_PASSWORD trong file .env. "
            "Hãy nhập mật khẩu của tài khoản cinematch_app rồi tải lại trang."
        )
    try:
        return mysql.connector.connect(**config)
    except mysql.connector.Error as exc:
        raise DatabaseUnavailable(
            f"Không thể kết nối MySQL {config['host']}:{config['port']}/{config['database']}: {exc.msg}"
        ) from exc


@contextmanager
def cursor(*, dictionary: bool = False, commit: bool = False) -> Iterator[Any]:
    connection = connect()
    db_cursor = connection.cursor(dictionary=dictionary)
    try:
        yield db_cursor
        if commit:
            connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        db_cursor.close()
        connection.close()


def ensure_schema() -> None:
    """Chạy các câu CREATE TABLE IF NOT EXISTS trong schema.sql."""
    if not SCHEMA_PATH.exists():
        raise DatabaseUnavailable(f"Không tìm thấy schema: {SCHEMA_PATH}")
    statements = [part.strip() for part in SCHEMA_PATH.read_text(encoding="utf-8").split(";") if part.strip()]
    connection = connect()
    db_cursor = connection.cursor()
    try:
        required_tables = {
            "admins", "genres", "topics", "ai_videos", "ai_video_genres", "ai_video_topics"
        }
        db_cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
        )
        existing_tables = {row[0] for row in db_cursor.fetchall()}
        if not required_tables.issubset(existing_tables):
            for statement in statements:
                db_cursor.execute(statement)

        required_columns = {
            "admins": {
                "last_login_at": "TIMESTAMP NULL DEFAULT NULL",
            },
            "ai_videos": {
                "content_type": "ENUM('ai_short','new_movie') NOT NULL DEFAULT 'ai_short'",
                "original_title": "VARCHAR(255) NOT NULL DEFAULT ''",
                "release_date": "DATE NULL",
                "original_language": "VARCHAR(12) NOT NULL DEFAULT 'vi'",
                "backdrop_url": "VARCHAR(2048) NOT NULL DEFAULT ''",
                "cast_json": "JSON NULL",
                "vote_average": "DECIMAL(3,1) NOT NULL DEFAULT 0.0",
                "vote_count": "INT UNSIGNED NOT NULL DEFAULT 0",
                "source_label": "VARCHAR(120) NOT NULL DEFAULT 'CineMatch Exclusive'",
                "created_by": "INT UNSIGNED NULL",
            },
        }
        for table, columns in required_columns.items():
            db_cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = %s",
                (table,),
            )
            existing = {row[0] for row in db_cursor.fetchall()}
            for column, definition in columns.items():
                if column not in existing:
                    db_cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")
        connection.commit()
    except mysql.connector.Error as exc:
        connection.rollback()
        raise DatabaseUnavailable(f"Không thể khởi tạo schema CineMatch: {exc.msg}") from exc
    finally:
        db_cursor.close()
        connection.close()


def healthcheck() -> Dict[str, Any]:
    with cursor(dictionary=True) as db_cursor:
        db_cursor.execute("SELECT DATABASE() AS database_name, VERSION() AS version")
        return db_cursor.fetchone()
