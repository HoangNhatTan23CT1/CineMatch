"""Blueprint quản trị phim độc quyền và phim chưa có trên TMDB."""
from __future__ import annotations

import hmac
import secrets
from functools import wraps
from typing import Any, Callable, Dict

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash

import custom_movie_service
import database


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


@admin_bp.app_context_processor
def inject_admin_helpers() -> Dict[str, Any]:
    return {"csrf_token": _csrf_token}


@admin_bp.before_request
def verify_csrf():
    if request.method != "POST":
        return None
    submitted = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if not expected or not hmac.compare_digest(submitted, expected):
        flash("Phiên biểu mẫu đã hết hạn. Vui lòng thử lại.", "error")
        return redirect(request.referrer or url_for("admin.login"))
    return None


def admin_required(view: Callable):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            return redirect(url_for("admin.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def _ensure_schema() -> str:
    try:
        database.ensure_schema()
        return ""
    except database.DatabaseUnavailable as exc:
        return str(exc)


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("admin_id"):
        return redirect(url_for("admin.dashboard"))

    database_error = _ensure_schema()
    if request.method == "POST" and not database_error:
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with database.cursor(dictionary=True, commit=True) as db_cursor:
            db_cursor.execute(
                "SELECT id, username, password_hash FROM admins WHERE username = %s LIMIT 1",
                (username,),
            )
            admin = db_cursor.fetchone()
            if admin and check_password_hash(admin["password_hash"], password):
                db_cursor.execute("UPDATE admins SET last_login_at = NOW() WHERE id = %s", (admin["id"],))
                session.clear()
                session["admin_id"] = int(admin["id"])
                session["admin_username"] = admin["username"]
                session["csrf_token"] = secrets.token_urlsafe(32)
                next_url = request.args.get("next", "")
                if next_url.startswith("/admin"):
                    return redirect(next_url)
                return redirect(url_for("admin.dashboard"))
        flash("Tên đăng nhập hoặc mật khẩu không đúng.", "error")

    return render_template("admin/login.html", database_error=database_error)


@admin_bp.post("/logout")
@admin_required
def logout():
    session.clear()
    return redirect(url_for("admin.login"))


@admin_bp.get("")
@admin_bp.get("/")
@admin_required
def dashboard():
    try:
        movies = custom_movie_service.list_movies(limit=250)
        published = sum(1 for movie in movies if movie["is_published"])
        ai_movies = sum(1 for movie in movies if movie["is_ai_generated"])
        return render_template(
            "admin/dashboard.html",
            movies=movies,
            stats={"total": len(movies), "published": published, "drafts": len(movies) - published, "ai": ai_movies},
        )
    except database.DatabaseUnavailable as exc:
        return render_template("admin/database_error.html", error=str(exc)), 503


@admin_bp.route("/movies/new", methods=["GET", "POST"])
@admin_required
def create_movie():
    if request.method == "POST":
        try:
            movie_id = custom_movie_service.create_movie(request.form, int(session["admin_id"]))
            flash("Đã thêm phim vào thư viện CineMatch.", "success")
            return redirect(url_for("admin.edit_movie", movie_id=movie_id))
        except (ValueError, database.DatabaseUnavailable) as exc:
            flash(f"Không thể lưu phim: {exc}", "error")
    return render_template("admin/movie_form.html", movie=None, form_data=request.form)


@admin_bp.route("/movies/<int:movie_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_movie(movie_id: int):
    try:
        movie = custom_movie_service.get_movie(movie_id, include_unpublished=True)
    except database.DatabaseUnavailable as exc:
        return render_template("admin/database_error.html", error=str(exc)), 503
    if not movie:
        flash("Không tìm thấy phim cần sửa.", "error")
        return redirect(url_for("admin.dashboard"))

    if request.method == "POST":
        try:
            custom_movie_service.update_movie(movie_id, request.form)
            flash("Đã cập nhật thông tin phim.", "success")
            return redirect(url_for("admin.edit_movie", movie_id=movie_id))
        except (ValueError, database.DatabaseUnavailable) as exc:
            flash(f"Không thể lưu phim: {exc}", "error")
    return render_template("admin/movie_form.html", movie=movie, form_data=request.form)


@admin_bp.post("/movies/<int:movie_id>/publish")
@admin_required
def toggle_publish(movie_id: int):
    published = request.form.get("published") == "1"
    custom_movie_service.set_published(movie_id, published)
    flash("Đã xuất bản phim." if published else "Đã chuyển phim về bản nháp.", "success")
    return redirect(request.referrer or url_for("admin.dashboard"))


@admin_bp.post("/movies/<int:movie_id>/delete")
@admin_required
def delete_movie(movie_id: int):
    custom_movie_service.delete_movie(movie_id)
    flash("Đã xóa phim khỏi thư viện nội bộ.", "success")
    return redirect(url_for("admin.dashboard"))
