"""CRUD trên schema ai_videos hiện có và chuyển đổi sang định dạng CineMatch."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

import database


SEPARATOR = "\x1f"


def _safe_url(value: Any) -> str:
    url = str(value or "").strip()
    return url if url.lower().startswith(("https://", "http://")) else ""


def _split(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    text = str(value)
    separator = SEPARATOR if SEPARATOR in text else ","
    return [item.strip() for item in text.split(separator) if item.strip()]


def _form_list(value: Any) -> List[str]:
    if isinstance(value, str):
        parts = value.replace("\n", ",").split(",")
    else:
        parts = value or []
    return list(dict.fromkeys(str(part).strip() for part in parts if str(part).strip()))


def _json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return _form_list(value)


def to_movie(row: Dict[str, Any]) -> Dict[str, Any]:
    release_date = row.get("release_date")
    if isinstance(release_date, (date, datetime)):
        release_date = release_date.isoformat()[:10]
    release_date = str(release_date or "")
    db_id = int(row["id"])
    duration_seconds = int(row.get("duration_seconds") or 0)
    content_type = row.get("content_type") or "ai_short"
    trailer_url = _safe_url(row.get("video_url"))
    source_url = _safe_url(row.get("source_url"))
    sources = []
    if source_url:
        sources.append({
            "label": row.get("source_label") or "Nguồn tác phẩm",
            "url": source_url,
            "kind": "official",
        })
    if trailer_url and trailer_url != source_url:
        sources.append({
            "label": row.get("platform") or "Xem phim / trailer",
            "url": trailer_url,
            "kind": "watch",
        })

    return {
        "id": -db_id,
        "custom_movie_id": db_id,
        "title": row.get("title") or "Untitled",
        "original_title": row.get("original_title") or "",
        "original_language": row.get("original_language") or "vi",
        "overview": row.get("description") or "",
        "release_date": release_date,
        "year": release_date[:4] if release_date else "N/A",
        "runtime": max(1, round(duration_seconds / 60)) if duration_seconds else 0,
        "vote_average": float(row.get("vote_average") or 0.0),
        "vote_count": int(row.get("vote_count") or 0),
        "popularity": 1000.0 if row.get("status") == "published" else 0.0,
        "poster_path": "",
        "poster_url": _safe_url(row.get("thumbnail_url")),
        "backdrop_path": "",
        "backdrop_url": _safe_url(row.get("backdrop_url")) or _safe_url(row.get("thumbnail_url")),
        "genres": _split(row.get("genre_names")),
        "director": row.get("author_name") or "",
        "cast": _json_list(row.get("cast_json")),
        "keywords": _split(row.get("topic_names")),
        "tagline": row.get("source_label") or "CineMatch Exclusive",
        "trailer_url": trailer_url,
        "source_url": source_url,
        "source_label": row.get("source_label") or "CineMatch Exclusive",
        "platform": row.get("platform") or "",
        "ai_tool": row.get("ai_tool") or "",
        "sources": sources,
        "watch_providers": [],
        "watch_region": "",
        "content_type": content_type,
        "is_ai_generated": content_type == "ai_short",
        "is_custom": True,
        "is_published": row.get("status") == "published",
        "created_at": str(row.get("created_at") or ""),
        "updated_at": str(row.get("updated_at") or ""),
    }


SELECT_MOVIES = f"""
    SELECT v.*,
           GROUP_CONCAT(DISTINCT g.name ORDER BY g.name SEPARATOR '{SEPARATOR}') AS genre_names,
           GROUP_CONCAT(DISTINCT t.name ORDER BY t.name SEPARATOR '{SEPARATOR}') AS topic_names
    FROM ai_videos v
    LEFT JOIN ai_video_genres vg ON vg.video_id = v.id
    LEFT JOIN genres g ON g.id = vg.genre_id
    LEFT JOIN ai_video_topics vt ON vt.video_id = v.id
    LEFT JOIN topics t ON t.id = vt.topic_id
"""


def list_movies(*, published_only: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
    query = SELECT_MOVIES
    params: List[Any] = []
    if published_only:
        query += " WHERE v.status = 'published'"
    query += " GROUP BY v.id ORDER BY v.created_at DESC LIMIT %s"
    params.append(max(1, min(int(limit), 500)))
    with database.cursor(dictionary=True) as db_cursor:
        db_cursor.execute(query, params)
        return [to_movie(row) for row in db_cursor.fetchall()]


def safe_list_published(limit: int = 20) -> List[Dict[str, Any]]:
    try:
        return list_movies(published_only=True, limit=limit)
    except Exception:
        return []


def get_movie(movie_id: int, *, include_unpublished: bool = False) -> Optional[Dict[str, Any]]:
    query = SELECT_MOVIES + " WHERE v.id = %s"
    params: List[Any] = [abs(int(movie_id))]
    if not include_unpublished:
        query += " AND v.status = 'published'"
    query += " GROUP BY v.id"
    with database.cursor(dictionary=True) as db_cursor:
        db_cursor.execute(query, params)
        row = db_cursor.fetchone()
        return to_movie(row) if row else None


def safe_get_movie(movie_id: int) -> Optional[Dict[str, Any]]:
    try:
        return get_movie(movie_id)
    except Exception:
        return None


def search_published(query: str = "", genres: Optional[List[str]] = None, limit: int = 20) -> List[Dict[str, Any]]:
    movies = safe_list_published(limit=200)
    needle = query.strip().casefold()
    selected_genres = {genre.casefold() for genre in (genres or [])}
    results = []
    for movie in movies:
        haystack = " ".join([
            movie["title"], movie["original_title"], movie["overview"],
            " ".join(movie["keywords"]), movie["director"], " ".join(movie["cast"]),
        ]).casefold()
        movie_genres = {genre.casefold() for genre in movie["genres"]}
        if needle and needle not in haystack:
            continue
        if selected_genres and not selected_genres.intersection(movie_genres):
            continue
        results.append(movie)
        if len(results) >= limit:
            break
    return results


def prepare_payload(form: Dict[str, Any]) -> Dict[str, Any]:
    release_date = str(form.get("release_date") or "").strip() or None
    runtime = max(0, int(form.get("runtime") or 0))
    is_ai = bool(form.get("is_ai_generated"))
    video_url = _safe_url(form.get("trailer_url"))
    platform = str(form.get("platform") or "").strip()
    if not platform and "youtube" in video_url.casefold():
        platform = "YouTube"
    return {
        "title": str(form.get("title") or "").strip(),
        "description": str(form.get("overview") or "").strip(),
        "video_url": video_url,
        "platform": platform or None,
        "thumbnail_url": _safe_url(form.get("poster_url")) or None,
        "duration_seconds": runtime * 60 or None,
        "author_name": str(form.get("director") or "").strip() or None,
        "source_url": _safe_url(form.get("source_url")) or None,
        "ai_tool": str(form.get("ai_tool") or "").strip() or None,
        "status": "published" if form.get("is_published") else "draft",
        "content_type": "ai_short" if is_ai else "new_movie",
        "original_title": str(form.get("original_title") or "").strip(),
        "release_date": release_date,
        "original_language": str(form.get("original_language") or "vi").strip()[:12],
        "backdrop_url": _safe_url(form.get("backdrop_url")),
        "cast_json": json.dumps(_form_list(form.get("cast")), ensure_ascii=False),
        "vote_average": max(0.0, min(10.0, float(form.get("vote_average") or 0.0))),
        "vote_count": max(0, int(form.get("vote_count") or 0)),
        "source_label": str(form.get("source_label") or "CineMatch Exclusive").strip()[:120],
        "genres": _form_list(form.get("genres")),
        "topics": _form_list(form.get("keywords")),
    }


MOVIE_COLUMNS = (
    "title", "description", "video_url", "platform", "thumbnail_url", "duration_seconds",
    "author_name", "source_url", "ai_tool", "status", "content_type", "original_title",
    "release_date", "original_language", "backdrop_url", "cast_json", "vote_average",
    "vote_count", "source_label",
)


def _sync_names(db_cursor: Any, movie_id: int, names: Sequence[str], *, lookup_table: str, junction_table: str, foreign_key: str) -> None:
    db_cursor.execute(f"DELETE FROM {junction_table} WHERE video_id = %s", (movie_id,))
    for name in names:
        db_cursor.execute(
            f"INSERT INTO {lookup_table} (name) VALUES (%s) ON DUPLICATE KEY UPDATE name = VALUES(name)",
            (name,),
        )
        db_cursor.execute(f"SELECT id FROM {lookup_table} WHERE name = %s", (name,))
        lookup_id = int(db_cursor.fetchone()[0])
        db_cursor.execute(
            f"INSERT IGNORE INTO {junction_table} (video_id, {foreign_key}) VALUES (%s, %s)",
            (movie_id, lookup_id),
        )


def _sync_taxonomy(db_cursor: Any, movie_id: int, payload: Dict[str, Any]) -> None:
    _sync_names(db_cursor, movie_id, payload["genres"], lookup_table="genres", junction_table="ai_video_genres", foreign_key="genre_id")
    _sync_names(db_cursor, movie_id, payload["topics"], lookup_table="topics", junction_table="ai_video_topics", foreign_key="topic_id")


def create_movie(form: Dict[str, Any], admin_id: int) -> int:
    payload = prepare_payload(form)
    columns = ", ".join(MOVIE_COLUMNS) + ", created_by"
    placeholders = ", ".join(["%s"] * (len(MOVIE_COLUMNS) + 1))
    values = [payload[column] for column in MOVIE_COLUMNS] + [admin_id]
    with database.cursor(commit=True) as db_cursor:
        db_cursor.execute(f"INSERT INTO ai_videos ({columns}) VALUES ({placeholders})", values)
        movie_id = int(db_cursor.lastrowid)
        _sync_taxonomy(db_cursor, movie_id, payload)
        return movie_id


def update_movie(movie_id: int, form: Dict[str, Any]) -> None:
    payload = prepare_payload(form)
    assignments = ", ".join(f"{column} = %s" for column in MOVIE_COLUMNS)
    db_id = abs(int(movie_id))
    values = [payload[column] for column in MOVIE_COLUMNS] + [db_id]
    with database.cursor(commit=True) as db_cursor:
        db_cursor.execute(f"UPDATE ai_videos SET {assignments} WHERE id = %s", values)
        _sync_taxonomy(db_cursor, db_id, payload)


def delete_movie(movie_id: int) -> None:
    with database.cursor(commit=True) as db_cursor:
        db_cursor.execute("DELETE FROM ai_videos WHERE id = %s", (abs(int(movie_id)),))


def set_published(movie_id: int, published: bool) -> None:
    with database.cursor(commit=True) as db_cursor:
        db_cursor.execute(
            "UPDATE ai_videos SET status = %s WHERE id = %s",
            ("published" if published else "draft", abs(int(movie_id))),
        )
