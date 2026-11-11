"""Kiểm thử dữ liệu và giao diện quản trị CineMatch Studio."""
import unittest
from unittest.mock import MagicMock, patch

import custom_movie_service
import database
import tmdb_service
from app import create_app


class AdminFeatureTests(unittest.TestCase):
    def test_existing_ai_video_maps_to_public_movie(self):
        movie = custom_movie_service.to_movie({
            "id": 7,
            "title": "Giấc mơ máy",
            "description": "Một phim ngắn về trí tuệ nhân tạo.",
            "video_url": "https://example.com/watch",
            "thumbnail_url": "https://example.com/poster.jpg",
            "duration_seconds": 540,
            "author_name": "CineMatch Lab",
            "status": "published",
            "content_type": "ai_short",
            "genre_names": "Drama\x1fScience Fiction",
            "topic_names": "AI\x1frobot",
            "cast_json": '["An", "Bình"]',
        })
        self.assertEqual(movie["id"], -7)
        self.assertEqual(movie["runtime"], 9)
        self.assertTrue(movie["is_ai_generated"])
        self.assertTrue(movie["is_published"])
        self.assertIn("Science Fiction", movie["genres"])

    def test_form_payload_preserves_movie_metadata(self):
        payload = custom_movie_service.prepare_payload({
            "title": "Phim mới",
            "overview": "Mô tả",
            "runtime": "12",
            "genres": "Drama, AI",
            "keywords": "robot, tương lai",
            "cast": "A, B",
            "is_ai_generated": "1",
            "is_published": "1",
            "vote_average": "8.5",
        })
        self.assertEqual(payload["duration_seconds"], 720)
        self.assertEqual(payload["content_type"], "ai_short")
        self.assertEqual(payload["status"], "published")
        self.assertEqual(payload["genres"], ["Drama", "AI"])
        self.assertEqual(payload["vote_average"], 8.5)

    def test_tmdb_detail_exposes_cast_sources_and_watch_links(self):
        movie = tmdb_service.format_movie_record({
            "id": 42,
            "title": "Nguồn phim",
            "credits": {
                "cast": [{"name": "Diễn viên A"}, {"name": "Diễn viên B"}],
                "crew": [{"job": "Director", "name": "Đạo diễn C"}],
            },
            "external_ids": {"imdb_id": "tt1234567"},
            "homepage": "https://example.com/movie",
            "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "abc123"}]},
            "watch/providers": {"results": {"VN": {
                "link": "https://example.com/watch",
                "flatrate": [{"provider_id": 1, "provider_name": "Cinema Plus", "logo_path": "/logo.png"}],
            }}},
        })
        self.assertEqual(movie["cast"], ["Diễn viên A", "Diễn viên B"])
        self.assertEqual(movie["director"], "Đạo diễn C")
        self.assertEqual(movie["watch_region"], "VN")
        self.assertEqual(movie["watch_providers"][0]["name"], "Cinema Plus")
        self.assertEqual({source["label"] for source in movie["sources"]}, {"TMDB", "IMDb", "Trang chính thức"})

    def test_admin_login_explains_missing_database_password(self):
        app = create_app()
        app.config.update(TESTING=True, SECRET_KEY="test-secret")
        with patch("admin.database.ensure_schema", side_effect=database.DatabaseUnavailable("Thiếu DB_PASSWORD")):
            response = app.test_client().get("/admin/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Thiếu DB_PASSWORD", response.get_data(as_text=True))

    def test_admin_templates_compile(self):
        app = create_app()
        with app.app_context():
            for template in (
                "admin/login.html", "admin/dashboard.html", "admin/movie_form.html", "admin/database_error.html"
            ):
                app.jinja_env.get_template(template)

    def test_custom_movie_detail_accepts_negative_ids(self):
        model = MagicMock()
        model.get_movie.return_value = {
            "id": -7, "title": "Phim AI", "overview": "Mô tả", "year": "2026",
            "runtime": 5, "vote_average": 0.0, "vote_count": 0, "genres": [],
            "cast": [], "director": "", "topics": [], "poster_url": "", "backdrop_url": "",
            "tagline": "CineMatch Exclusive", "trailer_url": "",
        }
        model.recommend.return_value = []
        model.genres = []
        model.topics = {}
        app = create_app(model)
        app.config.update(TESTING=True, SECRET_KEY="test-secret")
        response = app.test_client().get("/movie/-7")
        self.assertEqual(response.status_code, 200)
        model.get_movie.assert_called_once_with(-7)


if __name__ == "__main__":
    unittest.main()
