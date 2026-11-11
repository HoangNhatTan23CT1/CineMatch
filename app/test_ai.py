"""
@file test_ai.py
@module test_ai
@description Kiểm thử tích hợp AI: ONNX MiniLM Encoder, mã hóa thời gian thực và tìm kiếm ngữ nghĩa đa ngôn ngữ
"""
import unittest
from unittest.mock import patch
import numpy as np

from app import create_app
from recommender import MovieRecommender
from semantic import (
    AIUnavailable,
    Encoder,
    get_encoder,
    compute_candidate_semantic_scores,
    compute_query_semantic_scores,
    movie_to_text,
    fingerprint,
)
import tmdb_service


class AITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = MovieRecommender()
        cls.client = create_app(cls.db).test_client()

    def test_encoder_initialization_and_vector_shape(self):
        """Kiểm tra khởi tạo Encoder và sinh vector 384 chiều chuẩn hóa L2."""
        encoder = get_encoder()
        self.assertIsInstance(encoder, Encoder)
        vectors = encoder.encode(["Phim ve vu tru va ho den", "Space exploration movie"])
        self.assertEqual(vectors.shape, (2, 384))
        # Kiểm tra chuẩn hóa L2 norm ≈ 1.0
        norms = np.linalg.norm(vectors, axis=1)
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=4)

    def test_multilingual_semantic_retrieval(self):
        """Kiểm tra truy vấn tiếng Việt khớp chính xác phim tương ứng trong danh mục."""
        # 1. Truy vấn tiếng Việt về đồ chơi
        res_toy, total_toy, _, _ = self.db.search_ai("phim hoạt hình về những món đồ chơi biết nói")
        self.assertGreater(total_toy, 0)
        toy_titles = [m['title'] for m in res_toy[:3]]
        self.assertTrue(any("Toy Story" in t for t in toy_titles))
        self.assertGreater(res_toy[0]['ai_score'], 0.4)

        # 2. Truy vấn tiếng Việt về lỗ sâu và vũ trụ
        res_space, total_space, _, _ = self.db.search_ai("phim về du hành không gian và lỗ sâu vũ trụ")
        self.assertGreater(total_space, 0)
        top_titles = [m['title'] for m in res_space[:3]]
        self.assertIn("Interstellar", top_titles)

    def test_candidate_semantic_scoring(self):
        """Kiểm tra tính điểm ngữ nghĩa giữa phim gốc và các ứng viên."""
        target = tmdb_service.FALLBACK_MOVIES[0]  # Interstellar
        candidates = tmdb_service.FALLBACK_MOVIES[1:]
        scores = compute_candidate_semantic_scores(target, candidates)
        self.assertEqual(len(scores), len(candidates))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_movie_to_text_structure(self):
        """Kiểm tra chuyển đổi phim sang cấu trúc văn bản giàu ngữ cảnh."""
        movie = {
            'title': 'Test Movie',
            'overview': 'A hero saves the day.',
            'genres': ['Action', 'Sci-Fi'],
            'keywords': ['space', 'future'],
            'director': 'John Doe',
        }
        text = movie_to_text(movie)
        self.assertIn('Test Movie', text)
        self.assertIn('A hero saves the day', text)
        self.assertIn('Genres: Action, Sci-Fi', text)
        self.assertIn('Director: John Doe', text)

    def test_fingerprint_generation(self):
        """Kiểm tra tính toán fingerprint nhất quán."""
        movies = tmdb_service.FALLBACK_MOVIES[:3]
        fp1 = fingerprint(movies)
        fp2 = fingerprint(movies)
        self.assertEqual(fp1, fp2)
        # Thay đổi dữ liệu phải tạo fingerprint khác
        modified = [dict(m) for m in movies]
        modified[0]['overview'] = 'Totally different overview content'
        fp3 = fingerprint(modified)
        self.assertNotEqual(fp1, fp3)

    def test_ai_unavailable_graceful_fallback(self):
        """Kiểm tra fallback an toàn khi AI tạm thời không khả dụng."""
        with patch('recommender.compute_candidate_semantic_scores', side_effect=AIUnavailable('AI offline')):
            recs = self.db.recommend(157336, method='hybrid')
            self.assertGreater(len(recs), 0)
            self.assertIn('similarity', recs[0])


if __name__ == '__main__':
    unittest.main()
