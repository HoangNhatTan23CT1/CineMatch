"""
@file test_app.py
@module test_app
@description Bộ kiểm thử đơn vị tự động cho CineMatch: TMDB service, Two-Tier Hybrid Re-ranking, Explainability và Web Routes
"""
import unittest
from unittest.mock import patch
from app import create_app
from recommender import MovieRecommender, resolve_mood_key
import journey_service
import tmdb_service


class CineMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MovieRecommender()
        cls.client = create_app(cls.model).test_client()

    def test_tmdb_fallback_catalog(self):
        """Kiểm tra danh mục fallback khi chưa có API key luôn có sẵn và đầy đủ metadata."""
        self.assertGreaterEqual(len(tmdb_service.FALLBACK_MOVIES), 10)
        first = tmdb_service.FALLBACK_MOVIES[0]
        self.assertIn('title', first)
        self.assertIn('overview', first)
        self.assertIn('genres', first)
        self.assertIn('poster_path', first)

    def test_search_case_insensitive(self):
        """Tìm kiếm tên phim không phân biệt hoa/thường."""
        results, total, _, _ = self.model.search('aVaTaR')
        self.assertGreater(total, 0)
        self.assertTrue(any('avatar' in m['title'].lower() for m in results))

    def test_hybrid_recommendations_and_explainability(self):
        """Kiểm tra gợi ý theo phương pháp Hybrid trả về đầy đủ các trường Explainability."""
        # Dùng Interstellar (157336)
        recs = self.model.recommend(157336, method='hybrid')
        self.assertGreater(len(recs), 0)
        self.assertNotIn(157336, [m['id'] for m in recs])
        first = recs[0]
        self.assertIn('similarity', first)
        self.assertIn('match_percentage', first)
        self.assertIn('breakdown', first)
        self.assertIn('reason', first)
        # Kiểm tra chi tiết cấu trúc breakdown
        breakdown = first['breakdown']
        self.assertIn('semantic_score', breakdown)
        self.assertIn('genre_score', breakdown)
        self.assertIn('quality_score', breakdown)
        self.assertIn('has_genre_overlap', breakdown)

    def test_recommendation_methods(self):
        """Kiểm tra 3 phương pháp gợi ý (hybrid, ai, tfidf)."""
        for method in ['hybrid', 'ai', 'tfidf']:
            with self.subTest(method=method):
                recs = self.model.recommend(157336, method=method)
                self.assertGreater(len(recs), 0)
                scores = [m['similarity'] for m in recs]
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_soft_penalty_mechanism(self):
        """Kiểm tra cơ chế phạt điểm khi hoàn toàn lệch thể loại."""
        recs = self.model.recommend(157336, method='hybrid')
        for r in recs:
            if not r['breakdown']['has_genre_overlap']:
                # Điểm phải bị giảm đáng kể do hệ số phạt 0.4x
                self.assertLessEqual(r['similarity'], 0.5)

    def test_filter_and_topics(self):
        """Kiểm tra lọc thể loại và chủ đề."""
        results, total, _, _ = self.model.search(genre='Science Fiction')
        self.assertGreater(total, 0)
        self.assertTrue(all('Science Fiction' in m['genres'] for m in results))

    def test_pagination_bounds(self):
        """Kiểm tra ranh giới phân trang."""
        _, _, page1, _ = self.model.search(page=-5)
        self.assertEqual(page1, 1)

    def test_web_routes(self):
        """Kiểm tra các route web chính trả về HTTP 200."""
        for path in ['/', '/?q=Avatar', '/?method=hybrid', '/?mode=ai&q=space', '/journey', '/movie/157336']:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

    def test_mood_aliases_support_vietnamese_without_accents(self):
        self.assertEqual(resolve_mood_key('Tôi đang buồn'), 'sad')
        self.assertEqual(resolve_mood_key('lo lang'), 'anxious')
        self.assertEqual(resolve_mood_key('cần động lực'), 'motivated')

    def test_mood_search_route_renders_selected_emotion(self):
        profile = {
            'key': 'sad', 'label': 'Buồn', 'emoji': '😢',
            'description': 'Những câu chuyện ấm áp giúp bạn được đồng cảm và chữa lành.',
        }
        movie = {
            'id': 42, 'title': 'Phim chữa lành', 'year': 2026, 'runtime': 100,
            'vote_average': 8.0, 'vote_count': 500, 'genres': ['Drama'],
            'poster_url': '', 'overview': 'Một câu chuyện ấm áp.', 'ai_score': 0.88,
        }
        with patch.object(
            self.model, 'search_by_mood', return_value=([movie], 1, 1, 1, profile)
        ) as search_mock:
            response = self.client.get('/?mode=mood&mood=sad&q=nhẹ+nhàng')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Phim dành cho lúc', html)
        self.assertIn('Phim chữa lành', html)
        self.assertIn('AI Mood Matching', html)
        search_mock.assert_called_once_with('sad', context='nhẹ nhàng', page=1)

    def test_emotion_journey_respects_time_and_builds_stages(self):
        base_movies = [
            {
                'id': 101, 'title': 'Khởi đầu', 'runtime': 100, 'genres': ['Drama'],
                'vote_average': 8.0, 'vote_count': 1200, 'ai_score': 0.75,
                'overview': 'Một câu chuyện chữa lành.', 'year': 2024,
            },
            {
                'id': 102, 'title': 'Điểm sáng', 'runtime': 115, 'genres': ['Comedy', 'Family'],
                'vote_average': 7.8, 'vote_count': 900, 'ai_score': 0.72,
                'overview': 'Một hành trình nhiều hy vọng.', 'year': 2025,
            },
            {
                'id': 103, 'title': 'Quá dài', 'runtime': 190, 'genres': ['Drama'],
                'vote_average': 8.5, 'vote_count': 400, 'ai_score': 0.78,
                'overview': 'Một thiên sử thi.', 'year': 2023,
            },
        ]
        model = unittest.mock.MagicMock()
        model.search_by_mood.return_value = (base_movies, 3, 1, 1, {'key': 'sad'})

        def movie_detail(movie_id):
            movie = next(item for item in base_movies if item['id'] == movie_id)
            return dict(movie, cast=['A', 'B'], sources=[], watch_providers=[], trailer_url='')

        with patch('journey_service.tmdb_service.get_movie_detail', side_effect=movie_detail), \
             patch('journey_service.compute_query_semantic_scores', return_value=[0.55, 0.9, 0.6]):
            journey = journey_service.build_journey(
                model, 'buồn', 'calm', 240, 'family', 'gentle'
            )

        self.assertEqual(len(journey['movies']), 2)
        self.assertLessEqual(journey['total_duration'], 255)
        self.assertEqual(journey['movies'][0]['journey_stage'], '1 · Đồng cảm')
        self.assertEqual(journey['movies'][1]['journey_stage'], '2 · Chuyển biến')
        self.assertEqual(journey['target']['key'], 'calm')

    def test_emotion_journey_route_renders_prescription(self):
        fake_journey = {
            'current': {'key': 'sad', 'label': 'Buồn', 'emoji': '😢'},
            'target': {'key': 'calm', 'label': 'Nhẹ lòng', 'emoji': '🌤️'},
            'company': {'key': 'alone', 'label': 'Một mình', 'emoji': '🎧'},
            'intensity': {'key': 'gentle', 'label': 'Êm dịu', 'emoji': '🫧'},
            'duration': 120, 'total_duration': 100, 'remaining_duration': 20,
            'headline': 'Từ buồn đến nhẹ lòng', 'summary': 'Một hành trình riêng.',
            'movies': [{
                'id': 42, 'title': 'Phim chữa lành', 'year': 2026, 'runtime': 100,
                'vote_average': 8.0, 'vote_count': 500, 'genres': ['Drama'],
                'poster_url': '', 'overview': 'Một câu chuyện ấm áp.', 'cast': ['A'],
                'content_notes': ['Cảm xúc sâu'], 'journey_match': 88,
                'journey_stage': 'Tác phẩm phù hợp nhất', 'journey_reason': 'Phù hợp.',
                'trailer_url': '', 'sources': [],
            }],
        }
        with patch('app.journey_service.build_journey', return_value=fake_journey):
            response = self.client.get(
                '/journey?build=1&current_mood=sad&target_mood=calm&duration=120&company=alone&intensity=gentle'
            )
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('ĐƠN PHIM TỐI NAY', html)
        self.assertIn('Từ buồn đến nhẹ lòng', html)
        self.assertIn('Phim chữa lành', html)

    def test_api_recommendations_endpoint(self):
        """Kiểm tra JSON API gợi ý /api/recommendations/<id>."""
        res = self.client.get('/api/recommendations/157336?method=hybrid')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['method'], 'hybrid')
        self.assertGreater(data['count'], 0)
        self.assertIn('breakdown', data['recommendations'][0])

    def test_not_found(self):
        """Kiểm tra trang 404 an toàn."""
        self.assertEqual(self.client.get('/movie/999999999').status_code, 404)
        self.assertEqual(self.client.get('/nonexistent-page').status_code, 404)

    def test_natural_language_director_search(self):
        """Kiểm tra tìm kiếm tự nhiên theo đạo diễn (ví dụ: Phim việt nam của trấn thành đạo diễn)."""
        movies, total, _, _ = self.model.search("Phim việt nam của trấn thành đạo diễn")
        self.assertGreater(total, 0)
        titles = [m['title'] for m in movies]
        self.assertTrue(any("Mai" in t or "Bố Già" in t or "Nhà Bà Nữ" in t for t in titles))

    def test_xss_protection(self):
        """Kiểm tra lọc mã độc XSS trong query tìm kiếm."""
        response = self.client.get('/', query_string={'q': '<script>alert(1)</script>'})
        html = response.get_data(as_text=True)
        self.assertNotIn('<script>alert(1)</script>', html)


if __name__ == '__main__':
    unittest.main()
