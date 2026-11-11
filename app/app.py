"""
@file app.py
@module app
@description Controller Flask quản lý các route Web SSR, Dynamic Cinema Explorer, tìm kiếm kép và Two-Tier Hybrid Re-ranking
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import secrets
from typing import Optional, Dict, Any, List
from flask import Flask, abort, redirect, render_template, request, jsonify

from admin import admin_bp
import custom_movie_service
import database
import journey_service
from recommender import MovieRecommender
from semantic import AIUnavailable
import tmdb_service


@lru_cache(maxsize=1)
def get_model() -> MovieRecommender:
    """Khởi tạo và lưu cache singleton của MovieRecommender."""
    return MovieRecommender()


def create_app(model: Optional[MovieRecommender] = None) -> Flask:
    """Factory khởi tạo Flask Application."""
    app = Flask(__name__)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SECRET_KEY'] = database.get_env_value('SECRET_KEY') or secrets.token_hex(32)
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.register_blueprint(admin_bp)

    @app.after_request
    def set_cache_headers(response):
        """Bổ sung Cache-Control tối ưu cho tài nguyên tĩnh."""
        if request.path.startswith('/static/'):
            response.headers['Cache-Control'] = 'public, max-age=86400'
        return response

    def catalog() -> MovieRecommender:
        return model if model is not None else get_model()

    def get_chosen_method() -> str:
        raw = request.args.get('method', 'hybrid').lower()
        return raw if raw in ('hybrid', 'ai', 'tfidf') else 'hybrid'

    def get_recommendations_with_fallback(db: MovieRecommender, movie_id: int, genres=None, topics=None):
        chosen = get_chosen_method()
        try:
            return db.recommend(movie_id, genres=genres, topics=topics, method=chosen), chosen, ''
        except AIUnavailable as exc:
            app.logger.warning('[AI] AI unavailable: %s', exc)
            return db.recommend(movie_id, genres=genres, topics=topics, method='tfidf'), 'tfidf', f'{exc} Đang dùng gợi ý TF-IDF dự phòng.'
        except Exception as exc:
            app.logger.error('[Recommender] Lỗi gợi ý: %s', exc)
            return [], chosen, 'Không thể tải danh sách gợi ý lúc này.'

    @app.context_processor
    def inject_global_context():
        """Bổ sung context toàn cục cho mọi template."""
        return {
            'has_api_key': tmdb_service.has_valid_api_key(),
            'active_api_key': tmdb_service.get_active_api_key() or '',
        }

    @app.get('/')
    def home():
        query = request.args.get('q', '').strip()[:200]
        genres = list(dict.fromkeys(g for g in request.args.getlist('genre') if g))
        topics = list(dict.fromkeys(t for t in request.args.getlist('topic') if t))
        requested_mode = request.args.get('mode', 'title')
        mode = requested_mode if requested_mode in ('title', 'ai', 'mood') else 'title'
        selected_mood = request.args.get('mood', '').strip()[:40]
        mood_profile = None
        sort = request.args.get('sort', 'popular')
        if sort not in ('popular', 'rating', 'year'):
            sort = 'popular'
        page = request.args.get('page', 1, type=int)

        db = catalog()
        warning = ''
        chosen_method = get_chosen_method()

        # Kiểm tra xem có đang ở chế độ xem khám phá mặc định không
        is_explorer_view = (
            not query and not genres and not topics and not selected_mood
            and mode == 'title' and page == 1
        )
        trending_today = []
        top_rated = []
        now_playing = []
        new_releases = []

        if is_explorer_view:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_trending = executor.submit(tmdb_service.get_trending_movies, time_window='day', page=1)
                fut_top = executor.submit(tmdb_service.get_top_rated_movies, page=1)
                fut_now = executor.submit(tmdb_service.get_now_playing_movies, page=1)
                trending_today, _, _ = fut_trending.result()
                top_rated, _, _ = fut_top.result()
                now_playing, _, _ = fut_now.result()
            new_releases = custom_movie_service.safe_list_published(limit=8)

        # Xử lý tìm kiếm hoặc phân trang
        if mode == 'mood':
            movies, total, page, pages, mood_profile = db.search_by_mood(
                selected_mood, context=query, page=page
            )
            selected_mood = mood_profile['key']
        elif mode == 'ai' and query:
            try:
                movies, total, page, pages = db.search_ai(query, genres, topics, page)
            except AIUnavailable as exc:
                warning = str(exc)
                movies, total, page, pages = [], 0, 1, 1
        else:
            movies, total, page, pages = db.search(query, genres, sort, page, topics=topics)

        # Lấy phim mẫu và danh sách gợi ý nếu có tìm kiếm hoặc chọn phim
        matches = db.title_matches(query) if (mode == 'title' and query) else []
        source_id = request.args.get('source', type=int)

        source = None
        if source_id:
            source = db.get_movie(source_id)
        elif matches:
            source = matches[0]

        similar = []
        if source:
            similar, chosen_method, warning = get_recommendations_with_fallback(
                db, source['id'], genres=genres, topics=topics
            )

        return render_template(
            'index.html',
            movies=movies,
            total=total,
            page=page,
            pages=pages,
            genres=db.genres,
            topics=db.topics,
            selected_genres=genres,
            selected_topics=topics,
            selected_mood=selected_mood,
            mood_profile=mood_profile,
            moods=getattr(db, 'moods', []),
            query=query,
            sort=sort,
            mode=mode,
            method=chosen_method,
            warning=warning,
            matches=matches,
            source=source,
            similar=similar,
            is_explorer_view=is_explorer_view,
            trending_today=trending_today[:8],
            top_rated=top_rated[:8],
            now_playing=now_playing[:8],
            new_releases=new_releases[:8],
        )

    @app.get('/journey')
    def mood_journey():
        """Tạo đơn phim dẫn người dùng từ cảm xúc hiện tại tới cảm xúc mong muốn."""
        options = journey_service.journey_form_options()
        current_mood = request.args.get('current_mood', 'sad').strip()[:40]
        target_mood = request.args.get('target_mood', 'calm').strip()[:40]
        company = request.args.get('company', 'alone').strip()[:40]
        intensity = request.args.get('intensity', 'gentle').strip()[:40]
        duration = request.args.get('duration', 120, type=int) or 120
        submitted = request.args.get('build') == '1'
        journey = None
        warning = ''

        if submitted:
            try:
                journey = journey_service.build_journey(
                    catalog(), current_mood, target_mood, duration, company, intensity
                )
                current_mood = journey['current']['key']
                target_mood = journey['target']['key']
                company = journey['company']['key']
                intensity = journey['intensity']['key']
                duration = journey['duration']
                if not journey['movies']:
                    warning = 'Chưa tìm được tác phẩm phù hợp. Hãy thử một điểm đến cảm xúc khác.'
            except Exception as exc:
                app.logger.error('[MoodJourney] Không thể tạo hành trình: %s', exc)
                warning = 'CineMatch chưa thể tạo hành trình lúc này. Vui lòng thử lại sau ít phút.'

        return render_template(
            'journey.html',
            journey=journey,
            warning=warning,
            submitted=submitted,
            current_mood=current_mood,
            target_mood=target_mood,
            company=company,
            intensity=intensity,
            duration=duration,
            **options,
        )

    @app.get('/movie/<int(signed=True):movie_id>')
    def detail(movie_id: int):
        db = catalog()
        movie = db.get_movie(movie_id)
        if movie is None:
            abort(404)
        items, chosen_method, warning = get_recommendations_with_fallback(db, movie_id)
        return render_template(
            'index.html',
            movie=movie,
            recommendations=items,
            method=chosen_method,
            warning=warning,
            genres=db.genres,
            topics=db.topics,
        )

    @app.post('/settings/api-key')
    def update_api_key():
        """Cập nhật TMDB API Key từ Web UI."""
        api_key = request.form.get('api_key', '').strip()
        if api_key:
            tmdb_service.set_active_api_key(api_key, persist_to_env=True)
        next_url = request.form.get('redirect_to') or request.referrer or '/'
        return redirect(next_url)

    @app.get('/api/recommendations/<int(signed=True):movie_id>')
    def api_recommendations(movie_id: int):
        """API trả về danh sách gợi ý dạng JSON cho client."""
        db = catalog()
        chosen = get_chosen_method()
        recs = db.recommend(movie_id, method=chosen)
        return jsonify({
            'success': True,
            'movie_id': movie_id,
            'method': chosen,
            'count': len(recs),
            'recommendations': recs,
        })

    @app.errorhandler(404)
    def not_found(error):
        db = catalog()
        return render_template(
            'index.html',
            error='Không tìm thấy phim hoặc trang này.',
            genres=getattr(db, 'genres', []),
            topics=getattr(db, 'topics', {}),
        ), 404

    return app


app = create_app()

if __name__ == '__main__':
    from waitress import serve
    print('[App] Khoi dong CineMatch Web Application tren http://127.0.0.1:5000...', flush=True)
    get_model()
    serve(app, host='127.0.0.1', port=5000)
