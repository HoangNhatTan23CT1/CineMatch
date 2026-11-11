"""
@file recommender.py
@module recommender
@description Động cơ Gợi ý Two-Tier Real-Time Hybrid Re-ranking kết hợp TMDB API, ONNX Semantic Embedding và Bayesian Rating
"""
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tmdb_service
import custom_movie_service
from semantic import (
    AIUnavailable,
    compute_candidate_semantic_scores,
    compute_query_semantic_scores,
    movie_to_text,
)

# Bảng mở rộng khái niệm và thực thể ngữ nghĩa từ câu thoại / văn hóa đại chúng
CONCEPT_EXPANSIONS: Dict[Tuple[str, ...], List[str]] = {
    ('vua hải tặc', 'hải tặc', 'đảo hải tặc', 'cướp biển', 'pirate', 'luffy', 'one piece'): [
        'one piece', 'hải tặc', 'cướp biển', 'pirates of the caribbean', 'pirate'
    ],
    ('người nhện', 'nhện', 'spider man', 'spider-man', 'spiderman', 'peter parker'): [
        'spider-man', 'người nhện', 'spiderman'
    ],
    ('người dơi', 'kỵ sĩ bóng đêm', 'batman', 'dark knight', 'gotham'): [
        'batman', 'the dark knight', 'người dơi'
    ],
    ('lỗ sâu', 'hố đen', 'du hành vũ trụ', 'du hành không gian', 'du hành các vì sao', 'thiên hà', 'wormhole', 'black hole', 'interstellar'): [
        'interstellar', 'space travel', 'vũ trụ', 'star wars'
    ],
    ('đồ chơi', 'toy story', 'woody', 'buzz lightyear'): [
        'toy story', 'đồ chơi'
    ],
    ('sát thủ', 'trả thù con chó', 'john wick', 'hitman', 'assassin'): [
        'john wick', 'sát thủ', 'assassin'
    ],
    ('tàu đắm', 'chìm tàu', 'băng trôi', 'titanic', 'jack và rose'): [
        'titanic', 'tàu đắm', 'shipwreck'
    ],
    ('phù thủy', 'phép thuật', 'trường hogwarts', 'harry potter', 'vết sẹo tia chớp'): [
        'harry potter', 'phù thủy', 'magic', 'wizard'
    ],
    ('khủng long', 'công viên kỷ jura', 'jurassic', 'dinosaur'): [
        'jurassic', 'khủng long', 'dinosaur'
    ],
    ('ma cà rồng', 'hút máu', 'vampire', 'dracula', 'twilight'): [
        'vampire', 'dracula', 'twilight', 'ma cà rồng'
    ],
    ('zombie', 'xác sống', 'thây ma', 'đại dịch zombie', 'train to busan'): [
        'zombie', 'xác sống', 'train to busan'
    ],
    ('bố già', 'mafia', 'ông trùm', 'godfather', 'corleone'): [
        'godfather', 'mafia', 'bố già'
    ],
    ('giấc mơ', 'thâm nhập giấc mơ', 'tầng bậc giấc mơ', 'inception'): [
        'inception', 'giấc mơ', 'dream'
    ],
    ('pandora', 'người da xanh', 'navi', 'avatar'): [
        'avatar', 'pandora'
    ],
    ('chúa nhẫn', 'nhẫn chúa', 'chúa tể những chiếc nhẫn', 'lord of the rings', 'frodo'): [
        'lord of the rings', 'chúa nhẫn'
    ],
    ('đánh trống', 'tay trống', 'nhạc jazz', 'whiplash'): [
        'whiplash', 'tay trống'
    ],
    ('cá hề', 'tìm cá', 'nemo', 'đại dương'): [
        'finding nemo', 'nemo', 'cá hề'
    ],
    ('siêu anh hùng', 'avengers', 'biệt đội siêu anh hùng', 'iron man', 'người sắt', 'thanos', 'marvel', 'superhero'): [
        'avengers', 'superhero', 'iron man', 'marvel'
    ],
}


# Bảng liên kết ý định tác phẩm đặc trưng (Cultural / Franchise Intent)
FRANCHISE_INTENTS: Dict[Tuple[str, ...], str] = {
    ('vua hải tặc', 'đảo hải tặc', 'luffy'): 'one piece',
    ('người nhện', 'spiderman'): 'spider-man',
    ('người dơi', 'kỵ sĩ bóng đêm'): 'batman',
    ('lỗ sâu', 'hố đen', 'du hành vũ trụ'): 'interstellar',
    ('đồ chơi', 'buzz lightyear', 'woody'): 'toy story',
    ('sát thủ', 'trả thù con chó'): 'john wick',
    ('vết sẹo tia chớp', 'hogwarts'): 'harry potter',
    ('chúa nhẫn', 'nhẫn chúa'): 'lord of the rings',
}


def extract_semantic_search_terms(query: str) -> List[str]:
    """
    Trích xuất và mở rộng thực thể, từ khóa ngữ nghĩa từ câu truy vấn tự nhiên.
    Lọc bỏ các từ đệm, từ nối tiếng Việt / Anh và đối chiếu bảng mở rộng khái niệm.
    """
    q_lower = query.strip().lower()
    if not q_lower:
        return []

    terms: List[str] = []
    for keys, exps in CONCEPT_EXPANSIONS.items():
        if any(k in q_lower for k in keys):
            terms.extend(exps)

    # Loại bỏ các từ đệm/ngữ cảnh hội thoại thông dụng trong tiếng Việt
    stop_phrases_pattern = (
        r'\b(phim có|phim về|bộ phim về|bộ phim có|bộ phim|tìm phim|xem phim|phim|'
        r'với một đứa|với một người|với đứa|với một|với|'
        r'có một|kể về|nói về|'
        r'ước mơ sẽ trở thành|ước mơ trở thành|muốn trở thành|sẽ trở thành|trở thành|có ước mơ|ước mơ|'
        r'người mà|đứa mà|trong đó|nội dung về|nội dung)\b'
    )
    cleaned = re.sub(stop_phrases_pattern, ' ', q_lower)
    cleaned = re.sub(r'[\,\.\?\!\:\;\"\'\(\)\[\]\{\}\-\_]', ' ', cleaned)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
    if words:
        residual_phrase = ' '.join(words)
        if residual_phrase not in terms and len(residual_phrase) >= 3:
            terms.append(residual_phrase)

    # Bảo đảm không trùng lặp mà vẫn giữ thứ tự ưu tiên
    return list(dict.fromkeys(terms))

# Chủ đề bổ sung nhận diện qua từ khóa TMDB
TOPIC_KEYWORDS = {
    'Siêu anh hùng': {'superhero', 'super powers', 'marvel', 'dc comics', 'avengers'},
    'Zombie': {'zombie', 'zombie apocalypse', 'undead'},
    'Ma cà rồng': {'vampire', 'dracula'},
    'Du hành thời gian': {'time travel', 'time machine', 'time loop'},
    'Ngoài hành tinh': {'alien', 'alien invasion', 'alien planet', 'extraterrestrial'},
    'Không gian': {'space', 'space travel', 'spaceship', 'astronaut', 'outer space', 'wormhole'},
    'Trinh thám': {'detective', 'private detective', 'murder investigation', 'mystery'},
    'Điệp viên': {'spy', 'espionage', 'secret agent'},
    'Võ thuật': {'martial arts', 'kung fu', 'karate'},
    'Sinh tồn': {'survival', 'wilderness', 'stranded'},
    'Hậu tận thế': {'post-apocalyptic', 'apocalypse', 'dystopia'},
    'Cướp / trộm': {'heist', 'bank robbery', 'robbery'},
    'Thể thao': {'sport', 'sports', 'boxing', 'football', 'baseball', 'basketball'},
    'Học đường': {'high school', 'boarding school', 'high school student', 'college'},
}


# Hồ sơ cảm xúc dùng cho chế độ tìm phim theo tâm trạng. Mỗi hồ sơ mô tả
# loại trải nghiệm người xem cần ở thời điểm hiện tại, thay vì chỉ tìm phim
# có nội dung mang đúng cảm xúc tiêu cực đó.
MOOD_PROFILES: Dict[str, Dict[str, Any]] = {
    'happy': {
        'label': 'Vui vẻ', 'emoji': '😊',
        'aliases': ['vui', 'vui ve', 'hanh phuc', 'phan khoi', 'happy', 'cheerful'],
        'genres': ['Comedy', 'Animation', 'Family', 'Music'],
        'keywords': ['funny', 'friendship', 'celebration', 'feel-good', 'uplifting', 'happy ending'],
        'avoid_genres': ['Horror'],
        'prompt': 'phim vui vẻ hài hước tràn năng lượng tích cực, tình bạn, feel-good, uplifting và kết thúc hạnh phúc',
        'description': 'Thêm tiếng cười và giữ nguồn năng lượng tích cực.',
    },
    'sad': {
        'label': 'Buồn', 'emoji': '😢',
        'aliases': ['buon', 'tam trang', 'that tinh', 'sad', 'down', 'blue'],
        'genres': ['Drama', 'Romance', 'Family', 'Animation'],
        'keywords': ['healing', 'hope', 'family', 'friendship', 'second chance', 'comforting'],
        'avoid_keywords': ['chainsaw', 'assassin', 'serial killer', 'nightmare', 'torture'],
        'avoid_genres': ['Horror'],
        'prompt': 'phim cảm động ấm áp chữa lành, có hy vọng, tình thân, tình bạn và một hành trình vượt qua nỗi buồn',
        'description': 'Những câu chuyện ấm áp giúp bạn được đồng cảm và chữa lành.',
    },
    'anxious': {
        'label': 'Lo lắng', 'emoji': '😟',
        'aliases': ['lo lang', 'bat an', 'hoi hop', 'anxious', 'worried', 'nervous'],
        'genres': ['Comedy', 'Family', 'Animation', 'Romance'],
        'keywords': ['calm', 'cozy', 'gentle', 'hope', 'friendship', 'peaceful'],
        'avoid_keywords': ['murder', 'serial killer', 'war', 'torture', 'horror'],
        'avoid_genres': ['Horror', 'Thriller', 'Crime', 'War'],
        'prompt': 'phim nhẹ nhàng bình yên dễ xem, ít căng thẳng, không khí ấm cúng, hài hước và mang lại cảm giác an toàn',
        'description': 'Ưu tiên phim nhẹ nhàng, dễ xem để tâm trí được thư giãn.',
    },
    'stressed': {
        'label': 'Căng thẳng', 'emoji': '😣',
        'aliases': ['cang thang', 'ap luc', 'met moi', 'stress', 'stressed', 'exhausted'],
        'genres': ['Comedy', 'Animation', 'Adventure', 'Family'],
        'keywords': ['escape', 'fun', 'adventure', 'relaxing', 'friendship', 'feel-good'],
        'avoid_keywords': ['torture', 'serial killer', 'psychological horror'],
        'avoid_genres': ['Horror'],
        'prompt': 'phim giải trí cuốn hút giúp xả stress, hài hước, phiêu lưu vui nhộn, dễ xem và giàu năng lượng',
        'description': 'Tạm rời áp lực với một chuyến phiêu lưu dễ chịu và vui nhộn.',
    },
    'lonely': {
        'label': 'Cô đơn', 'emoji': '🥺',
        'aliases': ['co don', 'trong trai', 'lac long', 'lonely', 'alone'],
        'genres': ['Drama', 'Comedy', 'Romance', 'Family'],
        'keywords': ['friendship', 'community', 'belonging', 'family', 'connection', 'companion'],
        'prompt': 'phim về tình bạn, sự kết nối, gia đình và cảm giác thuộc về, ấm áp, đồng hành và đầy hy vọng',
        'description': 'Các câu chuyện về tình bạn và sự kết nối để bạn thấy được đồng hành.',
    },
    'romantic': {
        'label': 'Lãng mạn', 'emoji': '🥰',
        'aliases': ['lang man', 'yeu', 'dang yeu', 'romantic', 'in love'],
        'genres': ['Romance', 'Comedy', 'Drama'],
        'keywords': ['love', 'romance', 'relationship', 'first love', 'date', 'wedding'],
        'prompt': 'phim tình cảm lãng mạn, rung động, tình yêu chân thành, chemistry cuốn hút và khoảnh khắc ngọt ngào',
        'description': 'Những câu chuyện tình yêu giàu rung động và cảm xúc.',
    },
    'motivated': {
        'label': 'Cần động lực', 'emoji': '🔥',
        'aliases': ['dong luc', 'nan chi', 'mat dong luc', 'motivated', 'inspired', 'inspiration'],
        'genres': ['Drama', 'Adventure', 'History', 'Documentary'],
        'keywords': ['dream', 'success', 'overcome', 'inspiring', 'true story', 'perseverance'],
        'prompt': 'phim truyền cảm hứng về ước mơ, nghị lực, vượt khó, kiên trì, trưởng thành và chinh phục mục tiêu',
        'description': 'Tiếp thêm cảm hứng để bắt đầu lại và tiến về phía trước.',
    },
    'adventurous': {
        'label': 'Muốn phiêu lưu', 'emoji': '🤩',
        'aliases': ['phieu luu', 'hao huc', 'kham pha', 'adventurous', 'excited'],
        'genres': ['Adventure', 'Action', 'Fantasy', 'Science Fiction'],
        'keywords': ['journey', 'exploration', 'quest', 'discovery', 'epic adventure'],
        'prompt': 'phim phiêu lưu kỳ thú, khám phá thế giới mới, hành trình sử thi, giàu trí tưởng tượng và cảm giác phấn khích',
        'description': 'Bước vào một thế giới mới với hành trình giàu bất ngờ.',
    },
}


def _normalize_mood_text(value: str) -> str:
    normalized = unicodedata.normalize('NFD', str(value or '').strip().lower())
    ascii_like = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    ascii_like = ascii_like.replace('đ', 'd')
    return ' '.join(ascii_like.split())


def resolve_mood_key(value: str) -> Optional[str]:
    """Chuẩn hóa tên/biểu đạt tâm trạng tiếng Việt hoặc tiếng Anh thành khóa hồ sơ."""
    text = _normalize_mood_text(value)
    if not text:
        return None
    if text in MOOD_PROFILES:
        return text
    for key, profile in MOOD_PROFILES.items():
        if any(alias == text or (len(alias) >= 4 and alias in text) for alias in profile['aliases']):
            return key
    return None


def extract_movie_topics(movie: Dict[str, Any]) -> List[str]:
    """Trích xuất các chủ đề tiếng Việt từ keywords và overview của phim."""
    keywords = [k.lower() for k in movie.get('keywords', [])]
    overview = movie.get('overview', '').lower()
    found = []
    for topic, terms in TOPIC_KEYWORDS.items():
        if any(term in keywords or term in overview for term in terms):
            found.append(topic)
    return found


def calculate_bayesian_quality(vote_average: float, vote_count: int, c_mean: float = 6.9, m_min: int = 500) -> float:
    """
    Tính điểm uy tín chất lượng phim theo công thức Bayesian Weighted Rating (tiêu chuẩn IMDb):
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    Chuẩn hóa về khoảng [0.0, 1.0].
    """
    v = max(0, vote_count)
    r = max(0.0, min(10.0, vote_average))
    wr = (v / (v + m_min)) * r + (m_min / (v + m_min)) * c_mean
    # Chuẩn hóa từ thang điểm 3.0 -> 9.0 về 0.0 -> 1.0
    normalized = (wr - 3.5) / 5.5
    return max(0.0, min(1.0, float(normalized)))


def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Tính độ tương đồng Jaccard giữa hai tập hợp."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


class MovieRecommender:
    """
    Động cơ Gợi ý và Tìm kiếm Phim Hiện đại:
    - Sử dụng TMDB Live API (không phụ thuộc CSV tĩnh)
    - Two-Tier Real-Time Hybrid Re-ranking (ONNX MiniLM + TF-IDF Lexical + Jaccard Metadata + Bayesian Rating)
    - Khử nhiễu với Soft-penalty Thể loại
    - Sinh phân rã giải thích Explainability
    """

    def __init__(self):
        print("[Recommender][Init] Initializing MovieRecommender with TMDB Live Service...", flush=True)
        self._refresh_genres()

    def _refresh_genres(self):
        genre_list = tmdb_service.get_genres()
        self.genres = sorted([g["name"] for g in genre_list if g.get("name")])
        self.topics = {label: 10 for label in TOPIC_KEYWORDS}
        self.moods = [dict({'key': key}, **profile) for key, profile in MOOD_PROFILES.items()]

    def get_movie(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết đầy đủ của phim từ TMDB API."""
        movie = custom_movie_service.safe_get_movie(movie_id) if movie_id < 0 else tmdb_service.get_movie_detail(movie_id)
        if movie:
            movie['topics'] = extract_movie_topics(movie)
        return movie

    def search(
        self,
        query: str = '',
        genre: Any = '',
        sort: str = 'popular',
        page: int = 1,
        page_size: int = 12,
        topics: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        """Tìm kiếm phim qua TMDB API theo tên, thể loại và phân trang."""
        query = query.strip()
        selected_genres = [genre] if isinstance(genre, str) and genre else list(genre or [])
        topics = list(topics or [])

        if query:
            # Tìm kiếm theo tên
            movies, total, pages = tmdb_service.search_movies_by_title(query, page=page)
        elif selected_genres:
            # Lọc theo thể loại
            sort_map = {
                'popular': 'popularity.desc',
                'rating': 'vote_average.desc',
                'year': 'primary_release_date.desc'
            }
            sort_by = sort_map.get(sort, 'popularity.desc')
            movies, total, pages = tmdb_service.discover_by_genres(selected_genres, sort_by=sort_by, page=page)
        else:
            # Trang chủ mặc định: lấy trending
            movies, total, pages = tmdb_service.get_trending_movies(time_window='day', page=page)

        # Phim do quản trị viên xuất bản được ưu tiên ở trang kết quả đầu tiên.
        custom_movies = custom_movie_service.search_published(query, selected_genres, limit=page_size) if page == 1 else []
        if custom_movies:
            seen_ids = {movie['id'] for movie in custom_movies}
            movies = custom_movies + [movie for movie in movies if movie['id'] not in seen_ids]
            total += len(custom_movies)

        # Gắn topics và lọc nếu có yêu cầu
        filtered = []
        for m in movies:
            m['topics'] = extract_movie_topics(m)
            m_genres = set(m.get('genres', []))
            m_topics = set(m.get('topics', []))
            if selected_genres and not m_genres.intersection(set(selected_genres)):
                continue
            if topics and not m_topics.intersection(set(topics)):
                continue
            filtered.append(m)

        # Nếu lọc theo topic làm mất hết kết quả khi đang có query tìm kiếm, tự động nới lỏng topic
        if not filtered and query and topics and movies:
            for m in movies:
                m_genres = set(m.get('genres', []))
                if selected_genres and not m_genres.intersection(set(selected_genres)):
                    continue
                filtered.append(m)

        if not filtered and query and movies:
            filtered = movies

        # Cắt phân trang
        total_items = total if not (selected_genres or topics) else len(filtered)
        calc_pages = max(1, (total_items + page_size - 1) // page_size) if (selected_genres or topics) else pages
        page = min(max(1, page), calc_pages)
        res_items = filtered[:page_size] if (selected_genres or topics) else movies[:page_size]

        return res_items, total_items, page, calc_pages

    def search_ai(
        self,
        query: str,
        genres: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 12
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        """Tìm kiếm ngữ nghĩa đa ngôn ngữ (tiếng Việt/Anh) bằng mô hình ONNX Sentence Transformers kết hợp mở rộng thực thể."""
        query = query.strip()
        if not query:
            return self.search(genre=genres, topics=topics, page=page, page_size=page_size)

        candidate_dict: Dict[int, Dict[str, Any]] = {}
        direct_match_ids: Set[int] = set()

        for movie in custom_movie_service.safe_list_published(limit=100):
            candidate_dict[movie['id']] = movie

        # 1. Trích xuất các thực thể và cụm từ khóa ngữ nghĩa
        search_terms = extract_semantic_search_terms(query)

        # Nếu câu truy vấn ngắn (<= 5 từ), tìm kiếm trực tiếp nguyên văn câu truy vấn
        if len(query.split()) <= 5:
            s_movies, _, _ = tmdb_service.search_movies_by_title(query, page=1)
            for m in s_movies:
                candidate_dict[m['id']] = m
                direct_match_ids.add(m['id'])

        # 2. Tìm kiếm ứng viên đa tầng từ các từ khóa/thực thể đã trích xuất
        for term in search_terms:
            t_movies, _, _ = tmdb_service.search_movies_by_title(term, page=1)
            for m in t_movies[:10]:
                if m['id'] not in candidate_dict:
                    candidate_dict[m['id']] = m
                direct_match_ids.add(m['id'])

            # Bổ sung TMDB search với language='vi-VN' cho các term tiếng Việt
            if any(ord(c) > 127 for c in term) or term in ['hải tặc', 'cướp biển', 'vua hải tặc', 'người nhện', 'người dơi']:
                raw_vi = tmdb_service._get("/search/movie", {"query": term, "language": "vi-VN"})
                if raw_vi and raw_vi.get("results"):
                    for r in raw_vi["results"][:10]:
                        m_vi = tmdb_service.format_movie_record(r)
                        if m_vi["id"] not in candidate_dict:
                            candidate_dict[m_vi["id"]] = m_vi
                        direct_match_ids.add(m_vi["id"])

        # 3. Nếu tập ứng viên chưa đủ phong phú (< 20 phim), bổ sung từ discover/trending/top-rated và catalog fallback
        if len(candidate_dict) < 20:
            if genres:
                g_movies, _, _ = tmdb_service.discover_by_genres(genres, page=1)
                for m in g_movies:
                    if m['id'] not in candidate_dict:
                        candidate_dict[m['id']] = m

            t_movies, _, _ = tmdb_service.get_trending_movies(page=1)
            for m in t_movies:
                if m['id'] not in candidate_dict:
                    candidate_dict[m['id']] = m

            p_movies, _, _ = tmdb_service.get_top_rated_movies(page=1)
            for m in p_movies:
                if m['id'] not in candidate_dict:
                    candidate_dict[m['id']] = m

            for m in tmdb_service.FALLBACK_MOVIES:
                if m['id'] not in candidate_dict:
                    candidate_dict[m['id']] = m

        candidate_list = list(candidate_dict.values())
        if not candidate_list:
            candidate_list = tmdb_service.FALLBACK_MOVIES

        # Nhận diện ý định tác phẩm/thương hiệu cụ thể (Cultural Franchise Intent)
        q_lower = query.lower()
        matched_franchises = [
            franchise for keys, franchise in FRANCHISE_INTENTS.items()
            if any(k in q_lower for k in keys)
        ]

        # Tính điểm semantic vector
        scores = compute_query_semantic_scores(query, candidate_list)

        # Lọc theo thể loại & chủ đề (Intersection kết hợp thưởng điểm)
        selected_genres = set(genres or [])
        selected_topics = set(topics or [])
        ranked_items = []

        for idx, m in enumerate(candidate_list):
            m['topics'] = extract_movie_topics(m)
            m_genres = set(m.get('genres', []))
            m_topics = set(m.get('topics', []))

            # Phim phải có ít nhất 1 trong các thể loại đã chọn
            if selected_genres and not m_genres.intersection(selected_genres):
                continue
            # Phim phải có ít nhất 1 trong các chủ đề đã chọn
            if selected_topics and not m_topics.intersection(selected_topics):
                continue

            m_copy = dict(m)
            base_score = float(scores[idx]) if idx < len(scores) else 0.0

            # Ưu tiên các phim tìm được trực tiếp qua truy vấn/thực thể
            if m['id'] in direct_match_ids:
                base_score += 0.25

            # Thưởng điểm nếu nội dung phim chứa trực tiếp thực thể cốt lõi của truy vấn
            m_text = (str(m.get('title', '')) + ' ' + str(m.get('overview', '')) + ' ' + ' '.join(m.get('keywords', []))).lower()
            if any(term in m_text for term in search_terms if len(term) >= 4):
                base_score += 0.10

            # Thưởng điểm đặc biệt nếu khớp ý định tác phẩm cụ thể
            if matched_franchises and any(f in m_text or f in m.get('title', '').lower() for f in matched_franchises):
                base_score += 0.20

            # Thưởng điểm nếu chứa toàn bộ thể loại đã chọn
            if selected_genres and selected_genres.issubset(m_genres):
                base_score += 0.05
            # Thưởng điểm nếu khớp chủ đề
            if selected_topics and selected_topics.intersection(m_topics):
                base_score += 0.08

            # Kết hợp chất lượng Bayesian Rating để ưu tiên tác phẩm kinh điển/uy tín khi điểm tương quan cao
            quality = calculate_bayesian_quality(m.get('vote_average', 0.0), m.get('vote_count', 0))
            blended_score = base_score * 0.90 + quality * 0.10
            m_copy['ai_score'] = round(min(1.0, blended_score), 3)
            ranked_items.append(m_copy)

        # Nếu bộ lọc chủ đề quá khắt khe khiến mất sạch kết quả, tự động nới lỏng chủ đề
        if not ranked_items and selected_topics and candidate_list:
            for idx, m in enumerate(candidate_list):
                m_genres = set(m.get('genres', []))
                if selected_genres and not m_genres.intersection(selected_genres):
                    continue
                m_copy = dict(m)
                base_score = float(scores[idx]) if idx < len(scores) else 0.0
                if m['id'] in direct_match_ids:
                    base_score += 0.25
                m_text = (str(m.get('title', '')) + ' ' + str(m.get('overview', '')) + ' ' + ' '.join(m.get('keywords', []))).lower()
                if any(term in m_text for term in search_terms if len(term) >= 4):
                    base_score += 0.10
                if matched_franchises and any(f in m_text or f in m.get('title', '').lower() for f in matched_franchises):
                    base_score += 0.20
                if selected_genres and selected_genres.issubset(m_genres):
                    base_score += 0.05
                quality = calculate_bayesian_quality(m.get('vote_average', 0.0), m.get('vote_count', 0))
                blended_score = base_score * 0.90 + quality * 0.10
                m_copy['ai_score'] = round(min(1.0, blended_score), 3)
                ranked_items.append(m_copy)

        # Nếu vẫn không còn kết quả nào, giữ lại ứng viên kèm điểm semantic
        if not ranked_items and candidate_list:
            for idx, m in enumerate(candidate_list):
                m_copy = dict(m)
                base_score = float(scores[idx]) if idx < len(scores) else 0.0
                if m['id'] in direct_match_ids:
                    base_score += 0.25
                m_text = (str(m.get('title', '')) + ' ' + str(m.get('overview', '')) + ' ' + ' '.join(m.get('keywords', []))).lower()
                if any(term in m_text for term in search_terms if len(term) >= 4):
                    base_score += 0.10
                if matched_franchises and any(f in m_text or f in m.get('title', '').lower() for f in matched_franchises):
                    base_score += 0.20
                quality = calculate_bayesian_quality(m.get('vote_average', 0.0), m.get('vote_count', 0))
                blended_score = base_score * 0.90 + quality * 0.10
                m_copy['ai_score'] = round(min(1.0, blended_score), 3)
                ranked_items.append(m_copy)

        # Sắp xếp theo ai_score giảm dần kết hợp vote_count và popularity để giải quyết đồng điểm
        ranked_items.sort(key=lambda x: (-x['ai_score'], -x.get('vote_count', 0), -x.get('popularity', 0)))
        total = len(ranked_items)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), pages)
        start = (page - 1) * page_size
        return ranked_items[start:start + page_size], total, page, pages

    def search_by_mood(
        self,
        mood: str,
        context: str = '',
        page: int = 1,
        page_size: int = 12,
    ) -> Tuple[List[Dict[str, Any]], int, int, int, Dict[str, Any]]:
        """Gợi ý phim theo trạng thái cảm xúc, kết hợp hồ sơ tâm trạng và xếp hạng AI."""
        mood_key = resolve_mood_key(mood) or resolve_mood_key(context) or 'anxious'
        profile = dict(MOOD_PROFILES[mood_key])
        profile['key'] = mood_key
        semantic_query = f"{profile['prompt']}. {context.strip()}".strip()

        # Thu thập nhiều nguồn ứng viên để kết quả không bị bó hẹp vào một thể loại.
        with ThreadPoolExecutor(max_workers=3) as executor:
            mood_future = executor.submit(
                tmdb_service.discover_by_genres_any, profile['genres'], 'popularity.desc', 1
            )
            trending_future = executor.submit(tmdb_service.get_trending_movies, 'week', 1)
            top_future = executor.submit(tmdb_service.get_top_rated_movies, 1)
            mood_movies, _, _ = mood_future.result()
            trending_movies, _, _ = trending_future.result()
            top_movies, _, _ = top_future.result()

        candidate_map: Dict[int, Dict[str, Any]] = {}
        for movie in custom_movie_service.safe_list_published(limit=100):
            candidate_map[movie['id']] = movie
        for movie in mood_movies + trending_movies + top_movies + list(tmdb_service.FALLBACK_MOVIES):
            if movie.get('id') is not None:
                candidate_map.setdefault(movie['id'], movie)

        candidates = list(candidate_map.values())
        try:
            semantic_scores = compute_query_semantic_scores(semantic_query, candidates)
        except AIUnavailable:
            semantic_scores = np.zeros(len(candidates), dtype=float)

        wanted_genres = set(profile['genres'])
        unwanted_genres = set(profile.get('avoid_genres', []))
        wanted_keywords = {_normalize_mood_text(word) for word in profile.get('keywords', [])}
        avoid_keywords = {_normalize_mood_text(word) for word in profile.get('avoid_keywords', [])}
        ranked: List[Dict[str, Any]] = []

        for index, movie in enumerate(candidates):
            movie_copy = dict(movie)
            movie_copy['topics'] = extract_movie_topics(movie_copy)
            movie_genres = set(movie_copy.get('genres', []))
            genre_score = min(1.0, len(movie_genres.intersection(wanted_genres)) / 2.0)
            searchable = _normalize_mood_text(' '.join([
                str(movie_copy.get('title', '')),
                str(movie_copy.get('overview', '')),
                ' '.join(movie_copy.get('keywords', []) or []),
                ' '.join(movie_copy.get('topics', []) or []),
                ' '.join(movie_copy.get('genres', []) or []),
            ]))
            keyword_hits = sum(1 for keyword in wanted_keywords if keyword and keyword in searchable)
            keyword_score = min(1.0, keyword_hits / 3.0)
            penalty = 0.12 if any(keyword and keyword in searchable for keyword in avoid_keywords) else 0.0
            if movie_genres.intersection(unwanted_genres):
                penalty += 0.16
            semantic_score = float(semantic_scores[index]) if index < len(semantic_scores) else 0.0
            semantic_score = max(0.0, min(1.0, semantic_score))
            quality_score = calculate_bayesian_quality(
                movie_copy.get('vote_average', 0.0), movie_copy.get('vote_count', 0)
            )
            mood_score = max(
                0.0,
                semantic_score * 0.50 + genre_score * 0.22 + keyword_score * 0.13 + quality_score * 0.15 - penalty,
            )
            movie_copy['ai_score'] = round(min(1.0, mood_score), 3)
            movie_copy['mood_match'] = round(min(100.0, mood_score * 100), 1)
            movie_copy['mood_reason'] = profile['description']
            ranked.append(movie_copy)

        ranked.sort(key=lambda item: (
            -item['ai_score'], -item.get('vote_average', 0), -item.get('vote_count', 0)
        ))
        total = len(ranked)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), pages)
        start = (page - 1) * page_size
        return ranked[start:start + page_size], total, page, pages, profile

    def title_matches(self, query: str) -> List[Dict[str, Any]]:
        """Tìm phim làm mẫu độc lập với bộ lọc."""
        query = query.strip()
        if not query:
            return []
        movies, _, _ = tmdb_service.search_movies_by_title(query, page=1)
        custom_movies = custom_movie_service.search_published(query, limit=10)
        movies = custom_movies + [m for m in movies if m['id'] not in {x['id'] for x in custom_movies}]
        for m in movies:
            m['exact'] = (m['title'].casefold() == query.casefold())
            m['topics'] = extract_movie_topics(m)
        movies.sort(key=lambda x: (not x['exact'], -x.get('popularity', 0)))
        return movies[:10]

    def recommend(
        self,
        movie_id: int,
        top_n: int = 6,
        genres: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        method: str = 'hybrid'
    ) -> List[Dict[str, Any]]:
        """
        Two-Tier Real-Time Re-ranking:
        Tầng 1: Lấy 25-40 ứng viên từ TMDB API (similar, recommendations, genre discover).
        Tầng 2: Áp dụng Weighted Score Fusion + Soft-penalty thể loại + Explainability breakdown.
        """
        target = self.get_movie(movie_id)
        if not target:
            return []

        # 1. Thu thập ứng viên. Phim nội bộ dùng discover/trending vì chưa có TMDB id.
        if movie_id < 0:
            candidates, _, _ = tmdb_service.discover_by_genres(target.get('genres', [])[:2], page=1)
            if len(candidates) < 15:
                trending, _, _ = tmdb_service.get_trending_movies(page=1)
                candidates.extend(trending)
        else:
            candidates = tmdb_service.fetch_candidate_movies(movie_id, target_movie=target, limit=40)

        # Phim nội bộ đã xuất bản cũng tham gia tập ứng viên cho mọi nguồn phim.
        candidate_map = {candidate['id']: candidate for candidate in candidates}
        for custom_movie in custom_movie_service.safe_list_published(limit=40):
            candidate_map.setdefault(custom_movie['id'], custom_movie)
        candidates = list(candidate_map.values())[:60]
        # Loại bỏ chính phim mẫu
        candidates = [c for c in candidates if c['id'] != movie_id]

        if not candidates:
            return []

        # 2. Tính S_semantic (AI Contextual Similarity)
        try:
            semantic_scores = compute_candidate_semantic_scores(target, candidates)
        except AIUnavailable:
            semantic_scores = [0.0] * len(candidates)

        # 3. Tính S_lexical (TF-IDF Similarity trên overviews + keywords của tập ứng viên)
        target_text = f"{target.get('title', '')} {target.get('overview', '')} {' '.join(target.get('keywords', []))}"
        cand_texts = [f"{c.get('title', '')} {c.get('overview', '')} {' '.join(c.get('keywords', []))}" for c in candidates]
        all_texts = [target_text] + cand_texts
        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            lexical_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).ravel()
        except Exception:
            lexical_sim = np.zeros(len(candidates))

        # 4. Tính toán phân tầng Metadata & Bayesian Rating cho từng ứng viên
        target_genres = set(target.get('genres', []))
        target_cast = set(target.get('cast', [])[:3])
        target_director = target.get('director', '').strip()
        target_keywords = set(k.lower() for k in target.get('keywords', []))

        scored_candidates = []
        for i, c in enumerate(candidates):
            c_genres = set(c.get('genres', []))
            c_cast = set(c.get('cast', [])[:3])
            c_director = c.get('director', '').strip()
            c_keywords = set(k.lower() for k in c.get('keywords', []))

            # Bộ lọc genres và topics nếu người dùng chỉ định
            if genres and not set(genres).issubset(c_genres):
                continue
            c_topics = extract_movie_topics(c)
            if topics and not set(topics).issubset(set(c_topics)):
                continue

            # Thành phần 1: S_semantic
            s_sem = float(semantic_scores[i]) if i < len(semantic_scores) else 0.0

            # Thành phần 2: S_lexical
            s_lex = float(lexical_sim[i]) if i < len(lexical_sim) else 0.0

            # Thành phần 3: S_metadata
            s_genre = calculate_jaccard_similarity(target_genres, c_genres)
            dir_match = 1.0 if (target_director and target_director.lower() == c_director.lower()) else 0.0
            cast_match = calculate_jaccard_similarity(target_cast, c_cast)
            s_crew = 0.6 * dir_match + 0.4 * cast_match
            s_meta = 0.7 * s_genre + 0.3 * s_crew

            # Thành phần 4: S_quality (Bayesian Rating)
            s_qual = calculate_bayesian_quality(
                vote_average=float(c.get('vote_average', 0.0)),
                vote_count=int(c.get('vote_count', 0))
            )

            # Weighted Score Fusion
            if method == 'ai':
                raw_score = 0.85 * s_sem + 0.15 * s_meta
            elif method == 'tfidf':
                raw_score = 0.50 * s_lex + 0.40 * s_meta + 0.10 * s_qual
            else:  # hybrid
                raw_score = 0.40 * s_sem + 0.25 * s_lex + 0.25 * s_meta + 0.10 * s_qual

            # Soft-penalty: nếu hoàn toàn không chung thể loại chính, phạt điểm 0.4x
            shared_genres = sorted(list(target_genres.intersection(c_genres)))
            has_genre_overlap = len(shared_genres) > 0
            if not has_genre_overlap:
                final_score = raw_score * 0.40
            else:
                final_score = raw_score

            # Tạo Explainability Breakdown
            c_dict = dict(c)
            c_dict['topics'] = c_topics
            c_dict['similarity'] = round(float(final_score), 3)
            c_dict['match_percentage'] = min(99, max(15, int(round(final_score * 100))))
            c_dict['breakdown'] = {
                'semantic_score': round(s_sem * 100, 1),
                'genre_score': round(s_genre * 100, 1),
                'quality_score': round(s_qual * 10, 1),
                'has_genre_overlap': has_genre_overlap,
            }

            # Lý do tóm tắt
            reasons = []
            if shared_genres:
                reasons.append(f"Thể loại: {', '.join(shared_genres[:3])}")
            if dir_match > 0:
                reasons.append(f"Cùng đạo diễn: {target_director}")
            shared_cast = sorted(list(target_cast.intersection(c_cast)))
            if shared_cast:
                reasons.append(f"Diễn viên chung: {', '.join(shared_cast[:2])}")
            shared_kw = sorted(list(target_keywords.intersection(c_keywords)))
            if shared_kw:
                reasons.append(f"Chủ đề: {', '.join(shared_kw[:2])}")

            c_dict['reason'] = " • ".join(reasons) or "Cốt truyện và phong cách tương đồng sâu theo phân tích AI."
            scored_candidates.append(c_dict)

        # Sắp xếp theo điểm tương đồng giảm dần
        scored_candidates.sort(key=lambda x: (-x['similarity'], -x.get('popularity', 0)))
        return scored_candidates[:top_n]
