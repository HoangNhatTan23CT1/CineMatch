"""
@file tmdb_service.py
@module tmdb_service
@description Dịch vụ kết nối TMDB API v3 & OMDB API, hỗ trợ đa mirror (api.tmdb.org, tmdb-proxy), cache in-memory, retry và fallback demo
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Base URLs đa mirror: Ưu tiên api.tmdb.org (không bị chặn ở Việt Nam) và Vercel proxy
TMDB_BASE_URLS = [
    "https://api.tmdb.org/3",
    "https://tmdb-proxy.vercel.app/3",
    "https://api.themoviedb.org/3",
]
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE_URL = "https://image.tmdb.org/t/p/original"
ENV_PATH = Path(__file__).resolve().parent / ".env"

# Memory cache với timestamp và TTL đa tầng
_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_FEED_TTL = 3600      # 1 giờ cho danh sách feed (trending, top-rated, search)
_CACHE_DETAIL_TTL = 86400   # 24 giờ cho chi tiết phim cố định (movie detail)

# Singleton HTTP Session với connection pooling tái sử dụng kết nối TLS/TCP
_HTTP_SESSION: Optional[requests.Session] = None


def get_http_session() -> requests.Session:
    """Khởi tạo và tái sử dụng requests.Session với Connection Pooling."""
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=30, max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _HTTP_SESSION = session
    return _HTTP_SESSION

# Cấu hình API Key toàn cục trong runtime
_RUNNING_API_KEY: Optional[str] = None


def load_env_api_key() -> Optional[str]:
    """Đọc TMDB_API_KEY từ biến môi trường hoặc file .env."""
    key = os.environ.get("TMDB_API_KEY")
    if key and key.strip():
        return key.strip()

    # Chỉ đọc file .env của CineMatch để tránh phụ thuộc đường dẫn máy cá nhân.
    possible_envs = [ENV_PATH]
    for env_file in possible_envs:
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TMDB_API_KEY=") and not line.startswith("#"):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception as exc:
                print(f"[TMDB][Env] Error reading {env_file}: {exc}")

    return None


def get_active_api_key() -> Optional[str]:
    """Trả về API Key đang được áp dụng."""
    global _RUNNING_API_KEY
    if _RUNNING_API_KEY:
        return _RUNNING_API_KEY
    _RUNNING_API_KEY = load_env_api_key()
    return _RUNNING_API_KEY


def set_active_api_key(api_key: str, persist_to_env: bool = True) -> bool:
    """Cập nhật API Key mới và tùy chọn lưu vào file .env."""
    global _RUNNING_API_KEY
    clean_key = api_key.strip()
    _RUNNING_API_KEY = clean_key
    os.environ["TMDB_API_KEY"] = clean_key
    _CACHE.clear()

    if persist_to_env:
        try:
            lines = []
            key_written = False
            if ENV_PATH.exists():
                with open(ENV_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    if line.strip().startswith("TMDB_API_KEY="):
                        new_lines.append(f"TMDB_API_KEY={clean_key}\n")
                        key_written = True
                    else:
                        new_lines.append(line)
                lines = new_lines

            if not key_written:
                lines.append(f"TMDB_API_KEY={clean_key}\n")

            with open(ENV_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"[TMDB][Config] Saved API Key to {ENV_PATH}")
            return True
        except Exception as exc:
            print(f"[TMDB][Config] Error saving .env: {exc}")
            return False
    return True


def has_valid_api_key() -> bool:
    """Kiểm tra có API Key khả dụng hay không."""
    key = get_active_api_key()
    return bool(key and len(key) >= 16)


# ==========================================
# CURATED DEMO FALLBACK (Khi chưa có key / offline)
# ==========================================
FALLBACK_GENRES = [
    {"id": 28, "name": "Action"},
    {"id": 12, "name": "Adventure"},
    {"id": 16, "name": "Animation"},
    {"id": 35, "name": "Comedy"},
    {"id": 80, "name": "Crime"},
    {"id": 99, "name": "Documentary"},
    {"id": 18, "name": "Drama"},
    {"id": 10751, "name": "Family"},
    {"id": 14, "name": "Fantasy"},
    {"id": 36, "name": "History"},
    {"id": 27, "name": "Horror"},
    {"id": 10402, "name": "Music"},
    {"id": 9648, "name": "Mystery"},
    {"id": 10749, "name": "Romance"},
    {"id": 878, "name": "Science Fiction"},
    {"id": 10770, "name": "TV Movie"},
    {"id": 53, "name": "Thriller"},
    {"id": 10752, "name": "War"},
    {"id": 37, "name": "Western"},
]

FALLBACK_MOVIES: List[Dict[str, Any]] = [
    {
        "id": 157336,
        "title": "Interstellar",
        "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.",
        "release_date": "2014-11-05",
        "vote_average": 8.4,
        "vote_count": 34500,
        "popularity": 145.2,
        "poster_path": "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "backdrop_path": "/xJHokMbljvjADYdit5fK5VQsXEG.jpg",
        "genres": ["Adventure", "Drama", "Science Fiction"],
        "director": "Christopher Nolan",
        "cast": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
        "keywords": ["wormhole", "space travel", "astronaut", "black hole", "future"],
    },
    {
        "id": 27205,
        "title": "Inception",
        "overview": "Cobb, a skilled thief who steals corporate secrets through the use of dream-sharing technology, is given the inverse task of planting an idea into the mind of a C.E.O.",
        "release_date": "2010-07-15",
        "vote_average": 8.36,
        "vote_count": 36000,
        "popularity": 132.8,
        "poster_path": "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
        "backdrop_path": "/s3TBrRGB1jav7cUneYEvNkHQ5QO.jpg",
        "genres": ["Action", "Science Fiction", "Adventure"],
        "director": "Christopher Nolan",
        "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "keywords": ["dream", "subconscious", "heist", "mind bending", "architect"],
    },
    {
        "id": 155,
        "title": "The Dark Knight",
        "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets.",
        "release_date": "2008-07-16",
        "vote_average": 8.51,
        "vote_count": 32000,
        "popularity": 120.4,
        "poster_path": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "backdrop_path": "/nMKdUUepR0i5zn0y1T4CsSB5chy.jpg",
        "genres": ["Drama", "Action", "Crime", "Thriller"],
        "director": "Christopher Nolan",
        "cast": ["Christian Bale", "Heath Ledger", "Aaron Eckhart"],
        "keywords": ["dc comics", "joker", "batman", "vigilante", "gotham city"],
    },
    {
        "id": 19995,
        "title": "Avatar",
        "overview": "In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.",
        "release_date": "2009-12-15",
        "vote_average": 7.58,
        "vote_count": 31000,
        "popularity": 110.5,
        "poster_path": "/gKY6q7SjCkAU6FqvqWybDYgUKIF.jpg",
        "backdrop_path": "/vL5LR6WdxWPjLPFRLe133jXWsh5.jpg",
        "genres": ["Action", "Adventure", "Fantasy", "Science Fiction"],
        "director": "James Cameron",
        "cast": ["Sam Worthington", "Zoe Saldana", "Sigourney Weaver"],
        "keywords": ["alien", "pandora", "nature", "space", "avatar", "marine"],
    },
    {
        "id": 597,
        "title": "Titanic",
        "overview": "101-year-old Rose DeWitt Bukater tells the story of her life aboard the Titanic, 84 years later. A young Rose boards the ship with her mother and fiance, but falls in love with poor artist Jack Dawson.",
        "release_date": "1997-11-18",
        "vote_average": 7.9,
        "vote_count": 24000,
        "popularity": 98.7,
        "poster_path": "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
        "backdrop_path": "/yDI6D5ZQh67YU4r2ms8qcSbAvi8.jpg",
        "genres": ["Drama", "Romance"],
        "director": "James Cameron",
        "cast": ["Leonardo DiCaprio", "Kate Winslet", "Billy Zane"],
        "keywords": ["shipwreck", "tragic romance", "iceberg", "class divide", "atlantic ocean"],
    },
    {
        "id": 862,
        "title": "Toy Story",
        "overview": "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz.",
        "release_date": "1995-10-30",
        "vote_average": 7.97,
        "vote_count": 17800,
        "popularity": 90.2,
        "poster_path": "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
        "backdrop_path": "/3Rfvhy1Nl6sSGJwyjb0QiZzZYlB.jpg",
        "genres": ["Animation", "Adventure", "Family", "Comedy"],
        "director": "John Lasseter",
        "cast": ["Tom Hanks", "Tim Allen", "Don Rickles"],
        "keywords": ["toys", "friendship", "jealousy", "living toy", "childhood"],
    },
    {
        "id": 244786,
        "title": "Whiplash",
        "overview": "Under the direction of a ruthless instructor, a talented young drummer begins to pursue perfection at any cost, testing his physical and emotional limits.",
        "release_date": "2014-10-10",
        "vote_average": 8.38,
        "vote_count": 14500,
        "popularity": 75.3,
        "poster_path": "/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
        "backdrop_path": "/6bbZ6XyvgfjhQwfplVuh1nxudQc.jpg",
        "genres": ["Drama", "Music"],
        "director": "Damien Chazelle",
        "cast": ["Miles Teller", "J.K. Simmons", "Paul Reiser"],
        "keywords": ["drummer", "jazz", "perfectionism", "music teacher", "obsession"],
    },
    {
        "id": 138843,
        "title": "The Conjuring",
        "overview": "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse.",
        "release_date": "2013-07-18",
        "vote_average": 7.54,
        "vote_count": 11200,
        "popularity": 68.4,
        "poster_path": "/wVYREutTvI2tmxr6ujrHT704wGF.jpg",
        "backdrop_path": "/m1pHkKjHkO9m49wL6J48aA6m7pA.jpg",
        "genres": ["Horror", "Thriller"],
        "director": "James Wan",
        "cast": ["Patrick Wilson", "Vera Farmiga", "Ron Livingston"],
        "keywords": ["haunted house", "demon", "exorcism", "paranormal", "ghost"],
    },
    {
        "id": 299536,
        "title": "Avengers: Infinity War",
        "overview": "As the Avengers and their allies have continued to protect the world from threats too large for any one hero to handle, a new danger has emerged from the cosmic shadows: Thanos.",
        "release_date": "2018-04-25",
        "vote_average": 8.25,
        "vote_count": 29000,
        "popularity": 135.6,
        "poster_path": "/7WsyChQLEftFiDOVTGkv3hFpyyt.jpg",
        "backdrop_path": "/mDfJG3LC3Dqb67AZ52x3Z0jU0pw.jpg",
        "genres": ["Adventure", "Action", "Science Fiction"],
        "director": "Anthony Russo",
        "cast": ["Robert Downey Jr.", "Chris Hemsworth", "Mark Ruffalo"],
        "keywords": ["superhero", "marvel", "thanos", "infinity stones", "cosmic battle"],
    },
    {
        "id": 278,
        "title": "The Shawshank Redemption",
        "overview": "Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison.",
        "release_date": "1994-09-23",
        "vote_average": 8.71,
        "vote_count": 26800,
        "popularity": 115.1,
        "poster_path": "/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
        "backdrop_path": "/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg",
        "genres": ["Drama", "Crime"],
        "director": "Frank Darabont",
        "cast": ["Tim Robbins", "Morgan Freeman", "Bob Gunton"],
        "keywords": ["prison", "escape", "friendship", "hope", "wrongful imprisonment"],
    },
    {
        "id": 12,
        "title": "Finding Nemo",
        "overview": "Nemo, an adventurous young clownfish, is unexpectedly taken from his Great Barrier Reef home and thrust into a fish tank in a dentist's office. His father Marlin must find him.",
        "release_date": "2003-05-30",
        "vote_average": 7.82,
        "vote_count": 18900,
        "popularity": 82.5,
        "poster_path": "/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg",
        "backdrop_path": "/eCynaAOgYYiw5yN5lBwz3IxqvaW.jpg",
        "genres": ["Animation", "Family"],
        "director": "Andrew Stanton",
        "cast": ["Albert Brooks", "Ellen DeGeneres", "Alexander Gould"],
        "keywords": ["ocean", "clownfish", "father son", "great barrier reef", "rescue mission"],
    },
    {
        "id": 11,
        "title": "Star Wars",
        "overview": "Princess Leia is captured and held hostage by the evil Imperial forces in their effort to take over the galactic Empire. Venturesome Luke Skywalker and dashing captain Han Solo team together to save the beautiful princess.",
        "release_date": "1977-05-25",
        "vote_average": 8.2,
        "vote_count": 20100,
        "popularity": 95.0,
        "poster_path": "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
        "backdrop_path": "/zqkmTXzjkAgPn7q2w9L0B7v0M6y.jpg",
        "genres": ["Adventure", "Action", "Science Fiction"],
        "director": "George Lucas",
        "cast": ["Mark Hamill", "Harrison Ford", "Carrie Fisher"],
        "keywords": ["space opera", "jedi", "empire", "galactic war", "rebellion"],
    },
    {
        "id": 900667,
        "title": "One Piece Film Red",
        "overview": "Uta — the most beloved singer in the world whose voice has been described as other-worldly — is renowned for concealing her own identity when performing. Now, for the first time ever, she will reveal herself to the world at a live concert.",
        "release_date": "2022-08-06",
        "vote_average": 7.3,
        "vote_count": 920,
        "popularity": 78.4,
        "poster_path": "/8ibfhe4P7rhmn3lrPhOZzIJHA2B.jpg",
        "backdrop_path": "/bQXAqRx2gGHA65sPUMeq2PpxkWB.jpg",
        "genres": ["Animation", "Action", "Adventure", "Fantasy", "Music"],
        "director": "Goro Taniguchi",
        "cast": ["Mayumi Tanaka", "Kaori Nazuka", "Ado", "Kenjiro Tsuda", "Shuichi Ikeda"],
        "keywords": ["anime", "pirate", "singer", "concert", "island", "luffy", "straw hat"],
    }
]

# Đảm bảo FALLBACK_MOVIES luôn có poster_url và backdrop_url đầy đủ
for _m in FALLBACK_MOVIES:
    if not _m.get("poster_url") and _m.get("poster_path"):
        _m["poster_url"] = f"{TMDB_IMAGE_BASE_URL}{_m['poster_path']}"
    if not _m.get("backdrop_url") and _m.get("backdrop_path"):
        _m["backdrop_url"] = f"{TMDB_BACKDROP_BASE_URL}{_m['backdrop_path']}"
    _m["year"] = _m["release_date"][:4] if _m.get("release_date") else "N/A"


# ==========================================
# MULTI-MIRROR API CLIENT & HELPERS
# ==========================================
def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Gửi GET request qua danh sách mirror với HTTP Session pooling và TTL Caching thích ứng."""
    api_key = get_active_api_key()
    if not api_key:
        return None

    params = params or {}
    params["api_key"] = api_key

    cache_key = f"{endpoint}?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "api_key")
    now = time.time()

    # TTL thích ứng: Chi tiết phim lưu 24h, danh sách feed/search lưu 1h
    is_detail_endpoint = bool(re.match(r"^/movie/\d+$", endpoint))
    ttl = _CACHE_DETAIL_TTL if is_detail_endpoint else _CACHE_FEED_TTL

    if cache_key in _CACHE:
        timestamp, data = _CACHE[cache_key]
        if now - timestamp < ttl:
            return data

    session = get_http_session()

    # Thử lần lượt các endpoint mirror qua session connection pool
    for base_url in TMDB_BASE_URLS:
        url = f"{base_url}{endpoint}"
        try:
            response = session.get(url, params=params, timeout=4)
            if response.status_code == 200:
                data = response.json()
                _CACHE[cache_key] = (now, data)
                return data
            elif response.status_code == 401:
                print(f"[TMDB][Auth] Invalid API Key (HTTP 401) on {base_url}.")
                return None
            else:
                continue
        except requests.RequestException:
            continue

    print(f"[TMDB][Network] All TMDB mirrors failed for {endpoint}.")
    return None


def format_movie_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa dữ liệu thô từ TMDB API thành định dạng thống nhất của CineMatch."""
    raw_genres = raw.get("genres") or []
    if raw_genres and isinstance(raw_genres[0], dict):
        genres = [g.get("name", "") for g in raw_genres if g.get("name")]
    elif raw_genres and isinstance(raw_genres[0], str):
        genres = raw_genres
    else:
        genre_id_map = {g["id"]: g["name"] for g in FALLBACK_GENRES}
        genres = [genre_id_map.get(gid, "") for gid in raw.get("genre_ids", []) if gid in genre_id_map]

    credits = raw.get("credits") or {}
    cast_list = []
    director_name = raw.get("director") or ""
    if credits:
        for c in credits.get("cast", [])[:10]:
            cast_list.append(c.get("name", ""))
        if not director_name:
            for crew_member in credits.get("crew", []):
                if crew_member.get("job") == "Director":
                    director_name = crew_member.get("name", "")
                    break

    keywords_data = raw.get("keywords") or {}
    keywords = []
    if isinstance(keywords_data, dict):
        for kw in keywords_data.get("keywords", []):
            keywords.append(kw.get("name", ""))
    elif isinstance(keywords_data, list):
        keywords = [kw if isinstance(kw, str) else kw.get("name", "") for kw in keywords_data]

    poster_path = raw.get("poster_path") or ""
    poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else (raw.get("poster_url") or "")
    backdrop_path = raw.get("backdrop_path") or ""
    backdrop_url = f"{TMDB_BACKDROP_BASE_URL}{backdrop_path}" if backdrop_path else (raw.get("backdrop_url") or "")

    release_date = raw.get("release_date") or ""
    year = release_date[:4] if release_date and len(release_date) >= 4 else "N/A"

    orig_title = (raw.get("original_title") or "").strip()
    trans_title = (raw.get("title") or "").strip()
    orig_lang = raw.get("original_language") or ""
    if orig_lang == "vi" and orig_title:
        if trans_title and trans_title.lower() != orig_title.lower():
            display_title = f"{orig_title} ({trans_title})"
        else:
            display_title = orig_title
    else:
        display_title = trans_title or orig_title or "Unknown Title"

    movie_id = int(raw.get("id", 0))
    videos = (raw.get("videos") or {}).get("results", [])
    trailer = next(
        (
            video for video in videos
            if video.get("site") == "YouTube" and video.get("type") == "Trailer" and video.get("key")
        ),
        None,
    )
    if trailer is None:
        trailer = next((video for video in videos if video.get("site") == "YouTube" and video.get("key")), None)
    trailer_url = f"https://www.youtube.com/watch?v={trailer['key']}" if trailer else (raw.get("trailer_url") or "")

    external_ids = raw.get("external_ids") or {}
    imdb_id = raw.get("imdb_id") or external_ids.get("imdb_id") or ""
    tmdb_url = f"https://www.themoviedb.org/movie/{movie_id}" if movie_id else ""
    sources = []
    if tmdb_url:
        sources.append({"label": "TMDB", "url": tmdb_url, "kind": "database"})
    if imdb_id:
        sources.append({"label": "IMDb", "url": f"https://www.imdb.com/title/{imdb_id}/", "kind": "database"})
    homepage = raw.get("homepage") or ""
    if homepage:
        sources.append({"label": "Trang chính thức", "url": homepage, "kind": "official"})

    provider_data = raw.get("watch/providers") or raw.get("watch_providers") or {}
    provider_regions = provider_data.get("results", {}) if isinstance(provider_data, dict) else {}
    watch_region = "VN" if provider_regions.get("VN") else ("US" if provider_regions.get("US") else "")
    region_data = provider_regions.get(watch_region, {}) if watch_region else {}
    watch_link = region_data.get("link") or tmdb_url
    watch_providers = []
    seen_provider_ids = set()
    provider_types = (
        ("flatrate", "Thuê bao"), ("free", "Miễn phí"), ("ads", "Có quảng cáo"),
        ("rent", "Thuê"), ("buy", "Mua"),
    )
    for provider_type, type_label in provider_types:
        for provider in region_data.get(provider_type, []) or []:
            provider_id = provider.get("provider_id") or provider.get("provider_name")
            if provider_id in seen_provider_ids:
                continue
            seen_provider_ids.add(provider_id)
            logo_path = provider.get("logo_path") or ""
            watch_providers.append({
                "name": provider.get("provider_name") or "Nền tảng xem phim",
                "type": type_label,
                "url": watch_link,
                "logo_url": f"{TMDB_IMAGE_BASE_URL}{logo_path}" if logo_path else "",
            })

    return {
        "id": movie_id,
        "title": display_title,
        "original_title": orig_title,
        "original_language": orig_lang,
        "overview": raw.get("overview") or "",
        "release_date": release_date,
        "year": year,
        "vote_average": round(float(raw.get("vote_average", 0.0)), 1),
        "vote_count": int(raw.get("vote_count", 0)),
        "popularity": round(float(raw.get("popularity", 0.0)), 1),
        "poster_path": poster_path,
        "poster_url": poster_url,
        "backdrop_path": backdrop_path,
        "backdrop_url": backdrop_url,
        "genres": genres,
        "director": director_name,
        "cast": cast_list or raw.get("cast", []),
        "keywords": keywords or raw.get("keywords", []),
        "tagline": raw.get("tagline") or "",
        "runtime": raw.get("runtime") or 0,
        "trailer_url": trailer_url,
        "homepage": homepage,
        "imdb_id": imdb_id,
        "tmdb_url": tmdb_url,
        "source_url": raw.get("source_url") or "",
        "source_label": raw.get("source_label") or "TMDB",
        "sources": sources,
        "watch_region": watch_region,
        "watch_providers": watch_providers,
    }


def search_omdb_fallback(query: str) -> List[Dict[str, Any]]:
    """Tìm kiếm bổ sung qua OMDB API khi TMDB không có hoặc bị giới hạn."""
    omdb_key = os.environ.get("OMDB_API_KEY", "28eadd36")
    try:
        res = requests.get(f"http://www.omdbapi.com/?apikey={omdb_key}&s={query}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            results = []
            for item in data.get("Search", []):
                imdb_id = item.get("imdbID", "")
                num_id = int(imdb_id.replace("tt", "")) if imdb_id.startswith("tt") and imdb_id.replace("tt", "").isdigit() else abs(hash(item.get("Title", ""))) % 10000000
                poster = item.get("Poster") if item.get("Poster") != "N/A" else ""
                results.append({
                    "id": num_id,
                    "title": item.get("Title", ""),
                    "overview": f"Tác phẩm {item.get('Title')} ({item.get('Year')}).",
                    "release_date": item.get("Year", ""),
                    "year": item.get("Year", ""),
                    "vote_average": 8.0,
                    "vote_count": 500,
                    "popularity": 50.0,
                    "poster_path": "",
                    "poster_url": poster,
                    "backdrop_path": "",
                    "backdrop_url": "",
                    "genres": ["Action", "Adventure", "Animation"],
                    "director": "",
                    "cast": [],
                    "keywords": [query.lower()],
                })
            return results
    except Exception:
        pass
    return []


# ==========================================
# PUBLIC API METHODS (TMDB SERVICE)
# ==========================================
def get_genres() -> List[Dict[str, Any]]:
    """Lấy danh sách thể loại phim từ TMDB API hoặc bộ genres fallback."""
    data = _get("/genre/movie/list")
    if data and "genres" in data:
        return data["genres"]
    return FALLBACK_GENRES


def get_trending_movies(time_window: str = "day", page: int = 1) -> Tuple[List[Dict[str, Any]], int, int]:
    """Lấy danh sách phim thịnh hành (Trending Today/Week)."""
    data = _get(f"/trending/movie/{time_window}", {"page": page})
    if data and "results" in data:
        movies = [format_movie_record(m) for m in data["results"]]
        total = data.get("total_results", len(movies))
        pages = min(data.get("total_pages", 1), 50)
        return movies, total, pages

    return FALLBACK_MOVIES, len(FALLBACK_MOVIES), 1


def get_top_rated_movies(page: int = 1) -> Tuple[List[Dict[str, Any]], int, int]:
    """Lấy danh sách phim có điểm đánh giá cao nhất."""
    data = _get("/movie/top_rated", {"page": page})
    if data and "results" in data:
        movies = [format_movie_record(m) for m in data["results"]]
        total = data.get("total_results", len(movies))
        pages = min(data.get("total_pages", 1), 50)
        return movies, total, pages

    sorted_fallback = sorted(FALLBACK_MOVIES, key=lambda x: x["vote_average"], reverse=True)
    return sorted_fallback, len(sorted_fallback), 1


def get_now_playing_movies(page: int = 1) -> Tuple[List[Dict[str, Any]], int, int]:
    """Lấy danh sách phim đang chiếu rạp."""
    data = _get("/movie/now_playing", {"page": page})
    if data and "results" in data:
        movies = [format_movie_record(m) for m in data["results"]]
        total = data.get("total_results", len(movies))
        pages = min(data.get("total_pages", 1), 50)
        return movies, total, pages

    return FALLBACK_MOVIES, len(FALLBACK_MOVIES), 1


def resolve_natural_language_search(query: str, page: int = 1) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Phân tích câu truy vấn tự nhiên (ngôn ngữ tự nhiên tiếng Việt / tiếng Anh):
    - Đạo diễn / Diễn viên / Người nổi tiếng (ví dụ: 'phim của trấn thành đạo diễn', 'phim do christopher nolan đạo diễn', 'phim có tom cruise')
    - Quốc gia / Ngôn ngữ (ví dụ: 'phim việt nam', 'phim hàn quốc', 'phim hoạt hình nhật bản')
    - Thể loại kết hợp ngôn ngữ (ví dụ: 'phim kinh dị việt nam', 'phim hài hàn quốc')
    """
    query_clean = query.strip()
    if not query_clean:
        return [], 0, 0

    is_director = bool(re.search(r'\b(đạo diễn|dao dien|director|directed by|đạo diễn bởi)\b', query_clean, re.I))
    is_actor = bool(re.search(r'\b(diễn viên|dien vien|actor|actress|đóng|starring|thủ vai)\b', query_clean, re.I))

    lang = None
    if re.search(r'\b(việt nam|viet nam|vietnam|phim việt)\b', query_clean, re.I):
        lang = 'vi'
    elif re.search(r'\b(hàn quốc|han quoc|korea|korean)\b', query_clean, re.I):
        lang = 'ko'
    elif re.search(r'\b(nhật bản|nhat ban|japan|japanese|anime)\b', query_clean, re.I):
        lang = 'ja'
    elif re.search(r'\b(trung quốc|trung quoc|china|chinese|hoa ngữ)\b', query_clean, re.I):
        lang = 'zh'
    elif re.search(r'\b(thái lan|thai lan|thailand|thai)\b', query_clean, re.I):
        lang = 'th'

    genre_map = {
        'kinh dị': 27, 'hài': 35, 'hành động': 28, 'hoạt hình': 16,
        'tình cảm': 10749, 'lãng mạn': 10749, 'viễn tưởng': 878,
        'khoa học viễn tưởng': 878, 'phiêu lưu': 12, 'chiến tranh': 10752,
        'tội phạm': 80, 'hình sự': 80, 'tâm lý': 18, 'gia đình': 10751,
        'bí ẩn': 9648, 'horror': 27, 'comedy': 35, 'action': 28,
        'animation': 16, 'romance': 10749, 'drama': 18
    }
    found_genre_id = None
    for kw, gid in genre_map.items():
        if re.search(rf'\b{re.escape(kw)}\b', query_clean, re.I):
            found_genre_id = gid
            break

    # Tách thực thể tên người (Person Name Entity Extraction)
    stopwords = [
        'phim', 'bộ phim', 'tác phẩm', 'các phim', 'những bộ phim', 'danh sách phim',
        'của', 'do', 'đạo diễn bởi', 'đạo diễn', 'diễn viên', 'đóng vai', 'đóng', 'thủ vai',
        'việt nam', 'viet nam', 'vietnam', 'hàn quốc', 'nhật bản', 'trung quốc', 'thái lan',
        'movie', 'movies', 'film', 'films', 'directed by', 'director', 'actor', 'starring', 'by'
    ]
    cleaned = query_clean
    for sw in sorted(stopwords, key=len, reverse=True):
        cleaned = re.sub(rf'\b{re.escape(sw)}\b', ' ', cleaned, flags=re.I)
    candidate_name = ' '.join(cleaned.split())

    # 1. Thử tìm kiếm theo Nhân vật (Person Search) trên TMDB
    if len(candidate_name) >= 2:
        p_data = _get('/search/person', {'query': candidate_name})
        if p_data and p_data.get('results'):
            person = p_data['results'][0]
            pid = person['id']
            pname = person.get('name') or candidate_name
            credits = _get(f'/person/{pid}/movie_credits')
            if credits:
                raw_list = []
                if is_director:
                    raw_list = [c for c in credits.get('crew', []) if c.get('job') == 'Director']
                elif is_actor:
                    raw_list = credits.get('cast', [])
                else:
                    seen = set()
                    for c in credits.get('crew', []):
                        if c.get('job') == 'Director' and c.get('id') not in seen:
                            raw_list.append(c)
                            seen.add(c.get('id'))
                    for c in credits.get('cast', []):
                        if c.get('id') not in seen:
                            raw_list.append(c)
                            seen.add(c.get('id'))

                movies = []
                for m in raw_list:
                    rec = format_movie_record(m)
                    if is_director or any(c.get('job') == 'Director' for c in credits.get('crew', []) if c.get('id') == m.get('id')):
                        rec['director'] = pname
                    movies.append(rec)

                # Ưu tiên phim đúng ngôn ngữ nếu có yêu cầu (ví dụ: 'phim việt nam của...')
                if lang:
                    movies.sort(key=lambda x: (
                        not (x.get('original_language') == lang or lang in x.get('title', '').lower()),
                        -(x.get('popularity') or 0)
                    ))
                else:
                    movies.sort(key=lambda x: -(x.get('popularity') or 0))

                if movies:
                    return movies, len(movies), 1

    # 2. Thử Discover theo Quốc gia / Ngôn ngữ hoặc Thể loại
    if lang or found_genre_id:
        params: Dict[str, Any] = {'page': page, 'sort_by': 'popularity.desc'}
        if lang:
            params['with_original_language'] = lang
        if found_genre_id:
            params['with_genres'] = str(found_genre_id)
        d_data = _get('/discover/movie', params)
        if d_data and d_data.get('results'):
            movies = [format_movie_record(m) for m in d_data['results']]
            total = d_data.get('total_results', len(movies))
            pages = min(d_data.get('total_pages', 1), 50)
            return movies, total, pages

    return [], 0, 0


def search_movies_by_title(query: str, page: int = 1) -> Tuple[List[Dict[str, Any]], int, int]:
    """Tìm kiếm phim theo tên qua TMDB Search API kèm phân tích câu truy vấn tự nhiên (smart search) và OMDB fallback."""
    query = query.strip()
    if not query:
        return get_trending_movies(page=page)

    data = _get("/search/movie", {"query": query, "page": page, "include_adult": "false"})
    if data and data.get("results"):
        movies = [format_movie_record(m) for m in data["results"]]
        total = data.get("total_results", len(movies))
        pages = min(data.get("total_pages", 1), 50)
        return movies, total, pages

    # Phân tích câu hỏi tự nhiên (Đạo diễn, Diễn viên, Quốc gia, Thể loại...)
    nl_movies, nl_total, nl_pages = resolve_natural_language_search(query, page=page)
    if nl_movies:
        return nl_movies, nl_total, nl_pages

    # Nếu TMDB không ra kết quả (ví dụ series hoạt hình hoặc keyword đặc thù), gọi OMDB
    omdb_results = search_omdb_fallback(query)
    if omdb_results:
        return omdb_results, len(omdb_results), 1

    # Fallback search locally
    matches = [m for m in FALLBACK_MOVIES if query.lower() in m["title"].lower()]
    return matches, len(matches), 1


def discover_by_genres(genre_names: List[str], sort_by: str = "popularity.desc", page: int = 1) -> Tuple[List[Dict[str, Any]], int, int]:
    """Lọc phim theo thể loại kết hợp qua /discover/movie."""
    all_genres = {g["name"].lower(): g["id"] for g in get_genres()}
    genre_ids = [str(all_genres[name.lower()]) for name in genre_names if name.lower() in all_genres]
    params: Dict[str, Any] = {"page": page, "sort_by": sort_by}
    if genre_ids:
        params["with_genres"] = ",".join(genre_ids)

    data = _get("/discover/movie", params)
    if data and "results" in data:
        movies = [format_movie_record(m) for m in data["results"]]
        total = data.get("total_results", len(movies))
        pages = min(data.get("total_pages", 1), 50)
        return movies, total, pages

    matches = [
        m for m in FALLBACK_MOVIES
        if not genre_names or all(g in m["genres"] for g in genre_names)
    ]
    return matches, len(matches), 1


def discover_by_genres_any(genre_names: List[str], sort_by: str = "popularity.desc", page: int = 1) -> Tuple[List[Dict[str, Any]], int, int]:
    """Khám phá phim khớp ít nhất một thể loại, phù hợp cho gợi ý theo tâm trạng."""
    all_genres = {g["name"].lower(): g["id"] for g in get_genres()}
    genre_ids = [str(all_genres[name.lower()]) for name in genre_names if name.lower() in all_genres]
    params: Dict[str, Any] = {
        "page": page,
        "sort_by": sort_by,
        "vote_count.gte": 50,
        "include_adult": "false",
    }
    if genre_ids:
        params["with_genres"] = "|".join(genre_ids)

    data = _get("/discover/movie", params)
    if data and "results" in data:
        movies = [format_movie_record(m) for m in data["results"]]
        total = data.get("total_results", len(movies))
        pages = min(data.get("total_pages", 1), 50)
        return movies, total, pages

    matches = [
        movie for movie in FALLBACK_MOVIES
        if not genre_names or set(movie.get("genres", [])).intersection(genre_names)
    ]
    return matches, len(matches), 1


def get_movie_detail(movie_id: int) -> Optional[Dict[str, Any]]:
    """Lấy thông tin chi tiết đầy đủ của phim từ TMDB API hoặc fallback."""
    data = _get(
        f"/movie/{movie_id}",
        {"append_to_response": "credits,keywords,videos,similar,recommendations,external_ids,watch/providers"},
    )
    if data and data.get("id") and data.get("success") is not False:
        return format_movie_record(data)

    for m in FALLBACK_MOVIES:
        if m["id"] == movie_id:
            fallback = dict(m)
            fallback.setdefault("tmdb_url", f"https://www.themoviedb.org/movie/{movie_id}")
            fallback.setdefault("sources", [{"label": "TMDB", "url": fallback["tmdb_url"], "kind": "database"}])
            fallback.setdefault("watch_providers", [])
            return fallback
    return None


def fetch_candidate_movies(movie_id: int, target_movie: Optional[Dict[str, Any]] = None, limit: int = 40) -> List[Dict[str, Any]]:
    """
    Truy xuất tập 20–40 phim ứng viên tiềm năng từ TMDB API:
    - Gọi song song /movie/{id}/recommendations và /movie/{id}/similar bằng ThreadPool
    - /discover/movie theo thể loại tương đồng nếu cần
    - Làm giàu đa luồng (parallel enrichment) cho tối đa 8 ứng viên cốt lõi
    """
    candidates_dict: Dict[int, Dict[str, Any]] = {}

    # 1. Gọi song song recommendations và similar
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_rec = executor.submit(_get, f"/movie/{movie_id}/recommendations")
        fut_sim = executor.submit(_get, f"/movie/{movie_id}/similar")
        rec_data = fut_rec.result()
        sim_data = fut_sim.result()

    if rec_data and "results" in rec_data:
        for item in rec_data["results"]:
            cid = item.get("id")
            if cid and cid != movie_id:
                candidates_dict[cid] = format_movie_record(item)

    if sim_data and "results" in sim_data:
        for item in sim_data["results"]:
            cid = item.get("id")
            if cid and cid != movie_id and cid not in candidates_dict:
                candidates_dict[cid] = format_movie_record(item)

    if len(candidates_dict) < 15 and target_movie and target_movie.get("genres"):
        disc_movies, _, _ = discover_by_genres(target_movie["genres"][:2], page=1)
        for m in disc_movies:
            if m["id"] != movie_id and m["id"] not in candidates_dict:
                candidates_dict[m["id"]] = m

    if not candidates_dict:
        for m in FALLBACK_MOVIES:
            if m["id"] != movie_id:
                candidates_dict[m["id"]] = m

    candidate_list = list(candidates_dict.values())[:limit]

    # 2. Làm giàu song song metadata (cast, director) cho tối đa 8 ứng viên cốt lõi
    candidates_to_enrich = [
        m for m in candidate_list if not m.get("cast") or not m.get("director")
    ][:8]

    if candidates_to_enrich:
        enriched_map: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_id = {
                executor.submit(get_movie_detail, m["id"]): m["id"]
                for m in candidates_to_enrich
            }
            for fut in as_completed(future_to_id):
                mid = future_to_id[fut]
                try:
                    res = fut.result()
                    if res:
                        enriched_map[mid] = res
                except Exception:
                    pass

        enriched_list = []
        for m in candidate_list:
            if m["id"] in enriched_map:
                enriched_list.append(enriched_map[m["id"]])
            else:
                enriched_list.append(m)
        return enriched_list

    return candidate_list
