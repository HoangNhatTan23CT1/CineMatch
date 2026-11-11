"""Xây dựng Hành trình cảm xúc và đơn phim cá nhân hóa cho CineMatch."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any, Dict, List, Tuple

import tmdb_service
from recommender import MOOD_PROFILES, resolve_mood_key
from semantic import compute_query_semantic_scores


JOURNEY_DESTINATIONS: Dict[str, Dict[str, Any]] = {
    "calm": {
        "label": "Nhẹ lòng", "emoji": "🌤️",
        "prompt": "nhẹ nhàng, bình yên, chữa lành, ấm áp, dễ chịu và kết thúc có hy vọng",
        "genres": ["Family", "Animation", "Comedy", "Romance"],
        "description": "Khép lại buổi xem với cảm giác bình tâm và được an ủi.",
    },
    "joy": {
        "label": "Vui hơn", "emoji": "😄",
        "prompt": "vui vẻ, hài hước, tràn năng lượng, feel-good, tình bạn và kết thúc hạnh phúc",
        "genres": ["Comedy", "Animation", "Family", "Music"],
        "description": "Nạp thêm tiếng cười và năng lượng tích cực.",
    },
    "hope": {
        "label": "Có hy vọng", "emoji": "✨",
        "prompt": "truyền hy vọng, cơ hội thứ hai, vượt qua mất mát, chữa lành và hướng về tương lai",
        "genres": ["Drama", "Family", "Adventure"],
        "description": "Tìm lại niềm tin qua những câu chuyện về cơ hội thứ hai.",
    },
    "motivation": {
        "label": "Có động lực", "emoji": "🔥",
        "prompt": "truyền cảm hứng, nghị lực, theo đuổi ước mơ, vượt khó và chinh phục mục tiêu",
        "genres": ["Drama", "Adventure", "History", "Documentary"],
        "description": "Kết thúc với thôi thúc muốn đứng dậy và bắt đầu hành động.",
    },
    "connection": {
        "label": "Được kết nối", "emoji": "🤝",
        "prompt": "tình bạn, gia đình, cộng đồng, sự đồng hành, thuộc về và kết nối chân thành",
        "genres": ["Drama", "Comedy", "Romance", "Family"],
        "description": "Cảm nhận sự đồng hành qua tình bạn, gia đình và cộng đồng.",
    },
    "wonder": {
        "label": "Muốn khám phá", "emoji": "🌌",
        "prompt": "phiêu lưu kỳ thú, khám phá thế giới mới, giàu trí tưởng tượng, sử thi và bất ngờ",
        "genres": ["Adventure", "Fantasy", "Science Fiction", "Animation"],
        "description": "Rời màn hình với trí tưởng tượng rộng mở và cảm giác háo hức.",
    },
}


COMPANY_OPTIONS: Dict[str, Dict[str, Any]] = {
    "alone": {
        "label": "Một mình", "emoji": "🎧",
        "prompt": "phù hợp để xem một mình, giàu chiều sâu và có không gian suy ngẫm",
        "genres": ["Drama", "Animation", "Science Fiction"],
    },
    "couple": {
        "label": "Cùng người thương", "emoji": "💞",
        "prompt": "phù hợp cho hai người, giàu cảm xúc, gần gũi và có điều để trò chuyện",
        "genres": ["Romance", "Comedy", "Drama"],
    },
    "friends": {
        "label": "Cùng bạn bè", "emoji": "🍿",
        "prompt": "giải trí cuốn hút, vui nhộn, nhiều khoảnh khắc để cùng phản ứng và bàn luận",
        "genres": ["Comedy", "Action", "Adventure", "Mystery"],
    },
    "family": {
        "label": "Cùng gia đình", "emoji": "🏠",
        "prompt": "ấm áp, dễ xem cho gia đình, tình thân, hài hước và giàu giá trị tích cực",
        "genres": ["Family", "Animation", "Comedy", "Adventure"],
    },
}


INTENSITY_OPTIONS: Dict[str, Dict[str, Any]] = {
    "gentle": {
        "label": "Êm dịu", "emoji": "🫧",
        "prompt": "nhịp phim nhẹ nhàng, ít căng thẳng, dễ xem",
        "boost": ["Family", "Animation", "Comedy", "Romance"],
        "avoid": ["Horror", "Thriller", "War", "Crime"],
    },
    "balanced": {
        "label": "Cân bằng", "emoji": "🌗",
        "prompt": "cảm xúc cân bằng, có cao trào vừa phải và kết thúc thỏa mãn",
        "boost": ["Drama", "Adventure", "Comedy"],
        "avoid": [],
    },
    "intense": {
        "label": "Mạnh mẽ", "emoji": "⚡",
        "prompt": "cao trào mạnh, kịch tính, cuốn hút và giàu cảm xúc",
        "boost": ["Thriller", "Action", "Mystery", "Adventure"],
        "avoid": [],
    },
}


DURATION_OPTIONS = [
    {"value": 90, "label": "Khoảng 90 phút"},
    {"value": 120, "label": "Khoảng 2 giờ"},
    {"value": 180, "label": "Khoảng 3 giờ"},
    {"value": 240, "label": "Một buổi tối · 4 giờ"},
    {"value": 300, "label": "Movie marathon · 5 giờ"},
]


def journey_form_options() -> Dict[str, Any]:
    moods = [dict({"key": key}, **profile) for key, profile in MOOD_PROFILES.items()]
    destinations = [dict({"key": key}, **profile) for key, profile in JOURNEY_DESTINATIONS.items()]
    companies = [dict({"key": key}, **profile) for key, profile in COMPANY_OPTIONS.items()]
    intensities = [dict({"key": key}, **profile) for key, profile in INTENSITY_OPTIONS.items()]
    return {
        "moods": moods,
        "destinations": destinations,
        "companies": companies,
        "intensities": intensities,
        "durations": DURATION_OPTIONS,
    }


def _enrich_candidate(movie: Dict[str, Any]) -> Dict[str, Any]:
    movie_id = int(movie.get("id") or 0)
    if movie_id <= 0:
        enriched = dict(movie)
    else:
        detail = tmdb_service.get_movie_detail(movie_id)
        enriched = dict(movie)
        if detail:
            enriched.update(detail)
    enriched["ai_score"] = movie.get("ai_score", enriched.get("ai_score", 0.0))
    return enriched


def _content_notes(movie: Dict[str, Any]) -> List[str]:
    genres = set(movie.get("genres", []))
    notes = []
    if "Horror" in genres:
        notes.append("Không khí kinh dị")
    if genres.intersection({"Thriller", "Crime", "Mystery"}):
        notes.append("Có tình tiết căng thẳng")
    if genres.intersection({"War", "Action"}):
        notes.append("Có xung đột hoặc hành động")
    if "Drama" in genres:
        notes.append("Cảm xúc sâu")
    return notes[:2]


def _select_path(candidates: List[Dict[str, Any]], duration: int) -> List[Dict[str, Any]]:
    desired_count = 1 if duration < 170 else (2 if duration < 270 else 3)
    pool = candidates[:18]
    tolerance = 15

    for count in range(desired_count, 0, -1):
        best_combo: Tuple[Dict[str, Any], ...] | None = None
        best_score = -1.0
        for combo in combinations(pool, count):
            total_runtime = sum(int(movie.get("runtime") or 105) for movie in combo)
            if total_runtime > duration + tolerance:
                continue
            score = sum(float(movie.get("journey_score", 0.0)) for movie in combo) / count
            time_fit = 1.0 - abs(duration - total_runtime) / max(duration, 1)
            genre_sets = [set(movie.get("genres", [])) for movie in combo]
            diversity = len(set().union(*genre_sets)) / max(1, sum(len(item) for item in genre_sets))
            combo_score = score + max(0.0, time_fit) * 0.12 + diversity * 0.04
            if combo_score > best_score:
                best_score = combo_score
                best_combo = combo
        if best_combo:
            return list(best_combo)

    return pool[:1]


def build_journey(
    model: Any,
    current_mood: str,
    target_mood: str,
    duration: int,
    company: str,
    intensity: str,
) -> Dict[str, Any]:
    """Tạo một đơn phim có đường cong cảm xúc, thời lượng và bối cảnh xem cụ thể."""
    current_key = resolve_mood_key(current_mood) or "sad"
    target_key = target_mood if target_mood in JOURNEY_DESTINATIONS else "calm"
    company_key = company if company in COMPANY_OPTIONS else "alone"
    intensity_key = intensity if intensity in INTENSITY_OPTIONS else "balanced"
    duration = min(300, max(90, int(duration or 120)))

    current = dict({"key": current_key}, **MOOD_PROFILES[current_key])
    target = dict({"key": target_key}, **JOURNEY_DESTINATIONS[target_key])
    company_profile = dict({"key": company_key}, **COMPANY_OPTIONS[company_key])
    intensity_profile = dict({"key": intensity_key}, **INTENSITY_OPTIONS[intensity_key])
    context = ". ".join([target["prompt"], company_profile["prompt"], intensity_profile["prompt"]])

    base_movies, _, _, _, _ = model.search_by_mood(
        current_key, context=context, page=1, page_size=42
    )
    if not base_movies:
        return {
            "current": current, "target": target, "company": company_profile,
            "intensity": intensity_profile, "duration": duration, "movies": [],
            "total_duration": 0,
        }

    # Danh sách TMDB không có runtime/credits. Làm giàu nhóm đầu song song để
    # đơn phim có thể thực sự tôn trọng quỹ thời gian và hiển thị nguồn xem.
    enriched_map: Dict[int, Dict[str, Any]] = {}
    to_enrich = base_movies[:16]
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {executor.submit(_enrich_candidate, movie): movie for movie in to_enrich}
        for future in as_completed(future_map):
            original = future_map[future]
            try:
                enriched = future.result()
            except Exception:
                enriched = dict(original)
            enriched_map[int(original["id"])] = enriched

    candidates = [enriched_map.get(int(movie["id"]), dict(movie)) for movie in to_enrich]
    target_scores = compute_query_semantic_scores(target["prompt"], candidates)
    company_genres = set(company_profile["genres"])
    target_genres = set(target["genres"])
    boost_genres = set(intensity_profile["boost"])
    avoid_genres = set(intensity_profile["avoid"])

    for index, movie in enumerate(candidates):
        genres = set(movie.get("genres", []))
        base_score = float(movie.get("ai_score", 0.0))
        target_score = float(target_scores[index]) if index < len(target_scores) else 0.0
        target_genre_score = min(1.0, len(genres.intersection(target_genres)) / 2.0)
        company_score = 1.0 if genres.intersection(company_genres) else 0.0
        intensity_score = 1.0 if genres.intersection(boost_genres) else 0.0
        penalty = 0.14 if genres.intersection(avoid_genres) else 0.0
        runtime = int(movie.get("runtime") or 105)
        runtime_fit = 1.0 if runtime <= duration + 15 else 0.0
        score = (
            base_score * 0.38 + target_score * 0.32 + target_genre_score * 0.12
            + company_score * 0.07 + intensity_score * 0.05 + runtime_fit * 0.06 - penalty
        )
        movie["runtime"] = runtime
        movie["target_score"] = round(max(0.0, min(1.0, target_score)), 3)
        movie["journey_score"] = round(max(0.0, min(1.0, score)), 3)
        movie["journey_match"] = round(max(0.0, min(100.0, score * 100)))
        movie["content_notes"] = _content_notes(movie)

    candidates.sort(key=lambda movie: (
        -movie["journey_score"], -movie.get("vote_average", 0), -movie.get("vote_count", 0)
    ))
    selected = _select_path(candidates, duration)
    selected.sort(key=lambda movie: movie.get("target_score", 0.0))

    stage_labels = {
        1: [("Tác phẩm phù hợp nhất", "Một lựa chọn cô đọng đưa bạn đến cảm xúc mong muốn trong quỹ thời gian hiện có.")],
        2: [
            ("1 · Đồng cảm", "Bắt đầu bằng một câu chuyện đủ gần với cảm xúc hiện tại để bạn không phải vội vàng thoát khỏi nó."),
            ("2 · Chuyển biến", "Khép lại bằng tác phẩm kéo cảm xúc về đúng điểm đến bạn đã chọn."),
        ],
        3: [
            ("1 · Đồng cảm", "Tác phẩm đầu tiên tạo không gian để nhận diện và ở cùng cảm xúc hiện tại."),
            ("2 · Chuyển nhịp", "Phần giữa mở rộng góc nhìn và bắt đầu thay đổi năng lượng của hành trình."),
            ("3 · Điểm đến", "Tác phẩm cuối củng cố cảm giác bạn muốn mang theo sau buổi xem."),
        ],
    }
    for movie, (stage, reason) in zip(selected, stage_labels[len(selected)]):
        movie["journey_stage"] = stage
        movie["journey_reason"] = reason

    total_duration = sum(int(movie.get("runtime") or 0) for movie in selected)
    return {
        "current": current,
        "target": target,
        "company": company_profile,
        "intensity": intensity_profile,
        "duration": duration,
        "movies": selected,
        "total_duration": total_duration,
        "remaining_duration": max(0, duration - total_duration),
        "headline": f"Từ {current['label'].lower()} đến {target['label'].lower()}",
        "summary": f"Một hành trình gồm {len(selected)} tác phẩm, được xếp theo nhịp cảm xúc và quỹ thời gian của bạn.",
    }
