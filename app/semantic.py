"""
@file semantic.py
@module semantic
@description Xử lý AI đa ngôn ngữ: Sentence Transformers (MiniLM), chạy ONNX Runtime trên CPU, hỗ trợ mã hóa thời gian thực (in-flight) và bộ nhớ đệm
"""
import hashlib
import json
from pathlib import Path
from threading import Lock
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
MODEL_FILE = 'onnx/model_quint8_avx2.onnx'
MODEL_DIR = ROOT / 'models' / 'multilingual-minilm'
INDEX_FILE = ROOT / 'models' / 'movie_embeddings.npz'


class AIUnavailable(RuntimeError):
    pass


def movie_to_text(movie: Dict[str, Any]) -> str:
    """Chuyển đổi thông tin phim thành chuỗi văn bản tự nhiên có cấu trúc cho AI Embedding."""
    title = movie.get('title', '')
    overview = movie.get('overview', '')
    genres = ', '.join(movie.get('genres', []))
    keywords = ', '.join(movie.get('keywords', []))
    tagline = movie.get('tagline', '')
    director = movie.get('director', '')
    parts = []
    if title:
        parts.append(title)
    if tagline:
        parts.append(tagline)
    if overview:
        parts.append(overview)
    if genres:
        parts.append(f"Genres: {genres}")
    if keywords:
        parts.append(f"Themes: {keywords}")
    if director:
        parts.append(f"Director: {director}")
    return ". ".join(parts)


def documents(movies):
    """Tạo chuỗi văn bản tài liệu cho danh sách phim."""
    if hasattr(movies, 'to_dict'):
        records = movies.to_dict('records')
    else:
        records = movies
    return [movie_to_text(m) for m in records]


def fingerprint(movies):
    """Tính dấu vân tay sha256 cho dữ liệu và cấu hình."""
    if hasattr(movies, 'to_dict'):
        ids = movies['id'].tolist()
    else:
        ids = [m['id'] for m in movies]
    payload = dict(model=MODEL_ID, revision=REVISION, file=MODEL_FILE,
                   max_length=128, pooling='masked-mean-l2-v1',
                   ids=ids, texts=documents(movies))
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class Encoder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        if not (MODEL_DIR / MODEL_FILE).exists():
            raise AIUnavailable('Chưa có mô hình AI. Chạy python setup_ai.py trong môi trường dự án.')
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL_DIR / MODEL_FILE),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.lock = Lock()
        self._text_cache: Dict[str, np.ndarray] = {}

    def encode(self, texts: List[str], batch_size: int = 16, progress: bool = False) -> np.ndarray:
        chunks = []
        with self.lock:
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer.encode_batch(texts[start:start+batch_size])
                values = dict(input_ids=np.array([x.ids for x in batch], dtype=np.int64),
                              attention_mask=np.array([x.attention_mask for x in batch], dtype=np.int64),
                              token_type_ids=np.array([x.type_ids for x in batch], dtype=np.int64))
                tokens = self.session.run(None, {k: v for k, v in values.items() if k in self.inputs})[0]
                mask = values['attention_mask'][..., None].astype(np.float32)
                vectors = (tokens * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                chunks.append(vectors.astype(np.float32))
                if progress and (start % 320 == 0 or start+batch_size >= len(texts)):
                    print(f'[AI][Encode] Encoded {min(start+batch_size, len(texts))}/{len(texts)}', flush=True)
        return np.vstack(chunks)

    def encode_text_cached(self, text: str) -> np.ndarray:
        """Mã hóa văn bản đơn lẻ có lưu cache trong bộ nhớ."""
        clean = text.strip()
        if clean in self._text_cache:
            return self._text_cache[clean]
        vec = self.encode([clean])[0]
        if len(self._text_cache) < 2000:
            self._text_cache[clean] = vec
        return vec

    def encode_movie(self, movie: Dict[str, Any]) -> np.ndarray:
        """Mã hóa một bộ phim thành vector 384 chiều."""
        text = movie_to_text(movie)
        return self.encode_text_cached(text)

    def encode_movies_cached(self, movies: List[Dict[str, Any]]) -> np.ndarray:
        """Mã hóa danh sách phim với In-Memory LRU Vector Cache, chỉ chạy ONNX cho phim mới."""
        if not movies:
            return np.zeros((0, 384), dtype=np.float32)

        texts = [movie_to_text(m) for m in movies]
        needed_indices = []
        needed_texts = []
        result_vecs = [None] * len(movies)

        with self.lock:
            for idx, text in enumerate(texts):
                if text in self._text_cache:
                    result_vecs[idx] = self._text_cache[text]
                else:
                    needed_indices.append(idx)
                    needed_texts.append(text)

        if needed_texts:
            new_vecs = self.encode(needed_texts)
            with self.lock:
                for idx_in_needed, orig_idx in enumerate(needed_indices):
                    vec = new_vecs[idx_in_needed]
                    result_vecs[orig_idx] = vec
                    if len(self._text_cache) < 2000:
                        self._text_cache[texts[orig_idx]] = vec

        return np.vstack(result_vecs)


_GLOBAL_ENCODER: Optional[Encoder] = None
_GLOBAL_ENCODER_LOCK = Lock()


def get_encoder() -> Encoder:
    """Singleton trả về phiên bản Encoder toàn cục."""
    global _GLOBAL_ENCODER
    with _GLOBAL_ENCODER_LOCK:
        if _GLOBAL_ENCODER is None:
            _GLOBAL_ENCODER = Encoder()
        return _GLOBAL_ENCODER


def compute_candidate_semantic_scores(
    target: Dict[str, Any],
    candidates: List[Dict[str, Any]]
) -> List[float]:
    """
    Tính điểm Cosine Similarity thời gian thực giữa phim đích và danh sách ứng viên (sử dụng vector cache).
    Trả về danh sách float từ 0.0 đến 1.0.
    """
    if not candidates:
        return []
    try:
        encoder = get_encoder()
        target_vec = encoder.encode_movie(target)  # shape (384,)
        candidate_vecs = encoder.encode_movies_cached(candidates)  # shape (N, 384)
        raw_scores = candidate_vecs @ target_vec
        # Cosine similarity nằm trong [-1, 1], chuẩn hóa về [0, 1]
        normalized = [max(0.0, min(1.0, float(s))) for s in raw_scores]
        return normalized
    except Exception as exc:
        print(f"[AI][Semantic] Error computing semantic scores: {exc}")
        return [0.0] * len(candidates)


def compute_query_semantic_scores(
    query: str,
    candidates: List[Dict[str, Any]]
) -> List[float]:
    """Tính điểm tương đồng giữa câu truy vấn tìm kiếm và danh sách ứng viên (sử dụng vector cache)."""
    if not query.strip() or not candidates:
        return []
    try:
        encoder = get_encoder()
        query_vec = encoder.encode_text_cached(query)
        candidate_vecs = encoder.encode_movies_cached(candidates)
        raw_scores = candidate_vecs @ query_vec
        return [max(0.0, min(1.0, float(s))) for s in raw_scores]
    except Exception as exc:
        print(f"[AI][Semantic] Error computing query scores: {exc}")
        return [0.0] * len(candidates)


class SemanticIndex:
    """Lớp chỉ mục ngữ nghĩa cho danh mục cố định (Backward Compatibility)."""
    def __init__(self, movies):
        try:
            with np.load(INDEX_FILE, allow_pickle=False) as saved:
                if str(saved['fingerprint']) != fingerprint(movies):
                    raise AIUnavailable('Dữ liệu đã thay đổi. Chạy python setup_ai.py để cập nhật AI.')
                self.vectors = saved['vectors']
            if self.vectors.shape != (len(movies), 384) or not np.isfinite(self.vectors).all():
                raise ValueError('Invalid embedding cache')
            self.encoder = get_encoder()
        except AIUnavailable:
            raise
        except Exception as exc:
            raise AIUnavailable('AI chưa sẵn sàng. Chạy python setup_ai.py; tìm theo tên vẫn sử dụng được.') from exc

    @lru_cache(maxsize=128)
    def query_scores(self, text):
        try:
            return self.vectors @ self.encoder.encode([text])[0]
        except Exception as exc:
            raise AIUnavailable('Không xử lý được mô tả bằng AI. Hãy thử lại hoặc tìm theo tên.') from exc

    def movie_scores(self, index):
        return self.vectors @ self.vectors[index]
