# CineMatch — Hệ thống Gợi ý & Khám phá Điện ảnh Thời gian thực (AI Hybrid Engine)

CineMatch là ứng dụng gợi ý và tìm kiếm phim thông minh thời gian thực, tích hợp trực tiếp **The Movie Database (TMDB) API** kết hợp mô hình AI đa ngôn ngữ **Sentence Transformers (ONNX Runtime CPU)**. Ứng dụng triển khai thuật toán **Two-Tier Real-Time Hybrid Re-ranking** với giao diện **The Aesthetic OS (Obsidian Cinema & Material 3 Expressive)**.

> **Lưu ý:** Dự án **không phụ thuộc vào dataset CSV tĩnh có sẵn**. Dữ liệu phim, poster HD, backdrop và bình chọn cộng đồng được cập nhật liên tục trực tiếp từ TMDB API, kèm bộ đệm **Offline Demo Fallback** giúp hệ thống hoạt động ổn định mọi lúc.

---

## 1. Hướng dẫn Chạy Ứng dụng

Mở Terminal trong thư mục dự án và chạy:

```powershell
# Cài đặt thư viện phụ thuộc (nếu chưa cài)
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Khởi chạy ứng dụng Web
.\venv\Scripts\python.exe app.py
```

Mở trình duyệt tại **http://127.0.0.1:5000**. Nhấn **Ctrl+C** tại Terminal để dừng server.

### Trang quản trị phim riêng

Trang quản trị tại **http://127.0.0.1:5000/admin** cho phép thêm, sửa, xuất bản và xóa phim ngắn AI hoặc phim mới chưa có trên TMDB. Phim đã xuất bản xuất hiện trên trang chủ, kết quả tìm kiếm và tập ứng viên của bộ gợi ý.

1. Điền `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` và `SECRET_KEY` trong `.env`.
2. Chạy `python check_database.py` để tạo/kiểm tra schema.
3. Chạy `python create_admin.py` để tạo tài khoản đăng nhập.

### Cấu hình TMDB API Key (Tùy chọn & Khuyên dùng)
Hệ thống có sẵn chế độ **Offline Demo Fallback** để bạn trải nghiệm ngay lập tức. Để mở khóa toàn bộ danh mục hơn 800,000+ phim của TMDB theo thời gian thực:
1. Đăng ký tài khoản miễn phí tại [themoviedb.org](https://www.themoviedb.org/signup).
2. Vào **Settings → API** để lấy **API Key (v3)** hoặc **API Read Access Token**.
3. Bấm nút **⚙ Cấu hình TMDB API Key** trên thanh điều hướng của Web App (hoặc thêm `TMDB_API_KEY=your_key` vào file `.env`).

---

## 2. Kiến trúc & Các Tính năng Nổi bật

### 2.1. Động cơ Two-Tier Real-Time Hybrid Re-ranking
* **Tầng 1 — Candidate Retrieval:** Truy xuất 20–40 phim ứng viên từ TMDB API (`/similar`, `/recommendations`, `/discover`).
* **Tầng 2 — AI Weighted Score Fusion:**
  $$\text{FinalScore} = 0.40 \cdot S_{\text{semantic}} + 0.25 \cdot S_{\text{lexical}} + 0.25 \cdot S_{\text{metadata}} + 0.10 \cdot S_{\text{quality}}$$
  * $S_{\text{semantic}}$: Cosine Similarity từ mô hình ngôn ngữ ONNX MiniLM (384 chiều) tính toán thời gian thực on-the-fly.
  * $S_{\text{lexical}}$: Tương đồng từ khóa/mô tả TF-IDF trên tập ứng viên.
  * $S_{\text{metadata}}$: Độ trùng khớp thể loại (Jaccard) và đoàn làm phim (đạo diễn/diễn viên).
  * $S_{\text{quality}}$: Điểm chất lượng uy tín Bayesian Weighted Rating theo tiêu chuẩn IMDb ($v / (v + m) \cdot R + m / (v + m) \cdot C$).
* **Khử nhiễu với Soft-Penalty Thể loại:** Nếu phim ứng viên hoàn toàn không chia sẻ thể loại cốt lõi nào với phim mẫu, điểm tương đồng sẽ chịu hệ số phạt $\times 0.40$, loại bỏ hoàn toàn hiện tượng lệch thể loại.
* **Explainability Engine:** Mỗi gợi ý đều hiển thị huy hiệu Match Score (%) cùng bảng phân rã chi tiết 3 yếu tố (🎯 Cốt truyện, 🎬 Thể loại, ⭐ Đánh giá cộng đồng).

### 2.2. Tìm kiếm Kép Thông minh (Dual Search)
* **Theo Tên Phim:** Tìm kiếm thời gian thực qua TMDB Search API.
* **AI — Mô tả Ngữ nghĩa:** Nhập mô tả cốt truyện bằng **tiếng Việt** hoặc tiếng Anh (ví dụ: *"phim về du hành không gian và lỗ sâu vũ trụ"*, *"hoạt hình về đồ chơi biết nói"*). AI sẽ mã hóa câu truy vấn và xếp hạng theo khoảng cách vector ngữ nghĩa.

### 2.3. Dynamic Cinema Explorer (Trang chủ)
* Hàng phim **Thịnh Hành Hôm Nay (Trending Today)**.
* Hàng phim **Được Đánh Giá Cao Nhất (Top Rated)**.
* Hàng phim **Đang Chiếu Rạp (Now Playing)**.

---

## 3. Cấu trúc Mã nguồn

| File | Chức năng |
|---|---|
| `app.py` | Web Controller Flask, xử lý định tuyến, Explorer data, API Key settings, JSON API |
| `tmdb_service.py` | Kết nối TMDB API v3, quản lý API key, in-memory cache, poster CDN và fallback demo |
| `recommender.py` | Two-Tier Real-Time Hybrid Re-ranking Engine, Soft-penalty, phân tầng metadata |
| `semantic.py` | ONNX MiniLM Sentence Transformers, mã hóa vector 384 chiều thời gian thực trên CPU |
| `evaluate.py` | Bộ đo lường khoa học đa chiều: NDCG@6, Precision@6, Genre Consistency, Diversity |
| `templates/index.html` | Giao diện SSR với Dynamic Explorer, Explainability breakdown, Modal cài đặt API key |
| `static/style.css` | Hệ thống Design System The Aesthetic OS (Obsidian Cinema, M3 Tonal Depth, `--sp-*` tokens) |
| `test_app.py` | Bộ kiểm thử đơn vị cho Web Routes, API endpoints, Hybrid Recommender |
| `test_ai.py` | Bộ kiểm thử cho ONNX vector encoding, Semantic search đa ngôn ngữ |

---

## 4. Kiểm thử & Đánh giá Chất lượng

### Chạy Unit Tests tự động
```powershell
.\venv\Scripts\python.exe -m unittest -v
```

### Chạy Đánh giá Thực nghiệm (Benchmark & Report)
```powershell
.\venv\Scripts\python.exe evaluate.py
```
Kết quả đo lường và so sánh đối đầu giữa 3 phương pháp (*Baseline TF-IDF*, *Pure AI Semantic*, *Two-Tier Hybrid*) được ghi nhận tại [evaluation/report.md](evaluation/report.md) và [evaluation/results.json](evaluation/results.json).
