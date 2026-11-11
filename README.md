# CineMatch — Hệ thống gợi ý phim thông minh

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![AI](https://img.shields.io/badge/AI-ONNX%20MiniLM-7C3AED)
![License](https://img.shields.io/badge/License-Educational-0F766E)

CineMatch là đồ án trí tuệ nhân tạo về tìm kiếm, khám phá và gợi ý phim. Ứng dụng kết hợp dữ liệu thời gian thực từ TMDB với mô hình ngôn ngữ đa ngôn ngữ MiniLM chạy bằng ONNX Runtime để hiểu mô tả bằng tiếng Việt hoặc tiếng Anh.

## Tính năng chính

- Gợi ý phim bằng mô hình hybrid: semantic similarity, TF-IDF, metadata và điểm chất lượng.
- Tìm kiếm theo tên, mô tả tự nhiên, cảm xúc và hành trình xem phim.
- Giải thích điểm phù hợp của từng kết quả gợi ý.
- Trang quản trị cho phim ngắn AI và phim mới chưa có trên TMDB.
- Chế độ dữ liệu dự phòng để có thể chạy thử khi chưa cấu hình TMDB.
- Bộ đánh giá NDCG, Precision, tính nhất quán thể loại và độ đa dạng.

## Chạy dự án bằng VS Code

### Yêu cầu

- Python 3.12 trở lên
- Visual Studio Code
- MySQL 8 nếu cần dùng trang quản trị

Mở thư mục dự án bằng VS Code, sau đó chạy các lệnh sau trong Terminal:

```powershell
cd app
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Mở <http://127.0.0.1:5000> trong trình duyệt.

Ứng dụng có thể chạy bằng dữ liệu dự phòng. Để dùng danh mục phim TMDB đầy đủ, điền `TMDB_API_KEY` vào `app/.env`. Để bật trang quản trị, điền thêm các biến `DB_*` và `SECRET_KEY` trong cùng file.

### Bật tìm kiếm AI ngữ nghĩa

Mô hình ONNX có dung lượng lớn nên không lưu trong Git. Tải mô hình và tạo vector phim bằng lệnh:

```powershell
cd app
python setup_ai.py
```

### Chạy kiểm thử

```powershell
cd app
python -m unittest -v
```

## Cấu trúc dự án

```text
goiytimphim/
├── app/                   # Mã nguồn Flask, giao diện và kiểm thử
│   ├── data/              # Dataset TMDB dùng cho chế độ dự phòng
│   ├── evaluation/        # Kết quả và báo cáo đánh giá mô hình
│   ├── static/            # CSS và tài nguyên giao diện
│   ├── templates/         # Jinja templates
│   ├── app.py             # Flask application factory và routes
│   ├── recommender.py     # Hybrid recommendation engine
│   ├── semantic.py        # ONNX MiniLM encoder
│   └── tmdb_service.py    # TMDB API, cache và dữ liệu dự phòng
├── BaoCao.pdf             # Báo cáo đồ án
├── Slide.pdf              # Slide thuyết trình
└── DEPLOY.md              # Hướng dẫn chạy và triển khai
```

Tài liệu kỹ thuật chi tiết nằm trong [app/README.md](app/README.md). Hướng dẫn triển khai nằm trong [DEPLOY.md](DEPLOY.md).

## Thuật toán gợi ý

Điểm cuối cùng được hợp nhất từ bốn thành phần:

```text
FinalScore = 0.40 × Semantic
           + 0.25 × Lexical
           + 0.25 × Metadata
           + 0.10 × Quality
```

Hệ thống lấy ứng viên từ TMDB hoặc dataset dự phòng, chấm điểm lại theo nội dung và metadata, áp dụng hình phạt khi lệch thể loại, rồi trả về kết quả kèm phần giải thích.

## Tài liệu đồ án

- [Báo cáo](BaoCao.pdf)
- [Slide thuyết trình](Slide.pdf)
- [Báo cáo đánh giá mô hình](app/evaluation/report.md)

## Lưu ý bảo mật

Không commit file `.env`, API key, mật khẩu cơ sở dữ liệu hoặc `SECRET_KEY`. Hãy sao chép `app/.env.example` thành `app/.env` và chỉ lưu thông tin thật ở máy cá nhân.
