# Hướng dẫn chạy và triển khai CineMatch

## Chạy trong môi trường phát triển

```powershell
cd app
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Ứng dụng mặc định mở tại <http://127.0.0.1:5000>.

## Cấu hình môi trường

Các biến được đọc từ `app/.env`:

| Biến | Bắt buộc | Mục đích |
|---|---:|---|
| `TMDB_API_KEY` | Không | Dùng dữ liệu TMDB thời gian thực |
| `DB_HOST`, `DB_PORT` | Khi dùng quản trị | Máy chủ MySQL |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Khi dùng quản trị | Cơ sở dữ liệu CineMatch |
| `SECRET_KEY` | Khi triển khai | Ký Flask session |

Sau khi cấu hình MySQL, chạy:

```powershell
cd app
python check_database.py
python create_admin.py
```

## Chạy bằng Waitress

Không dùng Flask development server cho môi trường triển khai. Trên Windows có thể chạy:

```powershell
cd app
.\.venv\Scripts\waitress-serve.exe --listen=127.0.0.1:8000 app:app
```

Nếu công khai ứng dụng lên Internet, đặt reverse proxy HTTPS phía trước Waitress, cấu hình `SECRET_KEY` riêng và giới hạn quyền của tài khoản MySQL.

## Mô hình AI

Thư mục `app/models/` không được đưa lên Git vì mô hình ONNX vượt giới hạn dung lượng tệp của GitHub. Trên mỗi máy triển khai, chạy một lần:

```powershell
cd app
python setup_ai.py
```

Nếu chưa tải mô hình, ứng dụng tự chuyển sang phương pháp TF-IDF dự phòng.
