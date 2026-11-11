"""Ghi kết quả AI thực tế để chấm thủ công, không tự gán nhãn đúng/sai."""
import json
from pathlib import Path
from time import perf_counter
from recommender import MovieRecommender
from semantic import MODEL_ID, REVISION, fingerprint

QUERIES = [
    'phim về người mắc kẹt ngoài không gian',
    'phim về du hành thời gian',
    'phim về tình yêu trên một con tàu bị chìm',
    'phim hoạt hình về những món đồ chơi biết nói',
    'phim về một gia đình sống trong ngôi nhà bị ma ám',
    'phim về siêu anh hùng bảo vệ thành phố',
    'phim về một vụ cướp ngân hàng',
    'phim về điệp viên thực hiện nhiệm vụ bí mật',
    'phim về người sống sót trong đại dịch zombie',
    'phim về một tay trống trẻ theo đuổi ước mơ âm nhạc',
    'phim tài liệu về chim cánh cụt ở Nam Cực',
    'phim về mafia và gia đình tội phạm',
    'phim hài về nhóm bạn đi du lịch',
    'phim về con người chiến đấu với người ngoài hành tinh',
    'phim về cuộc phiêu lưu trong thế giới phép thuật',
    'phim về người bị bỏ lại trên hoang đảo',
    'phim về võ sĩ luyện tập để thi đấu quyền anh',
    'phim về học sinh trung học và tình bạn',
    'phim về cuộc vượt ngục',
    'phim về thám tử điều tra kẻ giết người hàng loạt',
]


def main():
    from evaluate import SAMPLE
    db = MovieRecommender()
    db.semantic()
    queries = []
    for query in QUERIES:
        start = perf_counter()
        movies, _, _, _ = db.search_ai(query, page_size=6)
        queries.append(dict(query=query, seconds=round(perf_counter()-start, 3),
                            results=[dict(id=m['id'], title=m['title'], score=m['ai_score']) for m in movies]))
    comparisons = []
    for title, _ in SAMPLE:
        source = db.title_matches(title)[0]
        comparisons.append(dict(title=title, tfidf=[m['title'] for m in db.recommend(source['id'])],
                                ai=[m['title'] for m in db.recommend(source['id'], method='ai')]))
    output = Path(__file__).parent/'evaluation'
    output.mkdir(exist_ok=True)
    result = dict(model=MODEL_ID, revision=REVISION, fingerprint=fingerprint(db.movies),
                  note='Chưa chấm độ phù hợp. Thời gian đo trên máy hiện tại sau khi đã nạp mô hình.',
                  queries=queries, comparisons=comparisons)
    (output/'ai_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# Kiểm tra AI tìm kiếm và gợi ý phim', '',
             'Mô hình đa ngôn ngữ MiniLM, bản ONNX lượng tử hóa chạy CPU. Dữ liệu gồm 4.803 phim.',
             'Đây là kết quả chạy thật, chưa có nhãn phù hợp của người chấm nên chưa kết luận AI chính xác hơn TF-IDF. Không lọc thể loại/chủ đề trong các ví dụ dưới đây.', '',
             '## 20 câu tìm kiếm tiếng Việt', '', '| Câu tìm | Top 6 theo thứ tự | Giây (đã nạp mô hình) |', '|---|---|---:|']
    for item in queries:
        lines.append(f"| {item['query']} | {'; '.join(m['title'] for m in item['results'])} | {item['seconds']} |")
    lines += ['', '## So sánh gợi ý trên cùng 12 phim', '', '| Phim mẫu | TF-IDF: top 6 | AI: top 6 |', '|---|---|---|']
    for item in comparisons:
        lines.append(f"| {item['title']} | {'; '.join(item['tfidf'])} | {'; '.join(item['ai'])} |")
    lines += ['', 'Chấm độc lập mức phù hợp cho từng gợi ý để so sánh Precision@6. Cùng một câu tìm có thể có nhiều phim phù hợp; không lấy độ tương đồng của mô hình làm nhãn đánh giá.',
              'Hạn chế: dữ liệu tĩnh; mô hình đọc tối đa 128 token mỗi phim; có thể hiểu sai phủ định, tên riêng, hoặc mô tả quá chung. Chế độ AI trả tối đa 60 phim có cosine dương, không có ngưỡng đảm bảo liên quan. Dùng các bộ lọc để chỉ định yêu cầu bắt buộc.', '',
              'Chạy lại: `python evaluate_ai.py`. Không sửa báo cáo đánh giá TF-IDF trước đó.']
    (output/'ai_report.md').write_text('\n'.join(lines),encoding='utf-8')
    for item in queries[:5]:
        print(json.dumps(item,ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
