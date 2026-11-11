## Nhận xét sơ bộ từ nội dung dữ liệu

Các nhận xét dưới đây do trợ lý phân tích mô tả và metadata trong CSV, **không phải khảo sát người dùng, không phải bộ nhãn chuẩn**. Cần người xem xác nhận độc lập bằng phiếu chấm. Không dùng các ví dụ này để tự suy ra Precision@6.

| Trường hợp | Nhận xét và bằng chứng |
|---|---|
| The Dark Knight | Cả 6 kết quả đều là phim về Batman. The Dark Knight Rises và Batman Begins còn chung đạo diễn, diễn viên và nhiều từ khóa. Hợp lý về nhân vật/bối cảnh, nhưng ít khám phá phim ngoài loạt Batman. |
| The Conjuring | Danh sách gồm The Conjuring 2, Insidious, Insidious: Chapter 2, The Apparition, Grave Encounters và Ouija. Mô tả đều gắn với linh hồn/hiện tượng siêu nhiên; đây là trường hợp có tính nhất quán rõ. |
| Avatar | Aliens và Star Trek Into Darkness có cơ sở về khoa học viễn tưởng, không gian/người ngoài hành tinh. Falcon Rising có nhân vật cựu lính thủy đánh bộ nhưng bối cảnh tội phạm ở Brazil; đây là gợi ý đáng nghi nếu người dùng thích thế giới ngoài hành tinh của Avatar. |
| Titanic | Danh sách nghiêng về tàu thuyền/tai nạn (Raise the Titanic, Poseidon, Ghost Ship…), trong khi không gợi ý nào có nhãn Romance. Có thể hợp với người quan tâm thảm họa trên biển, nhưng chưa đáp ứng tốt người tìm chuyện tình tương tự. Supernova nói về tàu vũ trụ, càng cho thấy từ chung không đủ đảm bảo cùng trải nghiệm. |
| Toy Story | Hai phần tiếp theo đứng đầu là hợp lý. The 40 Year Old Virgin và Factory Girl có câu chuyện khác rõ rệt; tên Andy xuất hiện trong cả các mô tả và có thể tạo liên hệ từ vựng không đúng ngữ nghĩa. Không nên suy từ nhãn Comedy chung thành phù hợp với phim gia đình. |
| Whiplash | Drumline có cơ sở vì cùng kể về tay trống trẻ và con đường khẳng định năng lực. Hop cũng nhắc tới tay trống nhưng là câu chuyện hài/hoạt hình; Spawn và Kung Fu Jungle khác xa chủ đề âm nhạc trong mô tả. Đây là một nhóm cần cải thiện. |
| March of the Penguins | Hệ thống tìm được phim có chim cánh cụt nhưng trộn tài liệu với hoạt hình và phim nhại. Last Vegas chỉ chung diễn viên Morgan Freeman trong metadata, không cùng chủ đề thiên nhiên. Antarctic Edge có bối cảnh Nam Cực và là phim tài liệu, nên có cơ sở hơn về cách tiếp cận. |

### Kết luận dùng cho báo cáo

Trên mẫu nhỏ này, hệ thống thể hiện khả năng tìm các phim cùng loạt hoặc có chủ đề đặc trưng. Tuy nhiên, một số gợi ý bị ảnh hưởng bởi từ trùng, tên riêng hoặc người tham gia phim, dẫn đến lệch nội dung. Các tỷ lệ metadata cao hơn đối chứng phổ biến không đủ để kết luận mức hài lòng của người dùng. Cần hoàn tất phiếu chấm trước khi công bố Precision@6.

Sau khi chấm, có thể thử tăng vai trò thể loại/chủ đề hoặc giảm ảnh hưởng tên riêng trong mô tả. Những thay đổi đó cần được đánh giá trên cùng bộ mẫu và thêm một nhóm phim mới; không chỉ tối ưu cho vài ví dụ vừa phát hiện. Lần này chưa sửa mô hình để giữ mốc so sánh ban đầu.
