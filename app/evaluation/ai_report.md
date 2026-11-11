# Kiểm tra AI tìm kiếm và gợi ý phim

Mô hình đa ngôn ngữ MiniLM, bản ONNX lượng tử hóa chạy CPU. Dữ liệu gồm 4.803 phim.
Đây là kết quả chạy thật, chưa có nhãn phù hợp của người chấm nên chưa kết luận AI chính xác hơn TF-IDF. Không lọc thể loại/chủ đề trong các ví dụ dưới đây.

## 20 câu tìm kiếm tiếng Việt

| Câu tìm | Top 6 theo thứ tự | Giây (đã nạp mô hình) |
|---|---|---:|
| phim về người mắc kẹt ngoài không gian | The Inhabited Island; Panic Room; Fortress; Cube; The Brave Little Toaster; Cloverfield | 0.391 |
| phim về du hành thời gian | Looper; Timecrimes; Project Almanac; Time Changer; Hot Tub Time Machine 2; The Visitors | 0.362 |
| phim về tình yêu trên một con tàu bị chìm | Vessel; Poseidon; Triangle; Abandoned; Swept Away; The Lovers | 0.366 |
| phim hoạt hình về những món đồ chơi biết nói | The Lego Movie; Jonah: A VeggieTales Movie; A LEGO Brickumentary; Toy Story 3; Stuart Little; Gremlins | 0.355 |
| phim về một gia đình sống trong ngôi nhà bị ma ám | Sinister 2; The Amityville Horror; Insidious; Darkness; Sinister; George Washington | 0.365 |
| phim về siêu anh hùng bảo vệ thành phố | Batman v Superman: Dawn of Justice; The Shadow; Megamind; All Superheroes Must Die; Batman; Defendor | 0.357 |
| phim về một vụ cướp ngân hàng | The Sting; Half Past Dead; Gangster's Paradise: Jerusalema; Albino Alligator; Absolute Power; Small Time Crooks | 0.374 |
| phim về điệp viên thực hiện nhiệm vụ bí mật | Point Break; Get Smart; The Last Shot; Now You See Me; The Interpreter; Half Past Dead | 0.356 |
| phim về người sống sót trong đại dịch zombie | Dawn of the Dead; Night of the Living Dead; The Walking Deceased; Diary of the Dead; Survival of the Dead; Day of the Dead | 0.354 |
| phim về một tay trống trẻ theo đuổi ước mơ âm nhạc | Drumline; Whiplash; Hop; Rudderless; The Pianist; Jersey Boys | 0.344 |
| phim tài liệu về chim cánh cụt ở Nam Cực | Farce of the Penguins; March of the Penguins; Dolphins and Whales: Tribes of the Ocean; Oceans; The Crocodile Hunter: Collision Course; Big Miracle | 0.359 |
| phim về mafia và gia đình tội phạm | Gangster's Paradise: Jerusalema; Half Past Dead; Albino Alligator; Donnie Brasco; Things to Do in Denver When You're Dead; This Thing of Ours | 0.346 |
| phim hài về nhóm bạn đi du lịch | Karachi se Lahore; Road Trip; 50/50; Losin' It; Date Movie; The Broken Hearts Club: A Romantic Comedy | 0.348 |
| phim về con người chiến đấu với người ngoài hành tinh | Meet Dave; U.F.O.; Independence Daysaster; Galaxy Quest; Space Pirate Captain Harlock; Attack the Block | 0.382 |
| phim về cuộc phiêu lưu trong thế giới phép thuật | The Prestige; Into the Woods; The Immigrant; The Chronicles of Narnia: The Voyage of the Dawn Treader; Stardust; Dragonslayer | 0.378 |
| phim về người bị bỏ lại trên hoang đảo | Oceans; Against the Wild; Virgin Territory; Meek's Cutoff; The Hills Have Eyes; Madadayo | 0.363 |
| phim về võ sĩ luyện tập để thi đấu quyền anh | Never Back Down 2: The Beatdown; The Fighter; Kung Pow: Enter the Fist; Kung Fu Jungle; Street Fighter; Fighting Tommy Riley | 0.379 |
| phim về học sinh trung học và tình bạn | Love and Other Catastrophes; The Spectacular Now; I Know What You Did Last Summer; The Dirties; Hoop Dreams; The Real Cancun | 0.368 |
| phim về cuộc vượt ngục | The Act of Killing; Disaster Movie; The Hotel New Hampshire; The Last Exorcism Part II; Scary Movie 2; The Baader Meinhof Complex | 0.345 |
| phim về thám tử điều tra kẻ giết người hàng loạt | The Act of Killing; Se7en; The Face of an Angel; Scream 3; Suspect Zero; Murder by Numbers | 0.368 |

## So sánh gợi ý trên cùng 12 phim

| Phim mẫu | TF-IDF: top 6 | AI: top 6 |
|---|---|---|
| Avatar | Aliens; Battle: Los Angeles; Falcon Rising; Apollo 18; Star Trek Into Darkness; Titan A.E. | Meet Dave; Aliens; Star Trek Beyond; Star Trek: Insurrection; Serenity; Krull |
| Titanic | Raise the Titanic; Ghost Ship; Poseidon; Pirates of the Caribbean: On Stranger Tides; In the Heart of the Sea; Supernova | Message in a Bottle; The Chambermaid on the Titanic; Cargo; Vessel; Lake Mungo; Donovan's Reef |
| Toy Story | Toy Story 3; Toy Story 2; The 40 Year Old Virgin; For Your Consideration; Factory Girl; Heartbeeps | Toy Story 3; Toy Story 2; The 40 Year Old Virgin; A Christmas Story; Real Steel; A.I. Artificial Intelligence |
| The Dark Knight | The Dark Knight Rises; Batman Returns; Batman Begins; Batman: The Dark Knight Returns, Part 2; Batman Forever; Batman | The Dark Knight Rises; Batman Returns; Batman v Superman: Dawn of Justice; Batman & Robin; Batman: The Dark Knight Returns, Part 2; Batman |
| The Conjuring | The Conjuring 2; Insidious: Chapter 2; The Apparition; Grave Encounters; Insidious; Ouija | The Conjuring 2; The Amityville Horror; Insidious: Chapter 2; Sinister; Ghostbusters; Paranormal Activity 3 |
| The Hangover | The Hangover Part II; Last Vegas; Very Bad Things; Multiplicity; Welcome to the Rileys; Three Men and a Baby | Old School; Coffee Town; Slackers; Three Men and a Baby; Jeff, Who Lives at Home; Home Alone |
| The Godfather | The Godfather: Part II; The Godfather: Part III; Blood Ties; This Thing of Ours; Mickey Blue Eyes; Summer of Sam | The Godfather: Part II; The Godfather: Part III; The Conformist; Hannibal Rising; The Funeral; Find Me Guilty |
| The Lord of the Rings: The Fellowship of the Ring | The Hobbit: An Unexpected Journey; The Hobbit: The Desolation of Smaug; The Lord of the Rings: The Two Towers; The Hobbit: The Battle of the Five Armies; The Lord of the Rings: The Return of the King; The Addams Family | The Hobbit: An Unexpected Journey; The Hobbit: The Desolation of Smaug; The Lord of the Rings: The Two Towers; The Hobbit: The Battle of the Five Armies; The Lord of the Rings: The Return of the King; Little Nicky |
| Whiplash | Drumline; Hop; Match Point; The Rocker; Spawn; Kung Fu Jungle | Drumline; The Outsiders; Shine; Idlewild; Green Room; The Chorus |
| Schindler's List | Woman in Gold; Saints and Soldiers; The Great Escape; Inglourious Basterds; U-571; Red Tails | Valkyrie; Woman in Gold; The Great Escape; Jakob the Liar; Dead Snow; Hannibal Rising |
| March of the Penguins | Farce of the Penguins; Happy Feet; Happy Feet Two; Madagascar: Escape 2 Africa; Last Vegas; Antarctic Edge: 70° South | Farce of the Penguins; Madagascar 3: Europe's Most Wanted; Happy Feet; Happy Feet Two; Antarctic Edge: 70° South; The Thing |
| The Good, the Bad and the Ugly | A Fistful of Dollars; Once Upon a Time in the West; One Man's Hero; Three Kings; Paint Your Wagon; White Fang | The Legend of Hell's Gate: An American Conspiracy; National Treasure; O Brother, Where Art Thou?; A Fistful of Dollars; Broken Horses; Django Unchained |

Chấm độc lập mức phù hợp cho từng gợi ý để so sánh Precision@6. Cùng một câu tìm có thể có nhiều phim phù hợp; không lấy độ tương đồng của mô hình làm nhãn đánh giá.
Hạn chế: dữ liệu tĩnh; mô hình đọc tối đa 128 token mỗi phim; có thể hiểu sai phủ định, tên riêng, hoặc mô tả quá chung. Chế độ AI trả tối đa 60 phim có cosine dương, không có ngưỡng đảm bảo liên quan. Dùng các bộ lọc để chỉ định yêu cầu bắt buộc.

Chạy lại: `python evaluate_ai.py`. Không sửa báo cáo đánh giá TF-IDF trước đó.