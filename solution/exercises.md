# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Quy luật: temperature càng thấp thì phản hồi càng ổn định, gần như lặp lại đáp án "phổ biến nhất". Temperature càng cao thì chủ đề và từ ngữ càng đa dạng nhưng khó đoán và dễ sai. Tuy vậy, đây vẫn là xác suất: 1.5 vẫn có thể ra cùng chủ đề với 0.0. Ở mức 1.5 văn bản vẫn mạch lạc vì `top_p=0.9` đã cắt bỏ phần đuôi các token xác suất thấp. Mỗi mức mới chạy 1 lần nên đây là quan sát định tính.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt temperature ≈ 0.2 (trong khoảng 0–0.3), giữ nguyên top_p vì chỉ nên chỉnh một trong hai tham số. Chatbot hỗ trợ khách hàng cần trả lời nhất quán và chính xác theo chính sách (giá, đổi trả, bảo hành): hai khách hỏi cùng một câu phải nhận cùng một thông tin, và bot không được tự "sáng tạo" ra điều khoản không có. Thí nghiệm ở Câu 1.1 cho thấy từ mức 1.0 model đã đổi chủ đề và nhầm tên gọi, điều không chấp nhận được khi nói chuyện với khách. Mình không đặt hẳn 0.0 để câu chữ vẫn tự nhiên, bớt máy móc. Còn độ chính xác thật sự phải đến từ system prompt và tài liệu chính sách đưa vào ngữ cảnh, chứ không thể chỉ dựa vào temperature.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> **Ước tính** (chỉ tính output, theo `PRICING_PER_1K_TOKENS` trong `template.py`):
> - Token đầu ra mỗi ngày: 10.000 × 3 × 350 = 10,5 triệu token
> - GPT-4o: 10.500 × $0,010 = $105/ngày (~$3.150/tháng)
> - GPT-4o-mini: 10.500 × $0,0006 = $6,30/ngày (~$189/tháng)
> - → GPT-4o đắt hơn khoảng 16,7 lần ($0,010 / $0,0006). Giá input cũng chênh đúng 16,7 lần ($0,0025 / $0,00015), nên dù tính thêm input thì tỷ lệ vẫn giữ nguyên.
>
> - Nên dùng GPT-4o: tác vụ cần suy luận nhiều bước, sai thì thiệt hại lớn và lượng gọi thấp. Ví dụ: trợ lý rà soát hợp đồng cho bộ phận pháp chế, nơi bỏ sót một điều khoản bất lợi tốn kém hơn rất nhiều so với tiền API.
> - Nên dùng mini: tác vụ đơn giản, lượng gọi lớn, cần phản hồi nhanh. Ví dụ: chatbot FAQ (giờ mở cửa, tra cứu đơn hàng) hoặc phân loại ticket, đúng kiểu workload 10.000 người dùng/ngày ở trên. Có thể kết hợp hai model: mini xử lý mặc định, chỉ chuyển câu khó sang GPT-4o (model routing).

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Bản **giáo viên** (172 từ) ví blockchain như "cuốn sổ ghi chép, mỗi trang là một khối nối vào trang trước", gần như không dùng thuật ngữ và kết thúc trọn vẹn. Bản **chuyên gia** dài hơn (208 từ, chạm trần `max_tokens=256` nên bị cắt giữa chừng), dày đặc thuật ngữ như sổ cái phân tán, mạng ngang hàng, mã băm, đồng thuận, có dẫn chứng Bitcoin/Ethereum và trình bày dạng danh sách. Cùng câu hỏi, cùng model, chỉ đổi system prompt đã làm thay đổi đối tượng người đọc, độ sâu, giọng văn và cả cấu trúc câu trả lời. System prompt đóng vai trò khung hành vi cho toàn bộ phản hồi, là cách rẻ nhất để điều chỉnh sản phẩm mà không cần đổi model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn tiếng Việt 121 từ (giới thiệu về Việt Nam): `count_tokens` (gpt-4o, encoding `o200k_base`) cho **158 token**, còn ước lượng 121 / 0,75 ≈ **161**, tức chỉ chênh **~2%**. Tuy nhiên đây là trùng hợp: với encoding cũ `cl100k_base` (GPT-3.5/GPT-4), cùng đoạn này ra **263 token**, cao hơn ước lượng **~63%**. Cùng nội dung viết bằng tiếng Anh chỉ tốn 122 token (~5,2 ký tự/token, so với ~3,4 ký tự/token của tiếng Việt), nghĩa là tiếng Việt tốn hơn khoảng 30%.
> Lý do: tokenizer BPE học chủ yếu từ dữ liệu tiếng Anh, nên phần lớn từ tiếng Anh là một token nguyên vẹn. Chữ có dấu của tiếng Việt (ệ, ổ, ế) chiếm nhiều byte UTF-8 và ít xuất hiện trong dữ liệu hơn, nên bị cắt vụn: `cl100k_base` tách "Việt" thành `Vi` + `ệ` + `t`; `o200k_base` có từ vựng lớn hơn nên giữ được "Việt" nhưng vẫn tách "hiếu" thành `hi` + `ếu`. Vì vậy không nên ước lượng chi phí tiếng Việt bằng số từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi **có người đang ngồi chờ đọc một câu trả lời dài**: chatbot, trợ lý viết bài, giải thích code. Đo thật với gpt-4o (bài ~300 từ, lấy trung vị 3 lần): bản stream hiện chữ đầu tiên sau **~1,4 s**, còn bản non-stream bắt người dùng nhìn màn hình trống **~4,7 s** rồi mới hiện cả bài. Tổng thời gian không nhanh hơn (stream xong sau ~6,3 s, mạng dao động), nhưng người dùng *cảm thấy* nhanh hơn nhiều và có thể dừng sớm nếu câu trả lời đi sai hướng. Ngược lại, non-streaming phù hợp khi **không có người đọc trực tiếp** hoặc cần kết quả trọn vẹn mới xử lý được: trả JSON để parse, phân loại hay trích xuất dữ liệu, xử lý batch chạy nền, câu trả lời rất ngắn, hoặc phải kiểm duyệt toàn bộ nội dung trước khi hiển thị. Non-streaming cũng đơn giản hơn: có sẵn `usage` để tính token, dễ retry và dễ test.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Delay cố định giữ nguyên nhịp dồn request vào server đang quá tải, nên server không có thời gian hồi phục. Exponential backoff giãn khoảng chờ rất nhanh (demo `retry_with_backoff` với `base_delay=0.1`: thử lại sau 0,1 → 0,2 → 0,4 s): lỗi thoáng qua thì được retry sớm, lỗi kéo dài thì client tự giảm tải, và `max_retries` chặn tổng số lần gọi.
> Nếu hàng nghìn client cùng gặp lỗi một lúc rồi cùng chờ đúng 1 giây, chúng sẽ **retry đồng loạt tại cùng thời điểm** (hiệu ứng *thundering herd*): server vừa hồi lại đã bị một đợt sóng mới đánh sập, lặp lại mỗi giây và không thoát ra được. Cách khắc phục là backoff theo cấp số nhân **kèm jitter**, tức cộng thêm thời gian ngẫu nhiên (ví dụ `delay * random.uniform(0.5, 1.5)`) để các client tản ra; bản `retry_with_backoff` hiện tại chưa có jitter nên các client vẫn có thể retry trùng nhịp.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> **Persona:** trợ giảng của khóa AI. System prompt đang dùng trong `template.py`: `"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."`
> - **"trả lời ngắn gọn":** học viên hỏi nhanh trong lúc code nên cần đúng ý chính, đồng thời giảm chi phí vì token output đắt gấp 4 lần input. Đo thật với câu "Temperature trong LLM là gì?": không có persona thì trả lời 176 từ / 215 token, có persona chỉ 56 từ / 69 token (giảm ~68% token output) mà vẫn đủ ý.
> - **"bằng tiếng Việt":** học viên là người Việt; nếu không chỉ định, model có thể chuyển sang tiếng Anh khi câu hỏi chứa nhiều thuật ngữ (token, streaming, backoff). Chữ "thân thiện" giữ giọng văn gần gũi, ví dụ bot chủ động chào "Chào Minh!".

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất: history chỉ giữ 3 lượt.** Thử thật 5 lượt: lượt 1 nói "Mình tên là Minh", đến lượt 5 hỏi "Mình tên là gì?" thì trợ lý trả lời "mình không có khả năng biết tên của bạn", vì `history[-6:]` đã cắt mất lượt 1. Trong một buổi hỏi đáp dài, trợ lý sẽ quên bối cảnh ban đầu (đang làm bài nào, đã gặp lỗi gì).
> **Cải thiện: tóm tắt phần hội thoại cũ thay vì bỏ hẳn (summary memory).**
> 1. Giữ thêm biến `summary = ""`. Trước khi cắt `history[-6:]`, lấy các message sắp bị bỏ (`history[:-6]`) và gọi `call_openai_mini` với prompt "Tóm tắt thông tin quan trọng về người dùng và nội dung đã trao đổi, tối đa 100 từ", kèm `summary` cũ và các message đó, rồi gán kết quả lại vào `summary`.
> 2. Khi ghép `messages`, đưa tóm tắt vào system prompt: `{"role": "system", "content": persona + "\nBối cảnh trước đó: " + summary}`.
> 3. Cộng token và chi phí của lần gọi tóm tắt vào thống kê phiên.
>
> Nhờ vậy trợ lý vẫn nhớ tên và bối cảnh, còn token input mỗi lượt chỉ tăng thêm khoảng 100 từ thay vì tăng theo toàn bộ độ dài hội thoại.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
