# NLP, Embedding và Transformer cho bài toán địa chỉ tiếng Việt

Tài liệu này tổng hợp chi tiết kiến thức từ đầu quá trình tìm hiểu đến self-attention. Mục tiêu cuối cùng là hiểu nền tảng trước khi học BERT, PhoBERT và fine-tune model nhận diện thành phần địa chỉ.

```text
NLP
→ Tokenization
→ Token ID
→ Embedding
→ Position embedding
→ Transformer Encoder
→ Self-attention
→ Contextual embedding
→ Token classifier
→ Nhãn NER
```

## 1. NLP là gì?

NLP, viết tắt của Natural Language Processing, là lĩnh vực giúp máy tính xử lý ngôn ngữ tự nhiên của con người.

Ngôn ngữ tự nhiên là những gì con người sử dụng hằng ngày:

```text
Tôi đang học NLP.
Giao hàng đến 12 Nguyễn Trãi, Quận 1.
Phim này hay nhưng đoạn kết hơi chán.
```

Nó khác ngôn ngữ lập trình. Code có cú pháp chặt chẽ; cùng một địa chỉ lại có thể được viết theo nhiều cách:

```text
12 Nguyễn Trãi, P. Bến Thành, Q1, TP.HCM
12 nguyen trai ben thanh q1 hcm
Số 12 đường Nguyễn Trãi ở Bến Thành
12 N.Trãi, BT, Q.1
```

NLP không phải một model cụ thể:

```text
NLP          = lĩnh vực
Transformer  = kiến trúc neural network dùng trong NLP
BERT         = model xây từ Transformer Encoder
PhoBERT      = model kiểu BERT được pretrain cho tiếng Việt
```

Có thể hình dung:

```text
Computer Vision
└── CNN
    └── ResNet

NLP
└── Transformer
    └── BERT
        └── PhoBERT
```

Trong project địa chỉ:

```text
Lĩnh vực: NLP
Bài toán: Vietnamese Address Parsing
Bài toán ML: Supervised learning
Dạng bài toán NLP: Named Entity Recognition
Dạng đầu ra: Token Classification / Sequence Labeling
Model dự kiến: PhoBERT
```

## 2. Máy tính nhìn văn bản như thế nào?

Con người nhìn chuỗi:

```text
12 Nguyễn Trãi, Phường Bến Thành
```

và hiểu gần như ngay:

```text
12                 = số nhà
Nguyễn Trãi        = tên đường
Phường Bến Thành   = đơn vị hành chính
```

Máy tính ban đầu chỉ nhận một chuỗi ký tự. Nó không tự biết Nguyễn Trãi là một cụm, Bến Thành là địa danh hay P. là viết tắt của Phường.

Để neural network xử lý được, văn bản phải đi qua chuỗi biến đổi:

```text
Văn bản
→ chia thành token
→ đổi token thành ID
→ đổi ID thành vector
→ dùng model xử lý quan hệ và ngữ cảnh
→ tạo dự đoán
```

## 3. Vì sao ngôn ngữ khó xử lý?

### Một từ có nhiều nghĩa

```text
Tôi đi trên đường Nguyễn Trãi.
Tôi cho đường vào cà phê.
```

Từ đường ở câu đầu là đường phố, ở câu sau là thực phẩm.

### Một ý có nhiều cách viết

```text
Thành phố Hồ Chí Minh
TP. Hồ Chí Minh
TPHCM
HCM
Sài Gòn
```

### Cùng một chuỗi có nhiều vai trò

```text
Tôi đang đọc thơ Nguyễn Trãi. → Nguyễn Trãi là tên người
Tôi sống ở đường Nguyễn Trãi. → Nguyễn Trãi là tên đường
```

### Thông tin có thể thiếu hoặc nhập nhằng

```text
Tân Phú
```

Chuỗi này chưa đủ để xác định một xã/phường cụ thể vì có thể có nhiều đơn vị cùng tên. Hệ thống tốt phải có khả năng trả về AMBIGUOUS thay vì luôn đoán một kết quả.

## 4. Các tầng thông tin trong NLP

### Tầng ký tự

Xử lý từng ký tự, hữu ích khi gặp mất dấu, lỗi chính tả, OCR hoặc ký tự đặc biệt:

```text
Nguyễn → N, g, u, y, ễ, n
```

### Tầng token

Văn bản được chia thành các đơn vị nhỏ:

```text
12 | Nguyễn_Trãi | , | Phường | Bến_Thành
```

Token không nhất thiết là một từ hoàn chỉnh. Subword tokenizer có thể chia một từ hiếm thành nhiều mảnh nhỏ.

### Tầng cú pháp

Cú pháp mô tả cấu trúc và vai trò ngữ pháp:

```text
Nam gửi hàng đến Hà Nội.

Nam      → chủ ngữ
gửi      → động từ chính
hàng     → tân ngữ
Hà Nội   → thành phần chỉ địa điểm
```

### Tầng ngữ nghĩa

Ngữ nghĩa mô tả ý nghĩa và vai trò trong sự việc:

```text
Nam      → người thực hiện
gửi      → hành động
hàng     → đối tượng bị tác động
Hà Nội   → địa điểm đích
```

Việc gán người thực hiện, đối tượng và địa điểm đích chính xác hơn được gọi là Semantic Role Labeling. Cú pháp và ngữ nghĩa liên quan chặt chẽ nhưng không phải một.

## 5. Một số bài toán NLP

### Text classification

Gán một nhãn cho toàn bộ văn bản:

```text
Phim này rất hay → POSITIVE
Phim quá chán    → NEGATIVE
```

### Named Entity Recognition

Tìm và phân loại thực thể:

```text
Nguyễn Văn An → PERSON
Hà Nội        → LOCATION
```

### Các bài toán khác

- Dịch máy.
- Hỏi đáp.
- Tóm tắt.
- Sinh văn bản.
- Tìm kiếm ngữ nghĩa.
- So sánh độ giống nhau giữa các câu.

## 6. Bài toán địa chỉ là NER có giám sát

Text classification thông thường trả một nhãn cho cả chuỗi. Address NER phải trả một nhãn cho từng token:

```text
12            → HOUSE_NUMBER
Nguyễn_Trãi   → STREET
Phường        → WARD
Bến_Thành     → WARD
Quận          → DISTRICT
1             → DISTRICT
```

Vì đầu ra là một chuỗi nhãn tương ứng với chuỗi token, đây là:

```text
Sequence Labeling
└── Token Classification
    └── Named Entity Recognition
```

Dữ liệu train có dạng:

```text
X = chuỗi token địa chỉ
Y = nhãn đúng của từng token
```

Đầu ra đúng đã được gán sẵn, nên đây là supervised learning.

So với Face Emotion Detection:

```text
Face Emotion Detection:
một ảnh → một nhãn cảm xúc

Address NER:
một chuỗi token → một chuỗi nhãn
```

## 7. Nhãn BIO

NER thường dùng:

- B, Beginning: token bắt đầu một thực thể.
- I, Inside: token tiếp tục thực thể đó.
- O, Outside: token không thuộc thực thể nào.

```text
Token       Label
12          B-HOUSE_NUMBER
Nguyễn      B-STREET
Trãi        I-STREET
,           O
Phường      B-WARD
Bến         I-WARD
Thành       I-WARD
```

BIO giúp xác định ranh giới. Sau dự đoán, hậu xử lý ghép:

```text
B-STREET + I-STREET → Nguyễn Trãi
B-WARD + I-WARD + I-WARD → Phường Bến Thành
```

## 8. Pipeline NLP của bài toán

```text
Văn bản thô
→ chuẩn hóa Unicode và khoảng trắng
→ tokenizer
→ token IDs
→ embedding
→ Transformer Encoder
→ contextual embedding
→ token classifier
→ nhãn BIO
→ hậu xử lý
→ các trường địa chỉ
```

Điểm cần phân biệt:

```text
Tokenizer       → tách văn bản thành token
Token classifier → gán nhãn cho token đã được tách
```

Classifier không tách token.

## 9. Tokenization và token ID

Ví dụ:

```text
Văn bản:
12 Nguyễn Trãi

Tokens:
[12, Nguyễn_Trãi]

Token IDs:
[134, 8201]
```

Token ID là số thứ tự trong vocabulary:

```text
Phường → ID 215
```

ID chỉ là mã định danh. ID 8201 không lớn hơn hay giàu ý nghĩa hơn ID 100.

Với một tokenizer cố định, cùng một token thường ánh xạ tới cùng một ID. Ví dụ token đường có thể vẫn mang ID 315 trong cả hai câu:

```text
Tôi đi trên đường Nguyễn Trãi.
Tôi cho đường vào cà phê.
```

Cùng token ID chưa có nghĩa model sẽ hiểu hai trường hợp giống nhau. Transformer sẽ phân biệt chúng bằng ngữ cảnh ở bước sau.

## 10. Vì sao không chỉ dùng one-hot?

Nếu vocabulary có năm token:

```text
Nguyễn     → [1, 0, 0, 0, 0]
Trãi       → [0, 1, 0, 0, 0]
Phường     → [0, 0, 1, 0, 0]
Bến_Thành  → [0, 0, 0, 1, 0]
Hà_Nội     → [0, 0, 0, 0, 1]
```

One-hot có hai hạn chế:

1. Vocabulary 50.000 token tạo vector 50.000 chiều nhưng gần như toàn số 0.
2. Mọi token đều cách nhau như nhau, nên không biểu diễn được mức độ liên quan.

Embedding dùng vector đặc, số chiều nhỏ hơn và có thể học được đặc trưng hữu ích.

## 11. Embedding và embedding matrix

Embedding ánh xạ token ID thành vector số thực có số chiều cố định:

```text
134  → [0.12, -0.38, 0.71, ..., 0.24]
8201 → [0.84,  0.15, -0.29, ..., 0.63]
215  → [0.73,  0.42, -0.16, ..., 0.51]
```

Các số trên chỉ minh họa, không phải ID hoặc vector thật của PhoBERT.

Nếu vocabulary có 64.000 token và hidden size là 768:

```text
Embedding matrix shape = [64000, 768]
```

Token ID là chỉ mục để lấy một hàng:

```python
vector = embedding_matrix[token_id]
```

Với năm token:

```text
Trước embedding: [5]
Sau embedding:   [5, 768]
```

Mỗi số trong danh sách token ID không chứa sẵn một vector. Nó được dùng để tra vector trong embedding matrix.

## 12. Embedding có phải dataset đã train không?

Không. Cần phân biệt:

```text
Dataset          = dữ liệu dùng để huấn luyện
Embedding matrix = tham số model học được
```

Ban đầu embedding matrix thường gần như ngẫu nhiên. Trong training:

```text
Input
→ embedding
→ model
→ prediction
→ loss
→ backpropagation
→ cập nhật embedding matrix
```

Sau khi train, embedding matrix được lưu trong checkpoint cùng các weight khác.

```text
PhoBERT checkpoint
├── embedding matrix đã học
└── Transformer weights đã học
```

Tải PhoBERT pretrained nghĩa là tải các tham số đã được học từ lượng lớn văn bản tiếng Việt. Fine-tuning cho Address NER tiếp tục điều chỉnh chúng bằng dữ liệu địa chỉ có nhãn, thường với learning rate nhỏ.

## 13. Static embedding và contextual embedding

Static embedding như Word2Vec thường cho mỗi từ một vector cố định:

```text
đường → luôn là vector A
```

Điều này khó biểu diễn riêng nghĩa đường phố và đường ăn.

BERT và PhoBERT dùng contextual embedding:

```text
Cùng token
→ cùng token ID
→ cùng initial token embedding
→ khác câu và ngữ cảnh
→ khác contextual embedding
```

Ví dụ:

```text
đường trong đi trên đường       → vector mang nghĩa đường phố
đường trong cho đường vào cà phê → vector mang nghĩa thực phẩm
```

Cùng token có initial embedding giống nhau; Transformer tạo output embedding khác nhau.

## 14. Position embedding

Self-attention nguyên bản không tự biết thứ tự token. Model cộng position embedding vào token embedding:

```text
Input vector = Token embedding + Position embedding
```

Nếu hidden size là 768:

```text
Token embedding:    [768]
Position embedding: [768]
Input vector:       [768]
```

Ví dụ:

```text
embedding(Nam) + position(0)
embedding(yêu) + position(1)
embedding(Lan) + position(2)
```

Nhờ vị trí, model phân biệt Nam yêu Lan với Lan yêu Nam.

Điểm quan trọng:

```text
Token embedding    → token là gì
Position embedding → token đứng ở đâu
Self-attention     → token liên quan đến các token nào
```

Position embedding chưa tạo ra ngữ cảnh ngữ nghĩa. Nó chỉ bổ sung thông tin thứ tự. Self-attention mới tổng hợp thông tin từ các token khác.

Position embedding thuộc phần chuẩn bị đầu vào của kiến trúc Transformer, trước các Encoder block. BERT và PhoBERT dùng learned position embedding, tức là vector vị trí cũng được cập nhật bằng backpropagation.

## 15. Transformer nào liên quan tới project?

Transformer đầy đủ cho dịch máy có Encoder và Decoder:

```text
Input → Encoder → Decoder → sinh chuỗi output
```

BERT và PhoBERT chỉ dùng phần Encoder:

```text
Token embedding + Position embedding
→ Transformer Encoder
→ Contextual embeddings
```

Address NER không cần Decoder vì không sinh một câu mới:

```text
PhoBERT Encoder → Token classifier → nhãn NER
```

Vì vậy hình dịch Anh sang Việt có liên quan ở phần embedding, position embedding và Encoder; phần Decoder không cần quan tâm cho bài hiện tại.

## 16. Một Transformer Encoder block

Một block gồm các phần chính:

```text
Input
→ Self-attention
→ Residual connection + Layer normalization
→ Feed-Forward Network
→ Residual connection + Layer normalization
→ Output
```

Các block được xếp chồng. Output của block trước là input của block sau.

```text
Self-attention:
các token trao đổi thông tin với nhau

Feed-Forward Network:
mỗi token xử lý riêng thông tin vừa nhận
```

## 17. Self-attention giải quyết vấn đề gì?

Ý nghĩa cốt lõi:

> Self-attention giúp mỗi token lấy thêm thông tin từ các token khác để hiểu vai trò của chính nó.

Xét:

```text
12 | Nguyễn_Trãi | Phường | Bến_Thành
```

Trước attention, vector của Bến_Thành cho biết:

```text
Tôi là token Bến Thành và đứng ở vị trí này.
```

Attention cho phép nó lấy thêm tín hiệu từ Phường:

```text
Tôi là Bến Thành, đứng ở vị trí này,
và trong cùng câu có từ Phường đứng trước tôi.
```

Biểu diễn mới có thêm ngữ cảnh, giúp token classifier dự đoán WARD.

## 18. Query, Key và Value

Mỗi attention head có ba ma trận tham số học được:

```text
W_Q, W_K, W_V
```

Từ vector đầu vào x_i của token i:

```text
q_i = x_i W_Q
k_i = x_i W_K
v_i = x_i W_V
```

Trực giác:

- Query: token hiện tại đang tìm loại thông tin gì.
- Key: đặc điểm để các Query so khớp với token này.
- Value: nội dung token sẽ truyền đi nếu được chú ý.

Key và Value được tách vì tiêu chí dùng để tìm một token không nhất thiết giống nội dung cần lấy từ token đó. Tiêu đề giúp tìm đúng sách; nội dung sách mới là thứ được đọc.

Query, Key và Value không phải embedding gốc. Chúng là ba phép chiếu học được từ cùng input vector.

## 19. Cách tính attention

Để xem token i nên lấy bao nhiêu thông tin từ token j:

```text
score(i, j) = q_i · k_j / sqrt(d_k)
```

Trong đó:

- Dấu chấm là tích vô hướng.
- Điểm lớn nghĩa là Query và Key phù hợp hơn.
- Chia cho căn bậc hai của d_k giúp giá trị ổn định khi số chiều lớn.

Model tính score giữa token đang xét và mọi token, sau đó dùng softmax:

```text
alpha(i, j) = softmax(score(i, j))
```

Các alpha trên một hàng có tổng bằng 1. Ví dụ minh họa khi xử lý Bến_Thành:

```text
12            → 0.05
Nguyễn_Trãi   → 0.10
Phường        → 0.55
Bến_Thành     → 0.30
```

Đây không phải xác suất Bến_Thành là phường. Nó chỉ là trọng số dùng để trộn thông tin.

Output attention của token i:

```text
z_i = tổng theo j của alpha(i, j) × v_j
```

Nếu mỗi Value có 768 chiều, mỗi tích alpha × Value vẫn có 768 chiều và z_i cũng có 768 chiều.

Công thức gộp:

```text
Q = X W_Q
K = X W_K
V = X W_V

Attention(Q, K, V)
= softmax(Q K^T / sqrt(d_k)) V
```

Ý nghĩa:

```text
Q so với K → quyết định lấy thông tin từ token nào
Trọng số nhân V → lấy và tổng hợp nội dung
Output → biểu diễn token đã có thêm ngữ cảnh
```

## 20. Model biết quan hệ bằng cách nào?

Ban đầu W_Q, W_K và W_V gần như ngẫu nhiên. Model chưa biết Bến_Thành nên chú ý tới Phường.

Giả sử classifier dự đoán:

```text
Bến_Thành → STREET
```

trong khi nhãn đúng là WARD. Loss tăng và gradient truyền ngược:

```text
Loss
→ token classifier
→ contextual embedding
→ attention weights
→ W_Q, W_K, W_V
→ các tầng và embedding trước đó
```

Qua nhiều mẫu:

```text
Phường Bến Thành
Phường Tân Định
Phường Dịch Vọng
```

các weight được điều chỉnh theo hướng tạo ra biểu diễn giúp classifier dự đoán đúng hơn.

Không có rule viết sẵn:

```python
if previous_token == "Phường":
    attention = 0.55
```

Attention học gián tiếp thông qua mục tiêu giảm loss.

Câu Bến_Thành chú ý mạnh tới Phường chỉ là ví dụ trực giác. Muốn biết model thật làm gì phải lấy attention matrix của từng layer và từng head. Attention cao cũng không chứng minh token đó là nguyên nhân duy nhất của dự đoán.

## 21. Multi-Head Attention

Một head có một bộ W_Q, W_K, W_V. Multi-head attention có nhiều bộ chạy song song:

```text
Head 1: W_Q1, W_K1, W_V1
Head 2: W_Q2, W_K2, W_V2
...
```

Nhờ các phép chiếu khác nhau, các head có thể học những kiểu quan hệ khác nhau. Output của các head được nối lại rồi qua một linear layer.

Không nên hiểu cứng rằng một head chắc chắn phụ trách phường, một head phụ trách quận. Đó chỉ là minh họa trực giác; vai trò thực tế có thể phân tán và khó diễn giải.

## 22. Residual, Layer Normalization và Feed-Forward

Attention output không thay thế hoàn toàn input. Residual connection cộng lại vector cũ:

```text
output = input + attention_output
```

Nhờ đó token giữ thông tin ban đầu và nhận thêm context.

Layer normalization giúp phân phối giá trị ổn định hơn trong mạng sâu.

Feed-Forward Network là một neural network nhỏ áp dụng độc lập cho từng token:

```text
hidden size
→ Linear mở rộng
→ activation
→ Linear thu về hidden size
```

Self-attention trao đổi thông tin giữa token; Feed-Forward biến đổi thông tin bên trong từng token. Qua nhiều Encoder block, contextual embedding ngày càng giàu thông tin.

## 23. Token classifier

Sau Transformer, mỗi token có contextual embedding. Nếu hidden size là 768 và có 11 nhãn:

```text
Linear layer: 768 → 11 logits
```

Ví dụ:

```text
Contextual embedding của Nguyễn_Trãi
→ điểm cho O, HOUSE_NUMBER, STREET, WARD, ...
→ chọn nhãn có điểm cao nhất
→ B-STREET
```

Shape của pipeline:

```text
input_ids:             [batch_size, sequence_length]
initial embeddings:    [batch_size, sequence_length, 768]
contextual embeddings: [batch_size, sequence_length, 768]
logits:                [batch_size, sequence_length, num_labels]
predicted labels:      [batch_size, sequence_length]
```

## 24. Toàn bộ luồng cho địa chỉ

```text
12 Nguyễn Trãi, Phường Bến Thành, Quận 1
        ↓
Tokenizer
        ↓
Tokens và token IDs
        ↓
Token embedding + Position embedding
        ↓
Transformer Encoder
  ├── Self-attention trao đổi thông tin
  ├── Residual và Layer Normalization
  └── Feed-Forward xử lý từng token
        ↓
Contextual embedding cho từng token
        ↓
Token classifier
        ↓
Nhãn BIO
        ↓
Hậu xử lý
        ↓
HOUSE_NUMBER, STREET, WARD, DISTRICT, PROVINCE
```

## 25. Parser và resolver

PhoBERT Address NER là parser. Nó trả lời:

```text
Đoạn nào là số nhà?
Đoạn nào là đường?
Đoạn nào là phường?
Đoạn nào là quận hoặc tỉnh?
```

Resolver mới trả lời:

```text
Đơn vị này có tồn tại không?
Mã hành chính là gì?
Quan hệ phường - tỉnh có hợp lệ không?
Địa chỉ cũ chuyển sang đơn vị mới nào?
```

Pipeline hoàn chỉnh là hybrid:

```text
PhoBERT NER parser
→ exact/fuzzy matching
→ kiểm tra hierarchy hành chính
→ old-to-new mapping
→ PASS, WARNING, AMBIGUOUS hoặc REJECTED
```

## 26. Những điểm dễ nhầm

```text
Token ID
≠ embedding vector

Embedding matrix
≠ dataset

Position embedding
≠ context ngữ nghĩa

Tokenizer
≠ token classifier

Self-attention
≠ toàn bộ Transformer

Attention weight
≠ xác suất nhãn

PhoBERT parser
≠ resolver hành chính
```

## 27. Bản đồ kiến thức hiện tại

```text
NLP
└── Address NER
    ├── Tokenizer tạo token IDs
    ├── Embedding biến ID thành vector
    ├── Position embedding thêm thứ tự
    ├── Transformer tạo contextual embeddings
    │   ├── Self-attention
    │   │   ├── Query
    │   │   ├── Key
    │   │   └── Value
    │   ├── Multi-head attention
    │   ├── Residual + Layer normalization
    │   └── Feed-Forward Network
    ├── Token classifier gán nhãn BIO
    └── Hậu xử lý ghép thành trường địa chỉ
```

Bước kiến thức tiếp theo là BERT: BERT dùng Transformer Encoder như thế nào, pretraining bằng Masked Language Modeling ra sao, rồi PhoBERT điều chỉnh quy trình đó cho tiếng Việt như thế nào.

# Phần II. Nền tảng toán học và biểu diễn dữ liệu

Phần này mô tả chính xác hơn các khái niệm bằng ký hiệu toán học và kích thước tensor.

## 28. Ký hiệu và tensor đầu vào

Giả sử một câu sau tokenization có n token:

```text
t_1, t_2, ..., t_n
```

Tokenizer ánh xạ chúng thành các ID:

```text
i_1, i_2, ..., i_n
```

Ký hiệu:

```text
V       = kích thước vocabulary
d_model = số chiều biểu diễn của model
n       = sequence length
B       = batch size
```

Tensor token ID của một batch:

```text
I ∈ N^(B × n)
```

Embedding matrix:

```text
E ∈ R^(V × d_model)
```

Tra mỗi ID trong I vào E tạo:

```text
X_token ∈ R^(B × n × d_model)
```

Ví dụ B = 32, n = 128 và d_model = 768:

```text
input_ids shape        = [32, 128]
token_embeddings shape = [32, 128, 768]
```

## 29. Embedding lookup và one-hot

Về toán học, embedding lookup tương đương nhân one-hot vector với embedding matrix. Nếu token i có one-hot vector o_i thuộc R^V:

```text
e_i = o_i E
```

Vì o_i chỉ có một phần tử bằng 1, phép nhân lấy đúng hàng i của E. Thư viện dùng lookup trực tiếp để không phải tạo vector one-hot rất lớn.

Embedding không có sẵn ý nghĩa do con người quy định. Ý nghĩa hữu ích xuất hiện vì gradient điều chỉnh E theo mục tiêu huấn luyện. Thông tin thường phân tán trên nhiều chiều, nên không thể mặc định chiều 1 là địa điểm hay chiều 2 là con người.

Hai vector có thể được so sánh bằng cosine similarity:

```text
cos(a, b) = (a · b) / (||a|| ||b||)
```

Cosine gần 1 nghĩa là hai vector có hướng gần nhau. Điều đó có thể phản ánh đồng nghĩa, cùng chủ đề, cùng vai trò ngữ pháp hoặc ngữ cảnh sử dụng tương tự; không nhất thiết chỉ là đồng nghĩa.

## 30. Tokenization tiếng Việt và subword

Trong tiếng Việt, dấu cách thường phân tách tiếng chứ không chắc chắn phân tách từ:

```text
sinh viên
Hồ Chí Minh
Bến Thành
```

Word segmentation có thể tạo:

```text
sinh_viên
Hồ_Chí_Minh
Bến_Thành
```

Sau đó subword tokenizer ánh xạ từ vào vocabulary. Khi từ không có nguyên vẹn trong vocabulary, nó được chia thành các mảnh đã biết. Cách này giới hạn kích thước vocabulary và xử lý được từ hiếm.

Với NER, subword tạo bài toán căn chỉnh nhãn. Nếu một từ được chia thành ba subword, cần quy định:

- Gán nhãn cho subword đầu và bỏ qua loss ở phần còn lại.
- Hoặc truyền nhãn BIO thích hợp cho mọi subword.

Sai ở bước label alignment có thể làm training sai dù kiến trúc model đúng.

Batch thường có thêm:

```text
attention_mask:
1 = token thật
0 = padding
```

Padding giúp các câu dài ngắn khác nhau nằm trong cùng tensor, nhưng phải bị che để model không coi padding là nội dung.

## 31. Tổng hợp embedding đầu vào

Ở dạng đơn giản:

```text
X = X_token + X_position
```

Cả hai tensor có shape [B, n, d_model]. Position embedding tại vị trí p là vector P_p. Phép cộng không tăng số chiều; vector kết quả vừa mang thông tin nhận dạng token vừa mang thông tin vị trí.

BERT còn có token-type embedding để phân biệt câu A và câu B. Với Address NER chỉ có một chuỗi, thành phần này không giữ vai trò trung tâm.

## 32. Self-attention dưới dạng ma trận

Với một câu, bỏ qua chiều batch:

```text
X ∈ R^(n × d_model)

W_Q ∈ R^(d_model × d_k)
W_K ∈ R^(d_model × d_k)
W_V ∈ R^(d_model × d_v)
```

Ba phép chiếu:

```text
Q = X W_Q ∈ R^(n × d_k)
K = X W_K ∈ R^(n × d_k)
V = X W_V ∈ R^(n × d_v)
```

Ma trận điểm:

```text
S = Q K^T / sqrt(d_k)
S ∈ R^(n × n)
```

Phần tử S[i, j] cho biết token i nên lấy bao nhiêu thông tin từ token j. Áp dụng softmax theo từng hàng:

```text
A = softmax(S)
A ∈ R^(n × n)
```

Mỗi hàng A[i, :] có tổng bằng 1. Output:

```text
Z = A V
Z ∈ R^(n × d_v)
```

Hàng Z[i] là tổng có trọng số của các Value, tức biểu diễn ngữ cảnh mới cho token i.

## 33. Vì sao chia cho căn bậc hai của d_k?

Khi d_k lớn, tích vô hướng Q · K có xu hướng lớn hơn. Giá trị quá lớn làm softmax rất nhọn:

```text
softmax([1, 2, 20]) ≈ [0, 0, 1]
```

Khi đó gradient của nhiều phần tử gần bằng 0, làm tối ưu khó ổn định. Chia cho sqrt(d_k) giúp kiểm soát độ lớn của score. Vì vậy cơ chế có tên Scaled Dot-Product Attention.

## 34. Attention mask

Trước softmax, vị trí không được phép chú ý nhận score âm rất lớn:

```text
S_masked = S + M
```

M bằng 0 ở vị trí hợp lệ và gần âm vô cùng ở vị trí bị che. Sau softmax, trọng số của vị trí bị che gần bằng 0.

PhoBERT Encoder chủ yếu dùng padding mask. Causal mask, không cho nhìn token tương lai, quan trọng với decoder sinh văn bản nhưng không phải cơ chế chính của PhoBERT.

# Phần III. Kiến trúc, huấn luyện và đánh giá

## 35. Multi-Head Attention dưới dạng toán học

Một head chỉ biểu diễn quan hệ trong một không gian chiếu. Với h head:

```text
head_r = Attention(X W_Qr, X W_Kr, X W_Vr)
```

Các head được nối:

```text
H = Concat(head_1, ..., head_h)
```

Sau đó chiếu về d_model:

```text
MultiHead(X) = H W_O
```

Thông thường d_k được chọn sao cho:

```text
d_k = d_model / h
```

Nhờ vậy tổng kích thước sau khi nối các head trở lại d_model. Mỗi head có tham số riêng nên có thể biểu diễn những kiểu quan hệ khác nhau, nhưng không nên mặc định từng head có một chức năng dễ gọi tên.

## 36. Residual connection và Layer Normalization

Với attention output MHA(X), residual tạo:

```text
H_1 = LayerNorm(X + MHA(X))
```

Residual cung cấp đường truyền trực tiếp cho thông tin và gradient, giúp huấn luyện mạng sâu ổn định hơn. Layer normalization chuẩn hóa các đặc trưng của từng token, giúp phân phối giá trị ổn định trong quá trình tối ưu.

Thứ tự LayerNorm trước hay sau sublayer phụ thuộc biến thể kiến trúc. Không nên coi một sơ đồ duy nhất là đúng cho mọi Transformer.

## 37. Feed-Forward Network

Mỗi token được xử lý độc lập bằng cùng một mạng:

```text
FFN(x) = activation(x W_1 + b_1) W_2 + b_2
```

Thường:

```text
d_model → d_ff → d_model
```

Self-attention trộn thông tin giữa các vị trí; FFN biến đổi phi tuyến thông tin bên trong từng vị trí. Với H_1:

```text
H_2 = LayerNorm(H_1 + FFN(H_1))
```

H_2 là output của một Encoder block và trở thành input của block kế tiếp.

## 38. Vì sao xếp nhiều Encoder layer?

Một layer thực hiện một lần trao đổi và biến đổi thông tin. Xếp nhiều layer cho phép model xây dựng biểu diễn theo nhiều mức:

```text
Layer thấp  → mẫu cục bộ, hình thái, vị trí
Layer giữa  → quan hệ giữa từ và cụm
Layer cao   → thông tin phục vụ ý nghĩa và nhiệm vụ
```

Đây là xu hướng khái quát, không phải quy tắc cứng cho từng layer. Mỗi output vẫn có shape [B, n, d_model], nhưng nội dung biểu diễn thay đổi.

## 39. Huấn luyện attention bằng backpropagation

Attention không có nhãn riêng nói token nào phải nhìn token nào. Model chỉ nhận mục tiêu cuối, chẳng hạn nhãn NER.

Với token i và nhãn đúng y_i, classifier tạo logits l_i. Cross-entropy:

```text
L_i = -log softmax(l_i)[y_i]
```

Loss của câu thường là trung bình hoặc tổng trên các token hợp lệ:

```text
L = tổng L_i / số token hợp lệ
```

Gradient đi ngược:

```text
Loss
→ classifier
→ contextual embeddings
→ Encoder layers
→ W_Q, W_K, W_V, W_O
→ token và position embeddings
```

Nếu việc chú ý từ Bến_Thành tới Phường giúp giảm loss, tối ưu sẽ có xu hướng điều chỉnh tham số để quan hệ hữu ích đó mạnh hơn. Đây là kết quả học từ dữ liệu, không phải luật được mã hóa thủ công.

## 40. Token classification head

Với output cuối:

```text
H ∈ R^(B × n × d_model)
```

Linear classifier có:

```text
W_c ∈ R^(d_model × C)
```

trong đó C là số nhãn. Logits:

```text
L = H W_c + b_c
L ∈ R^(B × n × C)
```

Argmax trên chiều C cho một nhãn mỗi token. Khi training dùng logits trực tiếp với cross-entropy; không cần tự gọi softmax trước hàm loss nếu framework đã kết hợp hai bước.

## 41. Loss mask và label alignment

Không phải mọi vị trí đều tham gia loss:

- Padding phải bị bỏ qua.
- Special token có thể bị bỏ qua.
- Subword không đại diện đầu từ có thể nhận ignore index.

Framework thường dùng nhãn -100 để CrossEntropyLoss bỏ qua vị trí. Nếu label alignment sai, metric và gradient đều sai.

Ví dụ một từ STREET bị chia thành ba subword:

```text
subword 1 → B-STREET
subword 2 → I-STREET hoặc -100, tùy chiến lược
subword 3 → I-STREET hoặc -100, tùy chiến lược
```

Chiến lược phải nhất quán giữa train, validation và hậu xử lý.

## 42. Precision, Recall và F1

Với một loại thực thể:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

- Precision cao: những thực thể model dự đoán thường đúng.
- Recall cao: model tìm được phần lớn thực thể thật.
- F1 cân bằng hai đại lượng.

NER nên đánh giá ở entity level, không chỉ token accuracy. Ví dụ model đoán đúng Bến nhưng bỏ Thành có thể đạt nhiều token đúng nhưng vẫn trích sai toàn bộ thực thể Bến Thành.

Project địa chỉ còn cần:

```text
house_number accuracy
street entity F1
ward entity F1
district entity F1
province entity F1
full-address exact match
ambiguous rate
not-found rate
latency và memory
```

Parser metric và resolver metric phải tách riêng để biết lỗi đến từ model hay dữ liệu hành chính.

## 43. Pretraining và fine-tuning

Nếu train toàn bộ Transformer chỉ bằng vài nghìn địa chỉ, model khó học tiếng Việt tốt. Pretraining cho model học cấu trúc ngôn ngữ từ corpus lớn trước.

```text
Pretraining:
văn bản tiếng Việt lớn
→ học biểu diễn ngôn ngữ tổng quát

Fine-tuning:
dữ liệu địa chỉ có nhãn
→ học HOUSE_NUMBER, STREET, WARD, DISTRICT, PROVINCE
```

Khi fine-tune, token-classification head thường được khởi tạo mới; PhoBERT Encoder và embeddings bắt đầu từ checkpoint pretrained. Có thể cập nhật toàn bộ model hoặc đóng băng một số tầng. Cập nhật toàn bộ thường mạnh hơn nhưng tốn tài nguyên và có nguy cơ overfit nếu dữ liệu nhỏ.

## 44. Những giới hạn cần hiểu

Contextual embedding không đảm bảo model biết sự thật hành chính. Nó chỉ biểu diễn mẫu học được từ dữ liệu.

Attention weight không phải lời giải thích nhân quả hoàn chỉnh. Một dự đoán còn phụ thuộc vào nhiều head, nhiều layer, residual, FFN và classifier.

Confidence từ softmax không mặc nhiên là xác suất đã được hiệu chỉnh. Muốn dùng ngưỡng nghiệp vụ phải đánh giá và calibration trên dữ liệu đại diện.

Model NER có thể tách đúng Tân Phú là WARD nhưng không xác định được đó là đơn vị nào nếu thiếu tỉnh. Đây là nhiệm vụ của resolver và dữ liệu tham chiếu.

## 45. Chuỗi xử lý hoàn chỉnh cho project

```text
Địa chỉ thô
→ chuẩn hóa có kiểm soát
→ word segmentation và subword tokenization
→ input_ids + attention_mask
→ token embedding + position embedding
→ nhiều PhoBERT Encoder layer
→ contextual embedding mỗi token
→ token-classification head
→ BIO labels
→ ghép subword và entity
→ ParsedAddress
→ exact/fuzzy resolution theo hierarchy
→ ánh xạ đơn vị cũ sang mới
→ ResolvedAddress và trạng thái chất lượng
```

Kiến thức tiếp theo sau tài liệu này:

```text
BERT
→ mục tiêu pretraining
→ Masked Language Modeling
→ cách fine-tune
→ PhoBERT và tokenization tiếng Việt
→ PhoBERT cho Address NER
```

# Phần IV. Checkpoint, attention và multi-head

## 46. Hidden size và checkpoint

Hidden size là số phần tử trong vector biểu diễn mỗi token. Với PhoBERT-base:

```text
hidden_size = d_model = 768
```

Nếu câu có n token, hidden states có shape [n,768]. Con số 768 là hyperparameter đã được chọn khi pretrain và được cố định trong checkpoint:

```text
Embedding matrix: [vocab_size,768]
Attention input:  [n,768]
Classifier NER:   768 → num_labels
```

Fine-tuning thay đổi giá trị weight nhưng không thay đổi hidden size. Không thể đổi riêng 768 thành 512 mà vẫn dùng nguyên checkpoint vì tensor sẽ lệch shape.

## 47. Từ token tới embedding

Vocabulary ánh xạ token sang ID:

```text
Phường      → 215
Nguyễn_Trãi → 8201
```

Vocabulary thường là hash map, không cần duyệt toàn bộ từ điển:

```python
token_id = vocabulary["Phường"]
vector = embedding_matrix[token_id]
```

Tokenizer và checkpoint phải dùng cùng vocabulary. Tokenizer giữ ánh xạ token ↔ ID; checkpoint giữ embedding matrix và model weights.

## 48. Tham số cố định và activation động

Checkpoint lưu:

```text
Embedding matrix E
Position embedding P
W_Q, W_K, W_V, W_O
FFN và LayerNorm weights
```

Với mỗi input, model mới tính:

```text
Token IDs → tra E → cộng P → X
Q = XW_Q
K = XW_K
V = XW_V
```

Vì vậy:

```text
E, P, W_Q, W_K, W_V → tham số lấy từ checkpoint
X, Q, K, V           → kết quả động của input hiện tại
```

Weight lưu cách xử lý; input quyết định kết quả.

## 49. Ý nghĩa chính xác của Q, K, V

Trực giác:

```text
Query → tiêu chí của token đang cần thông tin
Key   → đặc trưng dùng để được so khớp
Value → nội dung được truyền nếu được chú ý
```

Chính xác về toán học:

```text
Q và K tạo điểm tương thích.
Softmax biến điểm thành trọng số.
Trọng số dùng để lấy tổng có trọng số của V.
```

Không thể nhìn riêng một Query vector rồi đọc trực tiếp nó đang tìm danh từ hay địa điểm. Ý nghĩa chỉ được suy ra từ hành vi của model trên nhiều input.

## 50. Hàng và cột của các ma trận

Với n token và d_k chiều:

```text
Q:    [n,d_k]
K:    [n,d_k]
Kᵀ:   [d_k,n]
QKᵀ:  [n,n]
```

Trong Q:

```text
hàng = token đang tạo Query
cột = chiều ẩn của Query
```

Trong Kᵀ:

```text
hàng = chiều ẩn của Key
cột = token cung cấp Key
```

Trong QKᵀ:

```text
hàng i = token i đang hỏi
cột j = token j được hỏi
ô [i,j] = q_i · k_j
```

## 51. Vì sao chia cho sqrt(d_k)?

Mỗi score là tổng của d_k tích:

```text
q_i · k_j = q_i1k_j1 + ... + q_idk k_jdk
```

Với các thành phần có phương sai tương đối ổn định:

```text
Var(q_i · k_j) ≈ d_k
Std(q_i · k_j) ≈ sqrt(d_k)
```

Do đó chia cho sqrt(d_k) để score không tăng quá lớn:

```text
S = QKᵀ / sqrt(d_k)
```

Không chia cho sqrt(n), vì n chỉ là số token và số ô của attention matrix; mỗi ô được tạo từ d_k chiều.

## 52. Attention logits

S được gọi là scaled attention scores hoặc attention logits:

```text
A = softmax(S + mask)
```

Nếu score quá lớn, softmax gần one-hot và gradient rất nhỏ. Scaling giữ phân phối bớt bão hòa.

```text
Attention logits:
so sánh token với token, shape [n,n]

Classification logits:
so sánh nhãn của từng token, shape [n,num_labels]
```

## 53. Vì sao output không phải [n,n]?

[n,n] chỉ là attention weights A, cho biết lấy thông tin ở đâu. Model còn tính:

```text
Z = A V

A: [n,n]
V: [n,d_v]
Z: [n,d_v]
```

PhoBERT-base có 12 head, mỗi head thường trả [n,64]. Ghép lại:

```text
12 × 64 = 768
→ output [n,768]
```

Vì vậy:

```text
[n,n]   = quan hệ token với token
[n,768] = biểu diễn nội dung của n token
```

## 54. Multi-head không cắt thô input

Cách hiểu sai:

```text
Head 1 chỉ nhận feature 0–63
Head 2 chỉ nhận feature 64–127
```

Trong Transformer chuẩn, mỗi head nhìn toàn bộ input bằng learned projection:

```text
X [n,768] × W_Qh [768,64] → Q_h [n,64]
```

Trong code tối ưu, các weight được ghép:

```text
W_Q = [W_Q1 | ... | W_Q12]
W_Q shape = [768,768]
Q = XW_Q → [n,768]
```

Sau đó mới reshape:

```text
[n,768] → [n,12,64]
```

Vì vậy phần được chia thành head là Q, K, V sau projection, không phải raw input trước projection.

## 55. Nhiều góc nhìn và chi phí

Mỗi head có weight riêng nên tạo attention matrix riêng:

```text
A_1, A_2, ..., A_h
```

Single-head có một attention distribution cho mỗi Query. Multi-head có nhiều distribution độc lập. Các output được nối, không lấy trung bình:

```text
Z = Concat(Z_1, ..., Z_h) W_O
```

Về chi phí:

```text
Một head 512 chiều:
n² × 512

Bốn head 128 chiều:
4 × n² × 128 = n² × 512
```

Thiết kế này cho nhiều attention pattern mà tổng FLOPs bậc lớn gần tương đương. Các head có thể chạy song song, nhưng không chắc luôn nhanh hơn do overhead và phần cứng.

Không nên gán cứng mỗi head một chức năng. Một số head có thể học trùng nhau hoặc khó diễn giải.

## 56. Mọi token đều đi qua mọi head

Với n token:

```text
X:             [n,768]
Q_h, K_h, V_h: [n,64]
A_h:           [n,n]
```

Mỗi hàng A_h là attention của một Query token. Cùng một head tạo attention khác nhau cho từng token và từng câu.

Head không chứa danh sách kiến thức riêng của từng từ. Nó là phép biến đổi dùng chung:

```text
x_đường W_Qh  → q_đường
x_cà_phê W_Qh → q_cà_phê
x_Phường W_Qh → q_Phường
```

Cùng weight nhưng input khác tạo output khác. Muốn biết head chú ý đâu phải chạy câu cụ thể và xem attention matrix ở đúng layer, head và hàng Query.

## 57. Ghép head giúp hiểu ngữ cảnh thế nào?

Mỗi head có thể thu thập một loại bằng chứng khác nhau. Concatenate giữ các output ở những vùng tọa độ riêng:

```text
Head 1 [64] | Head 2 [64] | ... | Head 12 [64]
→ [768]
```

W_O và các layer tiếp theo học cách phối hợp các bằng chứng. Với từ đường:

```text
đi trên + Nguyễn Trãi → bằng chứng cho nghĩa đường phố
cho vào + cà phê      → bằng chứng cho nghĩa thực phẩm
```

Không nên hiểu một head bằng một nghĩa. Nghĩa theo ngữ cảnh hình thành từ nhiều head, nhiều layer, FFN và residual.

## 58. Position embedding được thêm lúc nào?

Thứ tự:

```text
Token IDs
→ Token embedding
→ cộng Position embedding
→ X
→ tạo Q, K, V
→ reshape thành nhiều head
→ self-attention
```

Position được thêm trước attention, nên Q, K, V đã mang thông tin vị trí.

```text
Token embedding    → token là gì
Position embedding → token đứng ở đâu
Self-attention     → token lấy thông tin từ đâu
```

## 59. Weight của project lấy từ đâu?

Khi train Transformer từ đầu, các weight gần như ngẫu nhiên. Project này nên tải PhoBERT pretrained:

```text
Embedding matrix → checkpoint
W_Q, W_K, W_V   → checkpoint
FFN, LayerNorm   → checkpoint
NER classifier   → khởi tạo mới
```

Khi fine-tune toàn model:

```text
weight_new = weight_pretrained - learning_rate × gradient
```

Các weight không học lại từ số 0; chúng bắt đầu từ kiến thức tiếng Việt rồi thích nghi với Address NER.

# Phần V. BERT và PhoBERT

## 60. BERT là gì?

BERT là Transformer Encoder đã được pretrain trên lượng lớn văn bản:

```text
Transformer Encoder + pretraining = BERT
```

Bidirectional nghĩa là mỗi token có thể dùng ngữ cảnh cả bên trái và bên phải. BERT tạo contextual embedding cho từng token.

Pretraining học ngôn ngữ tổng quát từ văn bản không cần gán nhãn thủ công. Fine-tuning dùng dữ liệu có nhãn để học nhiệm vụ cụ thể.

## 61. BERT nhận và trả gì?

Tokenizer thêm special tokens:

```text
BERT:             [CLS] câu [SEP]
RoBERTa/PhoBERT:  <s> câu </s>
```

Nếu input có n token:

```text
input_ids shape        = [n]
last_hidden_state shape = [n,768]
```

Mỗi hàng output là contextual embedding của một token. Với classification cả câu có thể dùng vector special token đầu; với NER phải dùng output của từng token.

Attention mask:

```text
1 → token thật
0 → padding
```

Nó ngăn attention lấy thông tin từ padding.

## 62. Masked Language Modeling

Trong pretraining:

```text
Tôi sống tại <mask>
```

Model phải dự đoán:

```text
<mask> → Hà_Nội
```

Encoder trả vector 768 chiều ở vị trí mask. MLM head tạo logits cho toàn vocabulary:

```text
[768] → [vocab_size]
```

Cross-entropy so prediction với token gốc. Gradient cập nhật MLM head, Transformer và embeddings.

Khi fine-tune NER, MLM head được thay bằng:

```text
NER head: 768 → num_labels
```

## 63. Logit

Logit là điểm thô trước softmax. Nó có thể âm hoặc dương và không cần tổng bằng 1:

```text
STREET   → 1.0
WARD     → 2.0
DISTRICT → -1.0
```

Softmax đổi logits thành phân phối tương đối. CrossEntropyLoss nhận logits trực tiếp; không cần gọi softmax trước loss.

```text
Attention logits      → chọn token cung cấp thông tin
Classification logits → chọn nhãn cho token
```

## 64. Tính hộp đen

Ta biết rõ công thức, kiến trúc, weight, hidden states, attention và logits. Phần khó là giải thích một khái niệm cụ thể được phân tán ở đâu trong hàng triệu weight.

```text
Không hộp đen về cơ chế tính toán
Khá hộp đen về ý nghĩa phân tán trong weight
```

Attention visualization, gradient, probing và ablation hỗ trợ phân tích nhưng không tạo lời giải thích nhân quả hoàn chỉnh.

Với địa chỉ, model chỉ nên làm parser. Resolver minh bạch dùng danh mục hành chính, hierarchy, fuzzy matching và trạng thái AMBIGUOUS để kiểm chứng kết quả.

## 65. PhoBERT

PhoBERT là model Transformer Encoder theo hướng RoBERTa, được pretrain cho tiếng Việt. Checkpoint chứa:

```text
Vocabulary/tokenizer phù hợp tiếng Việt
Embedding matrix pretrained
Position embedding pretrained
Transformer weights pretrained
```

PhoBERT-base chưa tự có các nhãn HOUSE_NUMBER, STREET, WARD hoặc DISTRICT. Ta thêm token-classification head và fine-tune bằng dữ liệu địa chỉ BIO.

```text
PhoBERT Encoder
→ contextual embedding [768] mỗi token
→ classifier [768,num_labels]
→ logits
→ BIO label
```

## 66. Hai tầng tách văn bản tiếng Việt

Cần phân biệt:

```text
Word segmentation
→ xác định từ/cụm từ tiếng Việt

Subword tokenization
→ mã hóa từ bằng các token có trong vocabulary của model
```

Ví dụ:

```text
Văn bản gốc:
12 Nguyễn Trãi

Sau word segmentation:
12 Nguyễn_Trãi
```

Nguyễn_Trãi tồn tại như một chuỗi trong dữ liệu, nhưng không nhất thiết tồn tại nguyên vẹn trong vocabulary hữu hạn của tokenizer.

## 67. Vì sao tokenizer chia subword?

Nếu vocabulary có nguyên token:

```text
Nguyễn_Trãi → một token ID
```

Nếu không có nguyên từ nhưng có các mảnh:

```text
Nguyễn_@@
Trãi
```

tokenizer biểu diễn:

```text
Nguyễn_Trãi → Nguyễn_@@ | Trãi
```

Ký hiệu @@ cho biết mảnh trước chưa kết thúc từ và cần nối với mảnh sau. Đây là ví dụ minh họa; cách chia thực tế phụ thuộc vocabulary của đúng tokenizer.

Việc một chuỗi có trong input không đồng nghĩa nó phải có nguyên mục trong vocabulary. Con người có thể viết tên mới; tokenizer dùng các mảnh đã biết để mã hóa nó.

## 68. Vì sao có một nhãn nhưng nhiều subword?

Nhãn được con người gán ở cấp từ trước khi subword tokenizer chạy:

```text
Nguyễn_Trãi → B-STREET
```

Sau đó tokenizer mới chia:

```text
Nguyễn_Trãi → Nguyễn_@@ | Trãi
```

Vì vậy:

```text
Dữ liệu gốc:   1 từ, 1 nhãn
Model input:   2 subword token
```

Label alignment là quy tắc căn nhãn word-level với subword-level.

## 69. Hai cách label alignment

Cách 1, chỉ giám sát subword đầu:

```text
Nguyễn_@@ → B-STREET
Trãi      → -100
```

-100 yêu cầu CrossEntropyLoss bỏ qua vị trí đó. Transformer vẫn nhìn token Trãi nếu attention mask bằng 1.

Cách 2, truyền nhãn xuống mọi subword:

```text
Nguyễn_@@ → B-STREET
Trãi      → I-STREET
```

Quy tắc BIO:

```text
B-X → B-X, I-X, I-X, ...
I-X → I-X, I-X, I-X, ...
O   → O, O, O, ...
```

Đối với PoC ban đầu, chỉ tính loss ở subword đầu thường đơn giản và tránh việc từ bị chia nhiều lần có trọng số loss lớn hơn.

## 70. Attention mask khác loss mask

```text
attention_mask = 0
→ Transformer bỏ qua token, thường dùng cho padding

label = -100
→ Transformer vẫn dùng token tạo context
→ nhưng NER loss bỏ qua vị trí đó
```

Ví dụ:

```text
Token        Attention mask   Label
<s>                 1        -100
Nguyễn_@@           1        B-STREET
Trãi                 1        -100
</s>                 1        -100
<pad>                0        -100
```

## 71. Luồng fine-tuning Address NER

```text
Địa chỉ gốc
→ word segmentation
→ từ và nhãn BIO word-level
→ PhoBERT subword tokenizer
→ word-to-subword mapping
→ label alignment
→ input_ids + attention_mask + labels
→ PhoBERT Encoder
→ token-classification logits
→ CrossEntropyLoss tại vị trí hợp lệ
→ backpropagation
→ checkpoint Address NER
```

Trước khi train toàn bộ dữ liệu, cần in thử các trường:

```text
raw text
word
subword
token ID
word ID
label gốc
label đã align
attention mask
```

Sai label alignment không nhất thiết gây lỗi chương trình, nhưng khiến model học nhãn sai.

## 72. Định hướng thực hiện

Không train Transformer từ đầu. Nên:

```text
1. Tải PhoBERT hoặc checkpoint Address NER có sẵn.
2. Xây golden test set độc lập.
3. Kiểm tra tokenizer và label alignment.
4. Chạy baseline model có sẵn.
5. Chỉ fine-tune PhoBERT riêng nếu baseline chưa đạt.
6. Tách PhoBERT parser khỏi resolver hành chính.
```

Khi fine-tune custom model:

```text
Embedding và Transformer → khởi đầu từ pretrained checkpoint
Token classifier          → khởi tạo theo bộ nhãn mới
```

Có thể đóng băng Encoder trong một lượt ngắn để kiểm tra pipeline, sau đó mở toàn bộ Encoder và dùng learning rate nhỏ để fine-tune thực sự.

# Phần VI. Hoàn thiện Encoder và chuẩn bị dataset NER

## 73. Multi-Head Self-Attention và Feed-Forward

Một Transformer Encoder block không chỉ có attention:

```text
Input
→ Multi-Head Self-Attention
→ Residual + LayerNorm
→ Feed-Forward Network
→ Residual + LayerNorm
→ Output
```

Self-attention nói về nguồn của Q, K, V:

```text
Q, K, V đều được tạo từ cùng X
```

Multi-head nói về số phép attention song song. PhoBERT-base dùng Multi-Head Self-Attention.

Feed-Forward Network xử lý riêng từng token sau khi các token đã trao đổi thông tin:

```text
768 → Linear → 3072 → activation → Linear → 768
```

```text
Multi-head attention → trộn thông tin giữa các token
FFN                  → biến đổi feature bên trong từng token
```

## 74. Số head và weight được tạo khi nào?

Số head là hyperparameter được chọn trước pretraining:

```text
hidden size = 768
num heads   = 12
d_head      = 64
```

Sau đó model mới khởi tạo W_Q, W_K, W_V gần như ngẫu nhiên và train chúng trong cấu trúc 12 head cố định.

Trong code, một matrix lớn:

```text
W_Q shape = [768,768]
```

có thể được xem là:

```text
W_Q = [W_Q1 | W_Q2 | ... | W_Q12]
```

Mỗi khối có shape [768,64]. Các giá trị ban đầu khác nhau phá vỡ tính đối xứng; backpropagation tạo gradient khác nhau cho từng khối.

Không train W_Q xong rồi mới tùy ý chọn số head. Đổi số head sau pretraining làm thay đổi cách nhóm các chiều, scaling, attention pattern và cách W_O kết hợp output.

## 75. BERT và RoBERTa

RoBERTa gần như giữ Transformer Encoder của BERT nhưng tối ưu quy trình pretraining:

```text
BERT:
Masked Language Modeling + Next Sentence Prediction

RoBERTa:
Masked Language Modeling
bỏ Next Sentence Prediction
dynamic masking
nhiều dữ liệu và bước training hơn
```

RoBERTa không phát minh lại self-attention; cải thiện chủ yếu nằm ở chiến lược pretraining. PhoBERT đi theo hướng RoBERTa và được pretrain cho tiếng Việt.

## 76. Dataset Address NER ở cấp từ

Trước tokenizer, con người tạo dataset gồm danh sách từ và nhãn BIO có cùng độ dài.

```text
Tokens:
[12, Nguyễn_Trãi, ,, Phường, Bến_Thành]

Labels:
[B-HOUSE_NUMBER, B-STREET, O, B-WARD, I-WARD]
```

Bộ nhãn cơ bản:

```text
O
B/I-HOUSE_NUMBER
B/I-STREET
B/I-WARD
B/I-DISTRICT
B/I-PROVINCE
```

Quy tắc annotation phải nhất quán. Nếu chọn bao gồm tiền tố hành chính thì toàn bộ Phường Bến Thành phải được gán WARD ở mọi mẫu.

JSONL, CoNLL, CSV hay Parquet chỉ là định dạng lưu. JSONL thuận tiện cho PoC vì dễ đọc, giữ được danh sách token/nhãn và xử lý từng dòng.

## 77. Tokenization và word mapping

Tokenizer có thể thêm special token và chia một từ thành subword:

```text
Từ gốc:
Nguyễn_Trãi

Model tokens:
Nguyễn_@@ | Trãi
```

Code phải giữ mapping:

```text
<s>         → không có word ID
12          → word 0
Nguyễn_@@   → word 1
Trãi        → word 1
,           → word 2
Phường      → word 3
Bến_Thành   → word 4
</s>        → không có word ID
```

Hai subword có cùng word ID vì đều sinh từ Nguyễn_Trãi.

## 78. Label alignment

Nhãn được con người gán ở cấp từ trước khi subword tokenizer chạy:

```text
Nguyễn_Trãi → B-STREET
```

Sau khi tokenizer chia, code tự căn nhãn. Chiến lược đơn giản:

```text
Nguyễn_@@ → B-STREET
Trãi      → -100
```

-100 yêu cầu CrossEntropyLoss bỏ qua vị trí đó. Một chiến lược khác là:

```text
Nguyễn_@@ → B-STREET
Trãi      → I-STREET
```

Dataset gốc chỉ cần word-level BIO labels. Các label subword và -100 thường được tạo tự động trong preprocessing, không cần con người nhập thủ công.

## 79. Special tokens

Các special token phổ biến của PhoBERT/RoBERTa:

```text
<s>      → bắt đầu chuỗi
</s>     → kết thúc hoặc phân tách chuỗi
<pad>    → làm các câu trong batch bằng độ dài
<mask>   → che token khi pretraining
<unk>    → nội dung tokenizer không mã hóa được
```

Chúng có token ID và embedding nhưng không nhận nhãn thực thể trong Address NER:

```text
<s>, </s>, <pad> → label -100
```

Phân biệt:

```text
attention_mask = 0
→ Transformer bỏ qua token, thường là padding

label = -100
→ Transformer vẫn có thể dùng token tạo context
→ nhưng loss bỏ qua vị trí đó
```

## 80. Hai tầng chuẩn bị nhãn

```text
Tầng 1, con người:
tạo word-level tokens và BIO labels

Tầng 2, code:
tokenize, thêm special tokens, chia subword
và tạo model-level labels
```

Luồng:

```text
Địa chỉ gốc
→ word segmentation
→ con người gán BIO ở cấp từ
→ lưu dataset
→ PhoBERT tokenizer
→ word-to-subword mapping
→ label alignment tự động
→ input_ids + attention_mask + labels
```

Đến đây mới hoàn thành dữ liệu đầu vào cho model; chưa thực hiện forward pass hay cập nhật weight.


---

# Lý thuyết NLP - Phần bổ sung

Tài liệu này nối tiếp `LY_THUYET_NLP_EMBEDDING_TRANSFORMER.md`, đi từ forward pass chi tiết của PhoBERT đến fine-tuning, đánh giá NER, resolver và vận hành hệ thống chuẩn hóa địa chỉ.

## 81. Đánh giá NER

Accuracy không phải chỉ số chính vì nhãn `O` thường chiếm đa số. Nếu 90/100 token là `O`, model đoán tất cả là `O` vẫn đạt accuracy 90% nhưng không tìm được thực thể nào.

```text
TP: thực thể được dự đoán và khớp đúng nhãn thật
FP: model dự đoán một thực thể nhưng thực thể đó sai
FN: thực thể thật tồn tại nhưng model bỏ sót

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Với NER cần đánh giá ở mức thực thể. Loại và toàn bộ ranh giới phải đúng:

```text
Nhãn thật: Phường/B-WARD Bến_Thành/I-WARD
Dự đoán:   Phường/B-WARD Bến_Thành/O
=> sai thực thể WARD
```

Nên báo cáo entity-level Precision/Recall/F1 tổng thể, F1 từng loại thực thể, validation loss và confusion matrix.

## 82. Confidence và giới hạn của NER

Token classifier tạo logits; softmax biến chúng thành phân bố xác suất:

```text
logits -> softmax -> xác suất từng nhãn
```

Xác suất lớn nhất thường được dùng làm confidence của token. Confidence cao không bảo đảm đúng; cần kiểm tra calibration và chọn threshold trên validation set.

PhoBERT NER có nhiệm vụ tách và phân loại:

```text
12               -> HOUSE_NUMBER
nguyen trai      -> STREET
phuong ben thanh -> WARD
quan 1           -> DISTRICT
```

NER có thể nhận ra chuỗi không dấu hoặc sai nhẹ nếu dữ liệu train có ví dụ tương tự, nhưng không nên chịu trách nhiệm sửa `ben thnah` thành `Bến Thành`. Việc đưa chuỗi về tên chính thức thuộc resolver:

```text
Raw address -> PhoBERT NER -> Resolver -> Canonical address
```

## 83. Xây dựng dataset địa chỉ

Khó có sẵn một dataset sạch đúng các nhãn `HOUSE_NUMBER`, `STREET`, `WARD`, `DISTRICT`, `PROVINCE`. Nên kết hợp:

```text
Danh mục hành chính chính thức
Dữ liệu đường/địa chỉ từ OpenStreetMap hoặc nguồn địa phương
Dữ liệu thật đã ẩn thông tin cá nhân
Dữ liệu synthetic sinh từ dữ liệu có cấu trúc
```

Danh mục resolver nên có:

```text
administrative_code, official_name, normalized_name, unit_type,
parent_code, effective_from, effective_to, status
```

Phải giữ cả danh mục hiện hành và lịch sử để xử lý tên cũ.

### 83.1 Sinh BIO tự động

Từ bản ghi có cấu trúc:

```json
{
  "house_number": "12",
  "street": "Nguyễn Trãi",
  "ward": "Phường Bến Thành",
  "district": "Quận 1",
  "province": "TP Hồ Chí Minh"
}
```

Sinh token và nhãn đồng thời:

```text
12             B-HOUSE_NUMBER
Nguyễn_Trãi    B-STREET
,              O
Phường         B-WARD
Bến_Thành      I-WARD
,              O
Quận           B-DISTRICT
1              I-DISTRICT
,              O
TP             B-PROVINCE
Hồ_Chí_Minh    I-PROVINCE
```

Không nên tạo xong chuỗi rồi tìm substring để gán nhãn vì tên trùng và biến thể dễ làm lệch span. Biến thể cần có: không dấu, viết tắt, typo nhẹ, thiếu thành phần, đổi thứ tự, dấu câu bất thường và tên hành chính cũ.

### 83.2 Chia và kiểm tra dữ liệu

Có thể khởi đầu với `80/10/10`. Phải chia bản ghi gốc trước, rồi mỗi tập tự sinh biến thể. Không để các biến thể của cùng địa chỉ xuất hiện ở cả train và test.

Tự động kiểm tra 100%:

```text
len(tokens) == len(labels)
Nhãn thuộc label set
I-X chỉ nối sau B-X hoặc I-X
Không có chuỗi rỗng/duplicate ngoài ý muốn
Phân bố nhãn và độ dài hợp lý
```

Kiểm tra thủ công để phát hiện lỗi ngữ nghĩa, ví dụ `Nguyễn_Trãi` bị gán `B-WARD`. Nên kiểm tra toàn bộ test set nếu quy mô cho phép.

## 84. Tensor đầu vào

Tensor là mảng nhiều chiều tối ưu cho neural network, GPU và automatic differentiation.

```text
input_ids      [32,20]
attention_mask [32,20]
labels         [32,20]
```

`32` là batch size; `20` là sequence length. `input_ids` chứa token ID; `attention_mask` phân biệt token thật với padding; `labels` chứa đáp án NER. `-100` loại special token, padding hoặc subword phụ khỏi token-classification loss.

## 85. Forward pass chi tiết của PhoBERT-base

Ký hiệu:

```text
B=32, L=20, H=768, heads=12, d_k=64, labels=11
```

### 85.1 Embedding và position

```text
input_ids [32,20]
-> token embedding [32,20,768]
```

Model tự tạo position IDs và tra position embedding matrix:

```text
[0,1,...,19] -> position embeddings [32,20,768]
```

```text
X0 = token_embeddings + position_embeddings
X0 -> LayerNorm -> Dropout
X0 [32,20,768]
```

Position embedding là weight đã học trong pretraining và lưu trong checkpoint. RoBERTa/PhoBERT có thể dùng offset cho padding, nhưng bản chất vẫn là position ID tra một hàng embedding.

### 85.2 LayerNorm và Dropout

LayerNorm chuẩn hóa riêng vector 768 chiều của từng token:

```text
x_hat  = (x - mean(x)) / sqrt(variance(x) + epsilon)
output = gamma * x_hat + beta
```

`gamma`, `beta` là weight học được. Với `[32,20,768]`, có 640 vector token được chuẩn hóa độc lập. LayerNorm không trộn token hay mẫu trong batch.

Trong training, Dropout ngẫu nhiên tắt một tỉ lệ activation và scale phần còn lại để giảm overfitting:

```text
model.train() -> Dropout hoạt động
model.eval()  -> Dropout tắt
```

### 85.3 Một Encoder block

```text
Multi-Head Self-Attention
-> Dropout + Residual + LayerNorm
-> Feed-Forward Network
-> Dropout + Residual + LayerNorm
```

Tạo Q, K, V theo cách gộp:

```text
X [32,20,768]
W_Q, W_K, W_V [768,768]
Q, K, V       [32,20,768]
-> reshape    [32,12,20,64]
```

Theo từng head, mỗi phép chiếu có weight `[768,64]` và tạo output `[32,20,64]`. Hai cách nhìn tương đương.

Attention:

```text
QK^T/sqrt(64) -> [32,12,20,20]
-> cộng attention mask
-> softmax
-> attention weights

attention weights x V
-> [32,12,20,64]
-> concatenate -> [32,20,768]
-> W_O -> [32,20,768]
```

Ma trận `[20,20]` chỉ là attention weights trung gian, không phải output cuối.

FFN:

```text
Linear(768 -> 3072)
-> GELU
-> Linear(3072 -> 768)
-> Dropout + Residual + LayerNorm
```

Self-attention trộn thông tin giữa các token; FFN biến đổi feature bên trong từng token.

### 85.4 Mười hai Encoder block

PhoBERT-base không chạy một attention rồi thêm 12 block. Nó có 12 block, mỗi block chứa attention và FFN:

```text
X0 -> Attention1 -> FFN1 -> X1
X1 -> Attention2 -> FFN2 -> X2
...
X11 -> Attention12 -> FFN12 -> X12
```

Block 2 không nhận trực tiếp `Q1,K1,V1`. Nó nhận output hoàn chỉnh `X1 [32,20,768]`, rồi dùng weight riêng:

```text
Q2=X1W_Q2, K2=X1W_K2, V2=X1W_V2
```

Các head của block trước đã được ghép về 768 chiều; block sau tạo 12 head mới.

### 85.5 Classification head

```text
X12 [32,20,768]
-> Dropout
-> Linear(768 -> 11)
-> logits [32,20,11]
-> argmax
-> prediction [32,20]
```

Tensor cuối chứa ID nhãn NER, không còn là token ID.

## 86. Training: batch, step, epoch

Một step:

```text
Forward -> loss -> backward -> optimizer update
```

```python
optimizer.zero_grad()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
```

Một epoch là một lần đi qua toàn bộ train set. Với 10.000 mẫu và batch 32 có khoảng 313 step/epoch.

## 87. Gradient, optimizer và scheduler

```text
gradient = dLoss/dweight
weight_new = weight_old - learning_rate * gradient
```

Với bước nhỏ:

```text
Loss(w + delta_w) gần bằng Loss(w) + gradient*delta_w
delta_w = -learning_rate*gradient
=> Loss_new gần bằng Loss_old - learning_rate*gradient^2
```

Đây là hướng giảm cục bộ khi learning rate đủ nhỏ. Gradient được tính lại mỗi step vì weight, batch, prediction và loss thay đổi. Gradient có thể đổi dấu sau khi đi qua điểm cực tiểu. Learning rate quá lớn gây overshoot.

PyTorch cộng dồn gradient, nên thường gọi `optimizer.zero_grad()` trước `backward()`. Backprop chỉ tính gradient; optimizer mới thay đổi weight.

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

Learning rate thường warmup rồi decay. Warmup bảo vệ pretrained weights ở các step đầu; decay tạo bước nhỏ hơn khi tinh chỉnh gần vùng tốt.

## 88. Validation và checkpoint

```python
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
```

`model.eval()` tắt Dropout; `torch.no_grad()` không tạo graph gradient. Hai thao tác không thay thế nhau. Trong Keras, `predict()` và `evaluate()` thực hiện ý tưởng tương đương tự động.

Overfitting thường có dạng:

```text
Train loss giảm
Validation loss tăng
Validation F1 giảm
```

Lưu checkpoint có validation entity-level F1 tốt nhất. Test set chỉ dùng sau khi khóa model, hyperparameter và threshold.

## 89. Inference

```text
Raw address
-> normalization cơ bản
-> word segmentation
-> tokenizer
-> PhoBERT logits
-> argmax
-> BIO tags
-> entities
```

Ghép `B-WARD` cùng các `I-WARD` liền sau thành một thực thể. Kết quả vẫn là chuỗi người dùng nhập, chưa phải tên chính thức.

## 90. Resolver

```text
Normalization
-> Exact matching
-> Alias matching
-> Fuzzy candidate generation
-> Hierarchy validation
-> Ranking/confidence
-> ACCEPTED, REVIEW hoặc UNRESOLVED
```

### 90.1 Normalization và alias

Áp dụng Unicode normalization, lowercase, chuẩn hóa dấu câu/khoảng trắng và tạo khóa không dấu. Không hard-code từng địa chỉ; chỉ cần alias domain nhỏ:

```text
p, p. -> phường
q, q. -> quận
tp    -> thành phố
tx    -> thị xã
tt    -> thị trấn
```

Luôn giữ input gốc.

### 90.2 Fuzzy matching

Levenshtein distance đếm số phép thêm, xóa hoặc thay ký tự ít nhất:

```text
ben than -> ben thanh
distance=1
similarity=1-distance/max_length=1-1/9 khoảng 0.889
```

Damerau-Levenshtein hỗ trợ lỗi đảo ký tự; Jaro-Winkler phù hợp tên ngắn; token similarity hỗ trợ khác thứ tự từ. Fuzzy score là độ giống chuỗi, không phải xác suất đúng.

### 90.3 Hierarchy và confidence

Mỗi đơn vị cần code và parent code. Resolve từ cấp lớn xuống nhỏ để giới hạn candidate. Nếu thiếu cấp cha, chỉ suy ra khi có một ứng viên rõ và đánh dấu `INFERRED`; nhiều ứng viên phải trả `AMBIGUOUS`.

Confidence cuối kết hợp:

```text
NER confidence
Fuzzy similarity
Exact/alias flag
Hierarchy consistency
Top1-top2 margin
Mức đầy đủ của địa chỉ
```

Top1 0.93 và top2 0.92 vẫn mơ hồ. Có thể dùng rule trước, sau đó train logistic regression từ kết quả đúng/sai để tạo confidence đã calibration.

Trạng thái nên có:

```text
EXACT, ALIAS, FUZZY_MATCHED, INFERRED, AMBIGUOUS, UNRESOLVED
```

## 91. Đánh giá resolver và end-to-end

```text
Top-1 accuracy = candidate đầu đúng / tổng mẫu
Coverage = mẫu tự động chấp nhận / tổng mẫu
Accepted precision = mẫu đúng trong nhóm tự động chấp nhận / nhóm tự động chấp nhận
```

Tự động chấp nhận nghĩa là dùng kết quả mà không cần người kiểm tra. Threshold cao thường làm coverage giảm và accepted precision tăng. Chọn threshold trên validation theo yêu cầu nghiệp vụ; không chọn trên test.

End-to-end cần đo accuracy từng field và full-address exact match, ngoài NER F1 và resolver metrics.

## 92. Human-in-the-loop, monitoring và versioning

Mẫu `REVIEW` là nguồn active learning tốt. Lưu input, NER prediction, resolver candidates, confidence và correction. Nếu NER sai loại thì bổ sung dataset NER; nếu NER đúng nhưng resolver sai thì sửa alias, fuzzy, hierarchy hoặc confidence.

Không train ngay theo từng phản hồi. Thu thập, kiểm tra, tạo dataset version mới, train/evaluate rồi mới deploy.

Theo dõi:

```text
ACCEPTED/REVIEW/UNRESOLVED rate
Confidence distribution
Tỉ lệ correction
Accepted precision qua audit định kỳ
```

Lưu `model_version`, `dataset_version`, `resolver_version`, `admin_catalog_version` và `threshold_version`.

## 93. Cấu trúc project đề xuất

```text
address-system/
|-- data/{raw,interim,processed}/
|-- src/{data,training,inference,resolver,evaluation}/
|-- configs/
|-- checkpoints/
|-- tests/
`-- requirements.txt
```

Thứ tự triển khai:

```text
1. Chốt schema và annotation guideline
2. Viết dataset validator
3. Tạo dataset nhỏ vài trăm mẫu
4. Chạy thử end-to-end
5. Khi pipeline đúng mới mở rộng dữ liệu
```

Mốc đầu tiên:

```text
JSONL -> tokenizer -> tensor -> PhoBERT -> loss/prediction
-> BIO entities -> resolver -> canonical result + status
```
