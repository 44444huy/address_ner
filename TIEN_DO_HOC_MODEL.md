# Tiến độ đọc hiểu PhoBERT Address NER

Tài liệu này ghi lại những phần đã đọc và đã hiểu trong project, đồng thời đánh
dấu vị trí hiện tại để lần sau tiếp tục đúng chỗ.

## 1. Kiến trúc tổng quát

Luồng xử lý của model:

```text
Văn bản
-> tokenizer
-> token ID
-> token embedding + position embedding
-> 12 Transformer Encoder block
-> contextual embedding [batch, sequence, 768]
-> token classifier
-> logits [batch, sequence, num_labels]
```

Các nội dung đã hiểu:

- Embedding matrix ánh xạ mỗi token ID thành một vector.
- Position embedding bổ sung thông tin vị trí của token.
- Self-attention dùng Query, Key và Value.
- Multi-head attention có nhiều bộ weight riêng để học nhiều quan hệ.
- Mỗi encoder block gồm attention, feed-forward, residual và LayerNorm.
- PhoBERT là backbone pretrained tiếng Việt do VinAI phát hành.
- Thư viện Hugging Face `transformers` cung cấp code để tải tokenizer và model.
- Token-classification head nhận vector của từng token và tạo logits cho các nhãn NER.
- Classification head mới phải được fine-tune cho bài toán địa chỉ.

## 2. Dữ liệu NER

Một record word-level có dạng:

```python
{
    "id": "addr_001",
    "tokens": ["Phường", "Bến_Thành"],
    "ner_tags": ["B-WARD", "I-WARD"],
}
```

Phân biệt:

```text
record -> dữ liệu một địa chỉ dạng chữ và nhãn BIO
sample -> một record đã được biến thành tensor
batch  -> nhiều sample được padding và ghép lại
```

Đã hiểu sự khác nhau giữa:

```text
tokens -> các từ word-level trong dataset
pieces -> các subword do PhoBERT tokenizer tạo ra
```

Ví dụ căn nhãn subword:

```text
Nguyễn_Trãi -> Nguyễn_@@ | Trãi

Nguyễn_@@ -> B-STREET
Trãi       -> -100
```

`-100` là `IGNORE_INDEX`. Các vị trí này không tham gia tính loss, gồm:

- BOS và EOS.
- Padding.
- Subword không phải subword đầu tiên của một word.

## 3. Tokenization

Đã đọc vòng lặp:

```python
for word_index, (word, tag) in enumerate(
    zip(record["tokens"], record["ner_tags"])
):
    pieces = tokenizer.tokenize(word)
```

Đã hiểu:

- `zip()` ghép từng word với nhãn tương ứng.
- `enumerate()` bổ sung vị trí `word_index`.
- `append()` thêm một phần tử.
- `extend()` thêm lần lượt nhiều phần tử.
- Tokenizer đổi word thành subword rồi thành token ID.
- Label ID được gán cho subword đầu tiên; phần còn lại nhận `-100`.

## 4. Dataset và DataLoader

`AddressNerDataset` biến record thành sample:

```text
record word-level
-> tokenize thành subword
-> đổi subword thành token ID
-> đổi nhãn BIO thành label ID
-> thêm BOS/EOS
-> chuyển list thành tensor
-> sample
```

`DynamicPaddingCollator` là class tự viết trong project. Nó padding các sample
trong cùng batch theo sample dài nhất:

```text
input_ids      -> pad_token_id
attention_mask -> 0
labels         -> -100
```

`build_dataloader()` tạo PyTorch `DataLoader` với collator trên.

Đã hiểu:

```python
batch = next(iter(dataloader))
```

- `iter(dataloader)` tạo iterator.
- `next(...)` lấy batch đầu tiên.

## 5. Dictionary và tensor

Đã phân biệt:

```text
dictionary.items() -> lấy các cặp key-value
tensor.item()       -> lấy một số Python từ tensor một phần tử
```

Đoạn:

```python
batch = {
    name: tensor.to(device)
    for name, tensor in batch.items()
}
```

tạo dictionary mới có cùng key nhưng các tensor đã được chuyển sang thiết bị
đích.

Mọi tensor đều có thuộc tính `device`:

```text
cpu    -> tensor nằm trong RAM
cuda:0 -> tensor nằm trong VRAM của GPU đầu tiên
```

## 6. Forward pass

Đã đọc:

```python
model.eval()

with torch.no_grad():
    outputs = model(**batch)
```

Ý nghĩa:

```text
model.eval()    -> chuyển model sang chế độ đánh giá
torch.no_grad() -> không tạo computation graph và không lưu gradient
model(**batch)  -> chạy forward pass
```

`model(**batch)` tương đương:

```python
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
```

Vì có `labels`, output chứa:

```python
outputs.logits
outputs.loss
```

Shape đã hiểu:

```text
input_ids:   [batch, sequence]
logits:      [batch, sequence, num_labels]
predictions: [batch, sequence]
```

```python
predictions = outputs.logits.argmax(dim=-1)
```

`argmax(dim=-1)` chọn label ID có logit lớn nhất cho từng token.

## 7. Phần đầu của training

Trong `train.py`, đã đọc:

```text
các import
các hyperparameter
set_seed()
move_batch()
supervised_token_count()
phần đầu train_one_epoch()
```

`set_seed()` cố định randomness của Python, NumPy, PyTorch CPU và CUDA để các
lần thử nghiệm dễ tái hiện.

`move_batch()` đưa tất cả tensor trong batch lên cùng device với model.

`supervised_token_count()`:

```python
def supervised_token_count(labels: torch.Tensor) -> int:
    return int((labels != IGNORE_INDEX).sum().item())
```

Hàm này:

1. Tìm các label khác `-100`.
2. Tạo tensor Boolean.
3. Đếm số `True`.
4. Chuyển tensor một phần tử thành số nguyên Python.

Đã đọc đến:

```python
model.train()
weighted_loss = 0.0
token_count = 0
```

`model.train()` bật chế độ training, đặc biệt là Dropout.

## 8. Batch, loss và gradient

Batch nhỏ vẫn xem toàn bộ dataset, nhưng chia dataset thành nhiều bước cập nhật
hơn. Với `N` record:

```text
số batch mỗi epoch = ceil(N / batch_size)
```

Trong token classification, cross-entropy lấy trung bình trên các token hợp lệ,
không tính vị trí có label `-100`.

Loss trung bình của batch:

```text
L_batch = (L_1 + L_2 + ... + L_N) / N
```

Đạo hàm của loss trung bình bằng trung bình các gradient:

```text
gradient_batch = (g_1 + g_2 + ... + g_N) / N
```

Batch nhỏ lấy trung bình trên ít mẫu hoặc token hơn nên gradient dễ dao động
hơn. Batch lớn thường cho gradient ổn định hơn.

## 9. Backpropagation và cập nhật weight

Công thức gradient descent cơ bản:

```text
weight_new = weight_old - learning_rate * gradient
```

Trong đó:

```text
gradient = đạo hàm của loss theo weight tại weight và batch hiện tại
```

Đã phân biệt:

```text
loss.backward()  -> tính gradient và lưu trong weight.grad
optimizer.step() -> dùng gradient để cập nhật weight
```

Project sử dụng AdamW. AdamW còn theo dõi trung bình gradient, trung bình bình
phương gradient và áp dụng weight decay, nhưng ý tưởng cốt lõi vẫn là cập nhật
weight theo hướng làm loss giảm.

## 10. Vị trí hiện tại

Đã đọc xong:

```text
labels.py
validate_dataset.py
tokenize_dataset.py
data_loader.py
forward_pass.py
```

Đang đọc:

```text
train.py
```

Vị trí chính xác là ngay trước vòng lặp:

```python
for batch in dataloader:
    batch = move_batch(batch, device)
    optimizer.zero_grad()
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()
```

Phần tiếp theo cần học là cách một batch thực hiện đầy đủ:

```text
đưa dữ liệu lên GPU
-> xóa gradient cũ
-> forward
-> tính loss
-> backward
-> optimizer cập nhật weight
```

Sau đó mới đến:

```text
evaluate_loss()
main() của train.py
vòng lặp nhiều epoch
lưu checkpoint tốt nhất
inference.py
evaluate.py
resolver.py
pipeline.py
```
