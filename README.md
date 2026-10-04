# Address NER Baseline

Baseline từng bước cho bài toán Vietnamese address NER bằng PhoBERT.

## Cách cấu hình

Project không dùng `argparse`. Mỗi module chạy trực tiếp bằng:

```powershell
$env:PYTHONPATH = "src"
python -m address_ner.<ten_module>
```

Không truyền đường dẫn hoặc option sau tên module. Trước khi chạy, mở file tương
ứng trong `src/address_ner` và sửa khối `Configuration` ở đầu file.

Ví dụ trong `train.py`:

```python
TRAIN_FILE = Path("data/synthetic/processed-v2-stratified/train.jsonl")
VALIDATION_FILE = Path("data/synthetic/processed-v2-stratified/validation.jsonl")
OUTPUT_DIR = Path("checkpoints/synthetic-v3")
BATCH_SIZE = 4
EPOCHS = 3
```

## Học pipeline cơ bản

Ba module này lần lượt cho thấy tokenization, dynamic padding và một forward
pass qua PhoBERT:

```powershell
python -m address_ner.tokenize_dataset
python -m address_ner.data_loader
python -m address_ner.forward_pass
```

Các file này mặc định đọc `data/sample.jsonl`. Dataset word-level chưa có
special token, subword hoặc nhãn `-100`; tokenizer tạo chúng khi căn nhãn.

Kiểm tra dataset và chạy test:

```powershell
python -m address_ner.validate_dataset
python -m unittest discover -s tests -v
```

## Dữ liệu thật và annotation

`curate_raw_addresses.py` đọc TXT hoặc CSV, làm sạch, loại trùng và tạo báo
cáo. Với CSV, đặt `CSV_ADDRESS_COLUMN` nếu không muốn tự dò tên cột.

```powershell
python -m address_ner.curate_raw_addresses
python -m address_ner.prepare_annotation
```

Các gợi ý tự động vẫn có trạng thái `needs_review`. Chỉ mẫu được con người
chuyển thành `approved` mới được đổi sang BIO để train.

## Pipeline synthetic hiện tại

Các hằng số mặc định tạo chuỗi dữ liệu:

```text
annotations.jsonl
-> annotations-v2.jsonl
-> bio-v2.jsonl
-> processed-v2-stratified/
-> checkpoints/synthetic-v3
```

Chạy lần lượt:

```powershell
python -m address_ner.generate_synthetic
python -m address_ner.generate_hard_cases
python -m address_ner.validate_annotations
python -m address_ner.convert_annotations
python -m address_ner.split_dataset
python -m address_ner.train
```

`generate_hard_cases` bổ sung negative, địa chỉ legacy, tiền tố đường và số
nhà phức tạp. `split_dataset` chia theo `group_id` và
`generation.kind` để tránh biến thể của cùng một địa chỉ lọt sang nhiều split.

Synthetic data chỉ dùng để bootstrap. Điểm trên synthetic test đo khả năng học
phân phối của bộ sinh, không thay thế đánh giá trên địa chỉ thật.

## Đánh giá model

Đánh giá test split, chạy inference từng record và đánh giá golden set:

```powershell
python -m address_ner.evaluate
python -m address_ner.inference
python -m address_ner.golden_cases
python -m address_ner.evaluate_golden
```

Golden benchmark là tập kiểm tra cố định và không được đưa vào train. File seed
đã được dùng trong quá trình phát triển nên chỉ là development set; đánh giá cuối
cần holdout địa chỉ thật chưa từng được xem trước.

## Catalog hành chính

Tải catalog hiện hành và bảng đối chiếu trước/sau ngày 01/07/2025:

```powershell
python -m address_ner.fetch_admin_catalog
python -m address_ner.fetch_legacy_catalog
python -m address_ner.build_legacy_mappings
```

Các module tải dữ liệu có truy cập mạng. Đường dẫn output và ngày đối chiếu nằm
trong khối cấu hình của từng file.

`DIRECT_CODE` và một đích `NOTE_FULL` có thể tạo `WARNING`.
`NOTE_PARTIAL`, nhiều đích hoặc xung đột tạo `AMBIGUOUS`; pipeline không tự
chọn đơn vị hiện hành khi bằng chứng không đủ.

## Đánh giá legacy mapping

```powershell
python -m address_ner.evaluate_legacy_mappings
```

Để tạo blinded holdout, đặt trong `legacy_holdout.py`:

```python
MODE = "prepare"
```

rồi chạy:

```powershell
python -m address_ner.legacy_holdout
```

Sau khi review, đổi `MODE = "finalize"` và chạy lại cùng lệnh. Muốn đánh giá
holdout vừa tạo, đổi `DATASET_PATH` và `OUTPUT_PATH` trong
`evaluate_legacy_mappings.py`.

## Admin resolver

Đặt `WARD_TEXT`, `PROVINCE_TEXT` và confidence trong `resolver.py`, sau đó:

```powershell
python -m address_ner.resolver
```

Resolver thử alias chính xác, alias sau khi bỏ dấu, rồi fuzzy matching. Kết quả
dùng các trạng thái `PASS`, `WARNING`, `AMBIGUOUS`, `REJECTED` và trả
tối đa ba ứng viên kèm điểm.

Đánh giá resolver:

```powershell
python -m address_ner.evaluate_resolver
```

Để quét grid threshold và margin, đặt `CALIBRATE = True` trong
`evaluate_resolver.py`.

## End-to-end inference

Đặt chuỗi cần xử lý trong `INPUT_TEXT` của `pipeline.py`, rồi chạy:

```powershell
python -m address_ner.pipeline
```

Pipeline thực hiện:

```text
chuỗi thô
-> làm sạch và word segmentation
-> PhoBERT NER
-> gom entity
-> current/legacy resolver
-> kết quả chuẩn hóa và issues
```

Output chứa entity parser nhận được, mã ward/province hiện hành, parser
confidence, resolver confidence, danh sách ứng viên và các cảnh báo.
