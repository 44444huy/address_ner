# Review legacy mapping holdout

Chỉ mở `legacy_mapping_holdout.blind.jsonl` khi review. Không xem file manifest hoặc `legacy_mappings_2025.csv` trước khi hoàn thành nhãn để tránh bị dự đoán hiện tại dẫn dắt.

Mỗi dòng là một đơn vị cũ, gồm ghi chú trên dòng nguồn và các ghi chú hiện hành có tên liên quan. Điền khối `review` theo quy tắc:

- `WARNING`: bằng chứng chính thức nói rõ toàn bộ đơn vị cũ đi vào đúng một đơn vị hiện hành.
- `AMBIGUOUS`: chỉ chuyển một phần, có nhiều đích, hoặc dòng nguồn chỉ nêu một đích có khả năng nhưng không chứng minh toàn bộ địa bàn.
- `REJECTED`: nguồn được cung cấp không đủ bằng chứng để nêu bất kỳ đích nào.

Ví dụ một đích toàn phần:

```json
"review": {
  "status": "WARNING",
  "current_ward_code": "02239",
  "candidate_ward_codes": ["02239"],
  "reviewer_note": "Hợp nhất toàn bộ vào Xã Thượng Nông"
}
```

Ví dụ chia tách:

```json
"review": {
  "status": "AMBIGUOUS",
  "current_ward_code": null,
  "candidate_ward_codes": ["11329", "11359"],
  "reviewer_note": "Địa bàn cũ được chia cho hai phường"
}
```

Không suy luận từ tên giống nhau. Nếu evidence chưa đủ, dùng `REJECTED`; không tra file prediction hiện tại. Sau khi review đủ 50 dòng:

```powershell
python -m address_ner.legacy_holdout finalize
python -m address_ner.evaluate_legacy_mappings `
  data/golden/legacy_mapping_cases.holdout.jsonl `
  --output reports/legacy_mappings.holdout.json
```

Lệnh `finalize` sẽ từ chối nếu còn case chưa review hoặc cấu trúc nhãn không hợp lệ.
