---

# YOLOv8 Pose Inference API

FastAPI service dùng YOLOv8 Pose để detect người và keypoints từ ảnh.

Service này được thiết kế như **inference worker**:

* Nhận ảnh qua HTTP
* Trả kết quả pose (bbox + keypoints)
* Không xử lý orchestration / scale (để Kafka / hệ thống khác lo)

---

## 1. Yêu cầu môi trường

* Python ≥ 3.9
* Linux (khuyến nghị)
* CPU hoặc GPU đều chạy được
  (GPU phụ thuộc vào môi trường torch/cuda, không cấu hình trong repo này)

---

## 2. Cấu trúc thư mục

```
api/
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
└── yolov8n-pose.pt        # YOLOv8 pose model weights
```

> Lưu ý: file `yolov8n-pose.pt` phải nằm cùng thư mục với `main.py`.

---

## 3. Cài đặt

### 3.1. Tạo virtualenv (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate
```

### 3.2. Cài dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Chạy service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Sau khi chạy:

* API: `http://localhost:8000`
* Swagger UI: `http://localhost:8000/docs`

Khi startup, service sẽ:

* Load model YOLOv8 Pose
* Chạy warmup 1 ảnh dummy để tránh request đầu bị chậm

---

## 5. API Endpoint

### `POST /v1/pose`

Infer pose từ một ảnh.

#### Content-Type

```
multipart/form-data
```

---

### 5.1. Request fields

| Field      | Type   | Required | Description                          |
| ---------- | ------ | -------- | ------------------------------------ |
| `image`    | file   | Yes      | Ảnh đầu vào (jpg, png, …)            |
| `cam_id`   | string | Yes      | ID camera / nguồn ảnh                |
| `ts`       | string | Yes      | Timestamp (nhiều format được hỗ trợ) |
| `image_id` | string | No       | Nếu không gửi, server tự sinh        |
| `conf`     | float  | No       | Confidence threshold (default: 0.25) |
| `iou`      | float  | No       | IoU threshold cho NMS (default: 0.7) |
| `max_det`  | int    | No       | Số detection tối đa (default: 300)   |

---

### 5.2. Timestamp `ts` hỗ trợ

* Epoch seconds:

  ```
  1700000000
  ```
* Epoch milliseconds:

  ```
  1700000000000
  ```
* Float seconds:

  ```
  1700000000.123
  ```
* ISO8601 có timezone:

  ```
  2026-01-14T10:30:00Z
  2026-01-14T10:30:00+07:00
  ```
* ISO8601 không timezone:

  ```
  2026-01-14T10:30:00
  ```

  (sẽ được hiểu là UTC)

Response sẽ **echo lại đúng chuỗi `ts` input**, không trả timestamp đã normalize.

---

## 6. Response format

```json
{
  "cam_id": "cam01",
  "ts": "2026-01-14T10:30:00Z",
  "image_id": "cam01-1705228200000-acde123456",
  "inference_ms": 12.34,
  "objects": [
    {
      "class_id": 0,
      "class_name": "person",
      "conf": 0.92,
      "bbox_xyxy": [x1, y1, x2, y2],
      "keypoints": [
        [x, y, conf],
        [x, y, conf]
      ],
      "bbox_meta": {
        "has_helmet": null,
        "has_vest": null,
        "falling": null
      }
    }
  ]
}
```

### Ghi chú

* `bbox_xyxy`: `[x_min, y_min, x_max, y_max]`
* `keypoints`: danh sách keypoints `[x, y, score]`
* `bbox_meta`: placeholder cho các nhãn bổ sung (hiện tại luôn `null`)
* `objects` có thể là mảng rỗng nếu không detect được người

---

## 7. Ví dụ gọi API

### cURL

```bash
curl -X POST "http://localhost:8000/v1/pose" \
  -F "image=@/path/to/image.jpg" \
  -F "cam_id=cam01" \
  -F "ts=2026-01-14T10:30:00Z"
```

---

## 8. Ghi chú thiết kế

* Service này **chỉ làm inference**
* Không xử lý batching, queue, scale
* Có thể được đặt sau Kafka / message queue / load balancer
* Model được load một lần khi startup

---

## 9. Troubleshooting nhanh

* **Không load được model**
  → Kiểm tra file `yolov8n-pose.pt` có tồn tại trong thư mục chạy hay không

* **Lỗi OpenCV / libGL**
  → Đảm bảo đã cài `opencv-python-headless` (đã có trong requirements)

* **Request đầu chậm**
  → Bình thường, model đã warmup khi startup

---
