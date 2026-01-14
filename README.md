
# YOLOv8 Pose Inference API

FastAPI service dùng YOLOv8 Pose để detect người và keypoints từ ảnh.

Service này được thiết kế như **inference worker**:
- Nhận ảnh qua HTTP
- Trả kết quả pose (bbox + keypoints)
- Không xử lý orchestration / scale (để Kafka / hệ thống khác lo)


## 1. Yêu cầu môi trường

- Python ≥ 3.9
- Linux (khuyến nghị)
- CPU hoặc GPU đều chạy được  
  (GPU phụ thuộc vào môi trường torch/cuda, không cấu hình trong repo này)


## 2. Cấu trúc thư mục

```

test-docker/
├── Dockerfile              # Container build instructions
├── main.py                 # FastAPI application
├── README.md               # Usage and API docs
├── requirements.txt        # Python dependencies
└── yolov8n-pose.pt         # YOLOv8 pose model weights

```

Lưu ý: file `yolov8n-pose.pt` phải nằm cùng thư mục với `main.py`.


## 3. Cài đặt (chạy local)

### 3.1. Tạo virtualenv (khuyến nghị)

```

python -m venv venv
source venv/bin/activate

```

### 3.2. Cài dependencies

```

pip install -r requirements.txt

```


## 4. Chạy service (local)

```

uvicorn main:app --host 0.0.0.0 --port 8000

```

Sau khi chạy:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

Khi startup, service sẽ:
- Load model YOLOv8 Pose
- Chạy warmup 1 ảnh dummy để tránh request đầu bị chậm


## 5. Chạy bằng Docker (khuyến nghị)

### 5.1. Build Docker image

Chạy trong thư mục `test-docker/` (nơi có `Dockerfile`):

```

docker build -t yolo-pose-api .

```

### 5.2. Run container

```

docker run --rm -p 8000:8000 yolo-pose-api

```

Sau khi container chạy:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

### 5.3. Ghi chú về Docker

- Base image: `python:3.11-slim`
- OpenCV chạy ở chế độ headless
- Model `yolov8n-pose.pt` được copy vào image khi build
- Service chạy bằng `uvicorn main:app`
- Dockerfile **chưa cấu hình GPU**
  - Nếu cần GPU, nên dùng base image `nvidia/cuda` và cài torch phù hợp


## 6. API Endpoint

### POST /v1/pose

Infer pose từ một ảnh.

Content-Type:
```

multipart/form-data

```


### 6.1. Request fields

| Field      | Type   | Required | Description                          |
|------------|--------|----------|--------------------------------------|
| image      | file   | Yes      | Ảnh đầu vào (jpg, png, …)            |
| cam_id     | string | Yes      | ID camera / nguồn ảnh                |
| ts         | string | Yes      | Timestamp (nhiều format được hỗ trợ) |
| image_id   | string | No       | Nếu không gửi, server tự sinh        |
| conf       | float  | No       | Confidence threshold (default: 0.25) |
| iou        | float  | No       | IoU threshold cho NMS (default: 0.7) |
| max_det    | int    | No       | Số detection tối đa (default: 300)   |


### 6.2. Timestamp ts hỗ trợ

- Epoch seconds:
```

1700000000

```

- Epoch milliseconds:
```

1700000000000

```

- Float seconds:
```

1700000000.123

```

- ISO8601 có timezone:
```

2026-01-14T10:30:00Z
2026-01-14T10:30:00+07:00

```

- ISO8601 không timezone:
```

2026-01-14T10:30:00

````

(sẽ được hiểu là UTC)

Response sẽ **echo lại đúng chuỗi ts input**, không trả timestamp đã normalize.


## 7. Response format

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
      "bbox_xyxy": [120.5, 45.2, 360.8, 720.1],
      "keypoints": [
        [150.1, 80.3, 0.98],
        [155.6, 120.9, 0.95]
      ],
      "bbox_meta": {
        "has_helmet": null,
        "has_vest": null,
        "falling": null
      }
    }
  ]
}
````

Ghi chú:

* bbox_xyxy: [x_min, y_min, x_max, y_max]
* keypoints: danh sách keypoints [x, y, score]
* bbox_meta: placeholder cho nhãn bổ sung
* objects có thể rỗng nếu không detect được người

## 8. Ví dụ gọi API

```
curl -X POST "http://localhost:8000/v1/pose" \
  -F "image=@/path/to/image.jpg" \
  -F "cam_id=cam01" \
  -F "ts=2026-01-14T10:30:00Z"
```

## 9. Ghi chú thiết kế

* Service này chỉ làm inference
* Không xử lý batching, queue, scale
* Có thể đặt sau Kafka / message queue / load balancer
* Model được load một lần khi startup
