import hashlib
import time as ttime
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 Pose Inference API", version="1.0.0")

# Load model once
model = YOLO("yolov8n-pose.pt")

# Default timezone for naive datetime strings (no tz info).
DEFAULT_NAIVE_TZ = timezone.utc


def read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def _normalize_epoch_to_ms(epoch: int) -> int:
    a = abs(epoch)
    if a < 1_000_000_000_000:          # seconds
        return int(epoch * 1000)
    if a < 1_000_000_000_000_000:      # milliseconds
        return int(epoch)
    if a < 1_000_000_000_000_000_000:  # microseconds
        return int(epoch // 1000)
    return int(epoch // 1_000_000)     # nanoseconds


def normalize_ts_to_ms(ts_value: Union[str, int, float, None]) -> Optional[int]:
    if ts_value is None:
        return None

    if isinstance(ts_value, (int, float)):
        if isinstance(ts_value, float) and not ts_value.is_integer():
            if abs(ts_value) < 1e12:
                return int(round(ts_value * 1000.0))
            return int(round(ts_value))
        return _normalize_epoch_to_ms(int(ts_value))

    if isinstance(ts_value, str):
        s = ts_value.strip()

        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return _normalize_epoch_to_ms(int(s))

        try:
            f = float(s)
            if abs(f) < 1e12:
                return int(round(f * 1000.0))
            return int(round(f))
        except Exception:
            pass

        iso = s
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(iso)
        except Exception as e:
            raise ValueError(f"Unsupported ts format: {ts_value}") from e

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=DEFAULT_NAIVE_TZ)

        return int(dt.timestamp() * 1000)

    raise ValueError("Unsupported ts type")


def make_ephemeral_image_id(cam_id: str, ts_ms: Optional[int], img_bytes: bytes) -> str:
    h = hashlib.sha1(img_bytes).hexdigest()[:10]
    tpart = str(ts_ms) if ts_ms is not None else "notime"
    return f"{cam_id}-{tpart}-{h}"


def _extract_objects_from_result(r) -> List[Dict[str, Any]]:
    """
    Helper dùng chung cho warmup self-test (không echo meta).
    """
    boxes = r.boxes
    kps = r.keypoints
    names = r.names if isinstance(r.names, dict) else {}

    objects: List[Dict[str, Any]] = []
    n = 0 if boxes is None else len(boxes)

    for i in range(n):
        b = boxes[i]
        cls_id = int(b.cls.item()) if b.cls is not None else 0
        conf_i = float(b.conf.item()) if b.conf is not None else None
        xyxy = [float(x) for x in b.xyxy[0].tolist()]

        kp_list = []
        if kps is not None and kps.xy is not None:
            xy = kps.xy[i].cpu().numpy()
            if getattr(kps, "conf", None) is not None and kps.conf is not None:
                cf = kps.conf[i].cpu().numpy()
                for (x, y), c in zip(xy, cf):
                    kp_list.append([float(x), float(y), float(c)])
            else:
                for (x, y) in xy:
                    kp_list.append([float(x), float(y), None])

        objects.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "conf": conf_i,
                "bbox_xyxy": xyxy,
                "keypoints": kp_list,
                "bbox_meta": {
                    "has_helmet": None,
                    "has_vest": None,
                    "falling": None,
                },
            }
        )
    return objects


@app.on_event("startup")
def warmup_and_selftest() -> None:
    """
    Warmup: chạy 1 ảnh dummy để tránh request đầu bị chậm.
    Self-test: parse output tối thiểu để chắc pipeline boxes/keypoints hoạt động.
    """
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        t0 = ttime.perf_counter()
        results = model.predict(source=dummy, imgsz=640, conf=0.25, iou=0.7, max_det=300, verbose=False)
        warm_ms = (ttime.perf_counter() - t0) * 1000.0

        r = results[0]
        objs = _extract_objects_from_result(r)

        # Dummy ảnh thường không có người => objs có thể rỗng, đó là bình thường.
        # Mình chỉ check "không crash" và schema keys có mặt.
        sample = objs[0] if len(objs) > 0 else None
        if sample is not None:
            needed_keys = {"bbox_xyxy", "keypoints", "bbox_meta", "class_id", "conf"}
            missing = needed_keys - set(sample.keys())
            if missing:
                raise RuntimeError(f"Warmup self-test missing keys: {missing}")

        print(f"[startup] Warmup OK in {warm_ms:.2f} ms | objects={len(objs)}")
    except Exception as e:
        # Fail fast: nếu warmup lỗi, server nên biết ngay.
        print(f"[startup] Warmup FAILED: {e}")
        raise


@app.post("/v1/pose")
async def pose_infer(
    image: UploadFile = File(...),
    cam_id: str = Form(...),
    ts: str = Form(...),  # input timestamp in any supported format
    image_id: Optional[str] = Form(None),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    max_det: int = Form(300),
):
    try:
        img_bytes = await image.read()
        img = read_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Normalize for internal use (tracking later). We DO NOT return ts_ms; we echo ts.
    try:
        ts_ms = normalize_ts_to_ms(ts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ts: {e}")

    final_image_id = image_id or make_ephemeral_image_id(cam_id, ts_ms, img_bytes)

    t0 = ttime.perf_counter()
    results = model.predict(source=img, conf=conf, iou=iou, max_det=max_det, verbose=False)
    infer_ms = (ttime.perf_counter() - t0) * 1000.0

    objects = _extract_objects_from_result(results[0])

    # IMPORTANT: return ts exactly as input (echo)
    return {
        "cam_id": cam_id,
        "ts": ts,
        "image_id": final_image_id,
        "inference_ms": infer_ms,
        "objects": objects,
    }
