import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv5x model and DeepSORT
model = YOLO("best.pt")
tracker = DeepSort(max_age=30)
PLAYER_CLS_IDX = 2  # class index for "player"

# Open video file
cap = cv2.VideoCapture("tacticam.mp4")
out = cv2.VideoWriter("output_tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []

    # Process YOLO detections
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        # Only keep "player" class and confident detections
        if cls != PLAYER_CLS_IDX or conf < 0.5:
            continue

        # Get original YOLO box
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        w, h = x2 - x1, y2 - y1

        # 🔧 Tighter bounding box (aggressive crop)
        x1 += int(0.12 * w)
        x2 -= int(0.12 * w)
        y1 += int(0.20 * h)
        y2 -= int(0.05 * h)  # optional: crop bottom a little

        detections.append(([x1, y1, x2, y2], conf, cls))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.det_class != PLAYER_CLS_IDX:
            continue

        # ✅ Only use box from real detection (not predicted guess)
        box = track.to_ltrb(orig=True, orig_strict=True)
        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)
        tid = track.track_id

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {tid}", (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracked Players", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
