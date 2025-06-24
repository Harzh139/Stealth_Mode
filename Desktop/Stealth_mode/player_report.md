# Player Re-Identification and Tracking

## 1. Approach and Methodology

### ✅ Detection:

* Used the provided `best.pt` YOLOv5 model (fine-tuned) via the `ultralytics` library.
* Focused only on the `"player"` class (class index 2) to avoid confusion with referees and goalkeepers.

### ✅ Tracking:

* Integrated `DeepSORT` and `StrongSORT` trackers for robust ID assignment.
* Applied appearance-based re-identification (ReID) using pre-trained OSNet (where applicable).
* Used YOLO-detected boxes for visual consistency instead of predicted tracker boxes.

### ✅ Post-Processing:

* Shrunk YOLO bounding boxes heuristically (e.g., -12% width, -20% height) to reduce excess padding.
* Filtered out low-confidence and overly small boxes.
* Rendered player labels only once per frame to avoid duplicates.

---

## 2. Techniques Attempted and Outcomes

| Technique                      | Purpose                                   | Outcome                                      |
| ------------------------------ | ----------------------------------------- | -------------------------------------------- |
| Shrink YOLO Boxes              | Tighter, cleaner visuals                  | Improved clarity, reduced overlap            |
| Box Size / Aspect Ratio Filter | Remove tiny/false positives               | Reduced clutter but needed careful tuning    |
| orig\_strict=True              | Skip ghost boxes from tracker predictions | Prevented floating/ghost boxes               |
| DeepSORT + StrongSORT          | Maintain consistent IDs                   | Worked well after ReID and parameter tuning  |
| Fixed ReID Format              | Convert YOLO boxes to (x1, y1, x2, y2)    | Prevented API mismatches                     |
| Jersey Color Heuristic         | Team identification                       | Discarded due to low reliability             |
| Circle Overlay Visualization   | Alternative to boxes                      | Abandoned for better clarity with rectangles |
| Seen-ID Label Filter           | Prevent duplicate ID tags per frame       | Resolved label clutter                       |

---

## 3. Challenges Encountered

### ⚠️ Model Limitations:

* YOLOv5 outputs had **loose boxes** (excess padding), leading to overlap.
* No support for **segmentation or pixel-level masks**, limiting visual precision.

### ❌ Bounding Box Format:

* YOLO's center format required conversion for ReID model compatibility.

### ❌ Class Confusion:

* Players vs. referees misclassified occasionally due to similar appearance. Solved by filtering strictly for `"player"` class.

### ❌ ReID Integration Issues:

* Original OSNet model download was broken (404). Switched to alternative from ByteTrack.
* API mismatches in `.get_features()` required careful reshaping.

### ❌ Kalman Filter Issues:

* Tracker state shape mismatches caused runtime errors. Solved with reshaping and fallback logic.

### ❌ Label Duplication:

* Multiple tags per player ("Player 5Player 8") were appearing. Solved using per-frame seen-ID sets.

### ❌ Visual Clutter:

* Floating boxes from Kalman predictions were misleading. Fixed by drawing only YOLO-based boxes.

---

## 4. Remaining Work and Future Steps

If more time/resources were available:

* ✅ **Use Segmentation**: Replace YOLOv5 with a `YOLOv8-seg` model to obtain precise body masks.
* ✅ **Jersey Number OCR**: Use EasyOCR or Tesseract to extract actual jersey numbers.
* ✅ **Team Classification**: Implement color clustering to differentiate teams.
* ✅ **Stronger ReID**: Train or fine-tune a domain-specific ReID model (e.g., OSNet on football dataset).
* ✅ **Longer Video Evaluation**: Test on full-match clips for real-world robustness.

---

## 📅 Submission Notes

* Python Version: 3.13.3 (latest; works, but 3.10+ is safer for production).
* Codebase is **fully self-contained**. No retraining or external dependencies.
* All core logic resides in `player_tracking.py`.
* Required packages listed in `requirements.txt`.
* Output is saved as `output_tracked.mp4`. Verified working end-to-end.

---

✅ **Status:** Complete and reproducible.
