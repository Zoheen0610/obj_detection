import cv2
import argparse
from collections import defaultdict
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.4

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def draw_counts(frame, counts):
    """Overlay per-class counts on the frame."""
    y = 30
    for label, count in sorted(counts.items()):
        text = f"{label}: {count}"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 28
    return frame


def run_on_image(model, image_path, output_path):
    """Extension 3: Run detection on a single image and save annotated result."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    counts = defaultdict(int)
    for cls_id in results.boxes.cls.tolist():
        counts[model.names[int(cls_id)]] += 1

    annotated = results.plot()
    annotated = draw_counts(annotated, counts)
    cv2.imwrite(output_path, annotated)
    print(f"[IMAGE] Saved annotated image → {output_path}")
    print(f"        Detections: {dict(counts)}")


def run_on_video(model, source, output_path=None):
    """
    Extension 1: Per-class object counting displayed live.
    Extension 2: Save annotated video to disk.
    source: 0 for webcam, or path to video file.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    # Extension 2 — set up video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"[VIDEO] Recording → {output_path}")

    print("[INFO] Press 'q' to quit.")
    total_counts = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # Extension 1 — count per class in this frame
        frame_counts = defaultdict(int)
        for cls_id in results.boxes.cls.tolist():
            label = model.names[int(cls_id)]
            frame_counts[label] += 1
            total_counts[label] += 1

        # Draw boxes + labels
        annotated = results.plot()

        # Draw live counts overlay
        annotated = draw_counts(annotated, frame_counts)

        # Extension 2 — write frame
        if writer:
            writer.write(annotated)

        cv2.imshow("YOLOv8 Object Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n[SUMMARY] Total detections across all frames:")
    for label, count in sorted(total_counts.items()):
        print(f"  {label}: {count}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detector")
    parser.add_argument("--source", default="0",
                        help="Input source: 0=webcam, path to video, or path to image")
    parser.add_argument("--save", default=None,
                        help="Path to save output (video .mp4 or image .jpg)")
    parser.add_argument("--image", action="store_true",
                        help="Treat source as a single image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded: {MODEL_PATH}")

    if args.image:
        out = args.save or "output.jpg"
        run_on_image(model, args.source, out)
    else:
        source = int(args.source) if args.source == "0" else args.source
        run_on_video(model, source, args.save)