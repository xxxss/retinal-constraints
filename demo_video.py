"""
Real-time video demo: Retina Preprocessor + YOLOv8 object detection.

Side-by-side: Original vs Retina-processed YOLO detection.

Usage:
    uv run python demo_video.py                         # webcam
    uv run python demo_video.py --input video.mp4       # video file
    uv run python demo_video.py --dog_strength 0.3      # adjust DoG
    uv run python demo_video.py --log_scale 0            # no log, DoG only
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from retina_preprocessor import RetinaPreprocessor


def draw_detections(frame, results, model):
    """Draw YOLO detection boxes on frame. Returns (annotated_frame, count)."""
    vis = frame.copy()
    count = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()
            label = model.names[int(box.cls[0].item())]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count += 1
    return vis, count


def run_demo(args):
    print("Loading YOLOv8-nano...")
    model = YOLO("yolov8n.pt")

    retina = RetinaPreprocessor(
        log_scale=args.log_scale,
        dog_sigma_center=args.dog_sigma_center,
        dog_sigma_surround=args.dog_sigma_surround,
        dog_strength=args.dog_strength,
        delta_threshold=args.delta_threshold,
    )

    cap = cv2.VideoCapture(0 if args.input == "0" else args.input)
    if not cap.isOpened():
        print("Error: cannot open video source:", args.input)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height}")
    print(f"Retina: log={args.log_scale} DoG={args.dog_strength} delta={args.delta_threshold}")
    print("Press 'q' to quit, 's' to screenshot")

    fps_history = {"orig": [], "retina": []}

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.input != "0":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Resize if too large
        h, w = frame.shape[:2]
        if max(h, w) > 720:
            scale = 720 / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            h, w = frame.shape[:2]

        # --- Original ---
        t0 = time.time()
        r_orig = model(frame, verbose=False, imgsz=640)
        t_orig = time.time() - t0

        # --- Retina ---
        t0 = time.time()
        retina_frame, stats = retina.process_video_frame(frame)
        r_retina = model(retina_frame, verbose=False, imgsz=640)
        t_retina = time.time() - t0

        fps_o = 1.0 / max(t_orig, 1e-6)
        fps_r = 1.0 / max(t_retina, 1e-6)
        fps_history["orig"].append(fps_o)
        fps_history["retina"].append(fps_r)

        # Draw
        vis_orig, n_orig = draw_detections(frame, r_orig, model)
        vis_retina, n_retina = draw_detections(retina_frame, r_retina, model)

        # Info overlay
        cv2.putText(vis_orig, "ORIGINAL", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_orig, f"FPS: {fps_o:.1f} | Det: {n_orig}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(vis_retina, "RETINA", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_retina, f"FPS: {fps_r:.1f} | Det: {n_retina}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_retina, f"Change: {stats['changed']:.0%}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        combined = np.hstack([vis_orig, vis_retina])
        cv2.imshow("Retina Vision Demo", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()

    if fps_history["orig"]:
        print(f"\nAvg FPS — Original: {np.mean(fps_history['orig']):.1f}, "
              f"Retina: {np.mean(fps_history['retina']):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Retina Vision Demo")
    parser.add_argument("--input", default="0",
                        help="0 = webcam, or path to video file")
    parser.add_argument("--log_scale", type=float, default=10.0)
    parser.add_argument("--dog_sigma_center", type=float, default=1.5)
    parser.add_argument("--dog_sigma_surround", type=float, default=3.0)
    parser.add_argument("--dog_strength", type=float, default=0.3)
    parser.add_argument("--delta_threshold", type=int, default=15)
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
