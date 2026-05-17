"""
Real-time video demo — GPU version with full AdaptiveDoG pipeline.

Usage:
    uv run python demo_video_gpu.py                      # webcam
    uv run python demo_video_gpu.py --dog_strength 0.5   # stronger DoG
    uv run python demo_video_gpu.py --log_scale 0        # no log, DoG only
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from retina_preprocessor_gpu import RetinaPreprocessorGPU


def draw_detections(frame, results, model):
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

    retina = RetinaPreprocessorGPU(
        log_scale=args.log_scale,
        dog_strength=args.dog_strength,
        noise_threshold=args.noise_threshold,
    )

    cap = cv2.VideoCapture(0 if args.input == "0" else args.input)
    if not cap.isOpened():
        print("Error: cannot open video source:", args.input)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}")
    print("Press 'q' to quit, 's' to screenshot")
    print("Press '1'-'5' to adjust DoG strength (0.1 - 0.5)")
    print("Press 'l' to toggle log compression")

    fps_hist = {"orig": [], "retina": []}
    log_on = args.log_scale > 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.input != "0":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        fh, fw = frame.shape[:2]
        if max(fh, fw) > 720:
            scale = 720 / max(fh, fw)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            fh, fw = frame.shape[:2]

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
        fps_hist["orig"].append(fps_o)
        fps_hist["retina"].append(fps_r)

        vis_orig, n_orig = draw_detections(frame, r_orig, model)
        vis_retina, n_retina = draw_detections(retina_frame, r_retina, model)

        # Overlays
        cv2.putText(vis_orig, "ORIGINAL", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_orig, f"FPS: {fps_o:.1f} | Det: {n_orig}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(vis_retina, "RETINA (GPU)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_retina, f"FPS: {fps_r:.1f} | Det: {n_retina}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        info = f"DoG: {retina.dog_strength:.1f} | Log: {'ON' if log_on else 'OFF'} | Noise: {stats['noise_var']:.4f}"
        cv2.putText(vis_retina, info, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        combined = np.hstack([vis_orig, vis_retina])
        cv2.imshow("Retina Vision Demo (GPU)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved: {fname}")
        elif key in [ord(str(i)) for i in range(1, 6)]:
            retina.dog_strength = (key - ord("0")) * 0.1
            print(f"DoG strength: {retina.dog_strength:.1f}")
        elif key == ord("l"):
            log_on = not log_on
            retina.log_scale = args.log_scale if log_on else 0
            retina.log_norm = float(np.log1p(retina.log_scale)) if retina.log_scale > 0 else 1.0
            print(f"Log compression: {'ON' if log_on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    if fps_hist["orig"]:
        print(f"\nAvg FPS — Original: {np.mean(fps_hist['orig']):.1f}, "
              f"Retina: {np.mean(fps_hist['retina']):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="0")
    parser.add_argument("--log_scale", type=float, default=10.0)
    parser.add_argument("--dog_strength", type=float, default=0.3)
    parser.add_argument("--noise_threshold", type=float, default=0.005)
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
