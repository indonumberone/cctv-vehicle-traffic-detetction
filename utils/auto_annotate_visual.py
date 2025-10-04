# auto_annotate_visual.py
# --------------------------------------------------------
# 1) Ambil semua gambar dari RAW_IMG_DIR
# 2) Prediksi YOLOv8n (COCO) -> simpan label Yolo (cx cy w h) utk motorcycle/car/truck
# 3) (Opsional) Tampilkan preview via cv2.imshow saat anotasi
#    - Tombol: q = stop preview & lanjut tanpa preview, Esc = berhenti total, p = pause/resume
# --------------------------------------------------------

import os, glob, sys, shutil
from pathlib import Path
from typing import List
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ======== KONFIGURASI ========
# Folder sumber gambar (punyamu):
RAW_IMG_DIR = Path("/home/muqsith/pengmas/video_train/dataset_raw/images_yolo_sampled")

# Folder output dataset (hanya images+labels; split bisa dibuat belakangan)
OUT_DIR = Path("/home/muqsith/pengmas/video_train/dataset_labeled_mct")

# Kelas lokal (urutan tetap)
LOCAL_CLASS_ORDER = ["motorcycle", "car", "truck"]  # 0,1,2

# Threshold deteksi
CONF = 0.30
IOU  = 0.5

# Tampilkan preview anotasi?
SHOW_PREVIEW = True       # set False jika tanpa jendela
SAVE_VIZ     = False      # True jika ingin simpan gambar ber-bbox ke OUT_DIR/viz
# =============================


def collect_images(folder: Path) -> List[Path]:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    imgs = []
    for e in exts:
        imgs.extend(folder.glob(e))
    return sorted(imgs)


def prepare_out_dirs():
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
    if SAVE_VIZ:
        (OUT_DIR / "viz").mkdir(parents=True, exist_ok=True)


def try_init_preview() -> bool:
    """Coba buat window preview. Jika gagal (headless), nonaktifkan SHOW_PREVIEW."""
    if not SHOW_PREVIEW:
        return False
    try:
        cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("annotate", 1280, 720)
        return True
    except Exception as e:
        print(f"[INFO] cv2.imshow tidak tersedia (mungkin headless). Nonaktifkan preview. Detail: {e}")
        return False


def draw_boxes(frame, result, names_map):
    if result.boxes is None or len(result.boxes) == 0:
        return frame
    for b in result.boxes:
        cls_id = int(b.cls.item())
        conf   = float(b.conf.item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        label = f"{names_map[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return frame


def main():
    if not RAW_IMG_DIR.exists():
        print(f"Folder sumber tidak ditemukan: {RAW_IMG_DIR}")
        sys.exit(1)

    imgs = collect_images(RAW_IMG_DIR)
    if not imgs:
        print(f"Tidak ada gambar di {RAW_IMG_DIR}")
        sys.exit(1)

    prepare_out_dirs()

    # Salin gambar ke OUT_DIR/images agar konsisten
    print(f"Total gambar ditemukan: {len(imgs)}")
    for p in tqdm(imgs, desc="Menyalin gambar"):
        dst = OUT_DIR / "images" / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    print(">> Memuat model YOLOv8n (COCO)...")
    model = YOLO("yolov8x.pt")
    names = model.model.names  # dict {id: name}
    name_to_id = {v:k for k,v in names.items()}

    # Cek kelas tersedia
    missing = [c for c in LOCAL_CLASS_ORDER if c not in name_to_id]
    if missing:
        print(f"Kelas tidak ditemukan di model: {missing}")
        sys.exit(1)

    target_global_ids = [name_to_id[c] for c in LOCAL_CLASS_ORDER]

    # Inisialisasi preview
    preview_enabled = try_init_preview()
    paused = False
    labels_dir = OUT_DIR / "labels"

    # Proses satu per satu (perlu untuk preview realtime)
    for img_path in tqdm(imgs, desc="Auto-annotate"):
        dst_img_path = OUT_DIR / "images" / img_path.name
        frame = cv2.imread(str(dst_img_path))
        if frame is None:
            # skip file rusak
            continue

        # Prediksi (filter 3 kelas target)
        r = model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            classes=target_global_ids,
            verbose=False
        )[0]

        H, W = r.orig_shape
        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_g = int(b.cls.item())
                cls_name = names[cls_g]
                if cls_name not in LOCAL_CLASS_ORDER:
                    continue
                local_id = LOCAL_CLASS_ORDER.index(cls_name)
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w  = (x2 - x1) / W
                h  = (y2 - y1) / H
                lines.append(f"{local_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Simpan label (boleh kosong)
        with open(labels_dir / f"{img_path.stem}.txt", "w") as f:
            f.write("\n".join(lines))

        # Preview (opsional)
        if preview_enabled:
            vis = frame.copy()
            vis = draw_boxes(vis, r, names)
            info = f"{img_path.name} | deteksi: {len(r.boxes) if r.boxes is not None else 0}"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("annotate", vis)
            if SAVE_VIZ:
                cv2.imwrite(str(OUT_DIR / "viz" / img_path.name), vis)

            # kontrol keyboard
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == 27:  # ESC -> stop total
                print("Dihentikan oleh pengguna (ESC).")
                break
            elif key == ord('q'):  # q -> matikan preview, lanjut tanpa window
                print("Preview dimatikan. Lanjut anotasi tanpa window.")
                cv2.destroyWindow("annotate")
                preview_enabled = False
            elif key == ord('p'):  # p -> pause/resume
                paused = not paused

    if preview_enabled:
        cv2.destroyAllWindows()

    # Tulis data.yaml minimal (tanpa split dulu)
    yaml_text = f"""# YOLOv8 data config (labels siap)
path: {OUT_DIR.resolve()}
train: images   # (belum di-split; bisa diubah nanti)
val: images
names:
  0: motorcycle
  1: car
  2: truck
"""
    with open(OUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_text)

    print("\n[SELESAI] Anotasi otomatis selesai.")
    print("Gambar & label tersimpan di:", OUT_DIR.resolve())
    print("File contoh label:", (labels_dir / (imgs[0].stem + '.txt')).resolve())
    print("Catatan: kamu bisa lakukan split train/val/test belakangan, atau langsung latih dengan Ultralytics.")


if __name__ == "__main__":
    main()
