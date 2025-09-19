from roboflow import Roboflow
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ========== KONFIG ==============
API_KEY = "6rZiGw36FzTZetd8dW6l"          # ganti
WORKSPACE_SLUG = "cctv-pgxpj"        # ganti sesuai URL
PROJECT_SLUG   = "cctv-t8clc"

OUT_DIR = Path("/home/muqsith/pengmas/video_train/dataset_labeled_mct")
SPLITS = ["train","val","test"]

MAX_WORKERS = 12       # jumlah thread paralel
MANIFEST = OUT_DIR/"upload_manifest.txt"
# ================================

rf = Roboflow(api_key=API_KEY)
proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_SLUG)

# baca file yang sudah sukses
done = set()
if MANIFEST.exists():
    done = set(MANIFEST.read_text().splitlines())

lock = threading.Lock()

def upload_one(split, img_path):
    key = f"{split}/{img_path.name}"
    if key in done:
        return f"skip {key}"

    lbl_path = OUT_DIR/split/"labels"/(img_path.stem + ".txt")
    if not lbl_path.exists():
        return f"no_label {key}"

    try:
        proj.upload(
            image_path=str(img_path),
            annotation_path=str(lbl_path),
            split=split
        )
        with lock:
            with MANIFEST.open("a") as f:
                f.write(key + "\n")
        return f"ok {key}"
    except Exception as e:
        return f"fail {key} -> {e}"

def run_split(split):
    imgs = sorted((OUT_DIR/split/"images").glob("*.*"))
    print(f"[{split}] total {len(imgs)} file")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(upload_one, split, img) for img in imgs]
        for fut in as_completed(futures):
            print(fut.result())

def main():
    for s in SPLITS:
        run_split(s)
    print("=== DONE ===")
    print("Manifest:", MANIFEST)

if __name__ == "__main__":
    main()
