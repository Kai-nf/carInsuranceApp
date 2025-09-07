# imageProcessor.py
from ultralytics import YOLO
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

# Path to CSV (always relative to this script)
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "vehicleImage.csv"

# Where to save crops
CROPS_DIR = BASE_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Column names
USER_COLS = ["User rear", "User left-front-side", "User right-front-side"]
CROPPED_COLS = ["Cropped rear", "Cropped left-front-side", "Cropped right-front-side"]

# YOLO model (pretrained COCO; can swap with fine-tuned weights)
model = YOLO("yolov8n.pt")

# Classes considered vehicles
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van"}

# Load CSV
df = pd.read_csv(CSV_FILE)

# Make sure cropped columns exist
for c in CROPPED_COLS:
    if c not in df.columns:
        df[c] = ""

for idx, row in tqdm(df.iterrows(), total=len(df)):
    for ucol, ccol in zip(USER_COLS, CROPPED_COLS):
        img_path = row.get(ucol, "")
        if not isinstance(img_path, str) or img_path.strip() == "":
            continue

        img_file = BASE_DIR / img_path
        if not img_file.exists():
            print(f"⚠️ File not found: {img_file}")
            continue

        # Run YOLO inference
        results = model(str(img_file))
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        # Pick the biggest vehicle bbox
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        names = model.names

        best_idx, best_area = None, 0
        for i, (box, cid) in enumerate(zip(xyxy, cls_ids)):
            name = names[cid]
            if name in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > best_area:
                    best_idx, best_area = i, area

        if best_idx is None:
            continue

        x1, y1, x2, y2 = map(int, xyxy[best_idx])
        img = cv2.imread(str(img_file))
        crop = img[y1:y2, x1:x2]

        crop_name = CROPS_DIR / f"{Path(img_file).stem}_{ucol.replace(' ', '_')}_crop.jpg"
        cv2.imwrite(str(crop_name), crop)

        df.at[idx, ccol] = str(crop_name.relative_to(BASE_DIR))

# Save back into the SAME CSV
df.to_csv(CSV_FILE, index=False)
print(f"✅ Updated CSV saved: {CSV_FILE}")
