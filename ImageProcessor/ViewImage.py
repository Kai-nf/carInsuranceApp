import pandas as pd
import cv2
from pathlib import Path

# Load dataset
CSV_FILE = Path(__file__).resolve().parent / "vehicleImage.csv"
df = pd.read_csv(CSV_FILE)

# Columns to show
COLUMNS = [
    "User rear", "Cropped rear",
    "User left-front-side", "Cropped left-front-side",
    "User right-front-side", "Cropped right-front-side"
]

for idx, row in df.iterrows():
    print(f"\nRow {idx+1}/{len(df)}")

    for col in COLUMNS:
        img_path = row.get(col, "")
        if isinstance(img_path, str) and img_path.strip() != "" and Path(img_path).exists():
            img = cv2.imread(str(img_path))
            cv2.imshow(col, img)
        else:
            print(f"⚠️ Missing: {col}")

    # Wait for key press to continue
    print("Press any key to continue, or 'q' to quit...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == ord("q"):
        break