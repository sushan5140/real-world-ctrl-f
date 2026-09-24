"""Create three printable, individually tagged object cards. Run once."""
from pathlib import Path
import cv2
import numpy as np
from tracker import load_objects

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "markers"
OUT.mkdir(exist_ok=True)
objects = load_objects(ROOT / "objects.json")
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for marker_id, info in objects.items():
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 400)
    # White quiet zone is vital for camera recognition.
    page = np.full((530, 480, 3), 255, dtype=np.uint8)
    page[22:422, 40:440] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    label = f"{info['name'].upper()}  /  #{marker_id}"
    cv2.putText(page, label, (30, 480), cv2.FONT_HERSHEY_SIMPLEX,
                0.76, (40, 45, 48), 2, cv2.LINE_AA)
    target = OUT / f"{marker_id}_{info['name'].lower().replace(' ', '_')}.png"
    cv2.imwrite(str(target), page)
    print(f"Created {target}")
