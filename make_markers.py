"""Create printable ArUco tag cards (DICT_4X4_50).

Creates, in ./markers/:
  * one PNG per built-in object tag (0_usb_drive.png, ...)
  * spare tags (by default #3-#8) you can register from the browser later
  * sheet_a4.png - all of them on one A4 page at 300 DPI, exact size

Run:  python make_markers.py [--size-mm 45] [--spare 6]
Print the A4 sheet at 100% / "actual size" (not "fit to page").
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

from catalog import load_builtin

ROOT = Path(__file__).resolve().parent
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DPI = 300


def write_png(path: Path, image: np.ndarray, dpi: int = DPI) -> None:
    """Write a PNG that carries its print resolution (pHYs chunk), so
    "actual size" printing produces the intended physical size."""
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Could not encode {path.name}')
    data = data.tobytes()
    ppm = int(round(dpi / .0254))
    body = b'pHYs' + struct.pack('>IIB', ppm, ppm, 1)
    chunk = struct.pack('>I', 9) + body + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF)
    end_of_ihdr = 8 + 4 + 4 + 13 + 4          # signature + IHDR length/type/data/crc
    path.write_bytes(data[:end_of_ihdr] + chunk + data[end_of_ihdr:])


def card(marker_id: int, label: str, marker_px: int) -> np.ndarray:
    quiet = marker_px // 8                 # white quiet zone is vital for detection
    marker = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, marker_px)
    height = marker_px + 2 * quiet + max(40, marker_px // 7)
    page = np.full((height, marker_px + 2 * quiet, 3), 255, dtype=np.uint8)
    page[quiet:quiet + marker_px, quiet:quiet + marker_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    scale = max(.5, marker_px / 520)
    cv2.putText(page, label, (quiet, quiet + marker_px + int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                .76 * scale, (40, 45, 48), max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.rectangle(page, (0, 0), (page.shape[1] - 1, page.shape[0] - 1), (215, 215, 215), 1)
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--size-mm', type=float, default=45, help='printed width of the black square')
    parser.add_argument('--spare', type=int, default=6, help='extra unassigned tags to create')
    args = parser.parse_args()
    out = ROOT / 'markers'
    out.mkdir(exist_ok=True)
    objects = load_builtin(ROOT / 'objects.json')
    entries = [(i, f'{o.name.upper()}  /  #{i}', f"{i}_{o.name.lower().replace(' ', '_')}.png")
               for i, o in objects.items()]
    spare_ids = [i for i in range(50) if i not in objects][:max(0, args.spare)]
    entries += [(i, f'SPARE TAG  /  #{i}', f'spare_{i}.png') for i in spare_ids]
    for marker_id, label, filename in entries:
        write_png(out / filename, card(marker_id, label, 400))
        print(f'Created markers/{filename}')
    # A4 sheet at true size.
    marker_px = int(round(args.size_mm / 25.4 * DPI))
    sheet_w, sheet_h = int(210 / 25.4 * DPI), int(297 / 25.4 * DPI)
    sheet = np.full((sheet_h, sheet_w, 3), 255, np.uint8)
    margin = int(10 / 25.4 * DPI)
    x, y, row_h = margin, margin, 0
    placed = 0
    for marker_id, label, _ in entries:
        tile = card(marker_id, label, marker_px)
        th, tw = tile.shape[:2]
        if x + tw > sheet_w - margin:
            x, y, row_h = margin, y + row_h + margin // 3, 0
        if y + th > sheet_h - margin:
            print('Sheet full; remaining tags are only in the individual PNG files.')
            break
        sheet[y:y + th, x:x + tw] = tile
        x += tw + margin // 3
        row_h = max(row_h, th)
        placed += 1
    cv2.putText(sheet, f'Real-World Ctrl+F tags - print at 100% (black square = {args.size_mm:g} mm)',
                (margin, sheet_h - margin // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 90), 2, cv2.LINE_AA)
    write_png(out / 'sheet_a4.png', sheet)
    print(f'Created markers/sheet_a4.png with {placed} tags at {args.size_mm:g} mm')


if __name__ == '__main__':
    main()
