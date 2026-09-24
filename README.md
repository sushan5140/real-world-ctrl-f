# REAL-WORLD CTRL+F — Version 0.1

**Concept:** Search for a physical object by name and highlight the last place a webcam saw it. All processing stays on your machine.

**Important:** This first build uses *printed ArUco tags attached to objects*. It does NOT yet identify arbitrary untagged USB drives, pens, etc. That distinction is intentional: the first milestone is to validate the detect → remember → find interaction before tackling object recognition and robotic pointing.

## Quick start (Windows PowerShell)

Open a terminal in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe make_markers.py
.\.venv\Scripts\python.exe app.py
```

The marker PNGs are generated locally in `markers/` by `make_markers.py`, so they are not committed to the repository. If PowerShell allows virtual environment activation, you can activate it and use `python` instead. On macOS/Linux, activate with `source .venv/bin/activate`. Try `python app.py --camera 1` if your default camera isn't the intended one.

## Physical setup

1. Generate and print the three PNG cards from `markers/` at a size with markers roughly 4–6 cm wide. Keep the white margin around the black tag.
2. Affix a card to each named object (or use a larger proxy tag alongside it for the first test; a separate proxy only tracks the card, not the object).
3. Place a webcam overhead, preferably with the whole desk visible, and keep the camera fixed.
4. Show each tag clearly to the camera. The detected tag gets a green rectangle.
5. Press **F** in the camera window, type `where is my usb drive` (or `keys`, `earphones`) and press **Enter**. Alternatively, use the **1/2/3** quick-search keys.
6. Cover the tag with a notebook *without moving the tagged object*. The overlay changes to **LAST SEEN HERE**, showing the last observed position and timestamp. The app saves an image of the last unobscured frame to `data/last_seen_0.jpg` (and similarly for other IDs).
7. Press **C** to clear or **Q** to quit. Camera footage is not streamed or continuously recorded.

**For the clean demo:** print the tag, show it on top of the object, and cover it. If you slide a covered object after it disappears, the app cannot know it moved; it correctly reports *last seen*, not *current position*. Test in good lighting with minimal glare.

## Configuration

Edit `objects.json` to rename tags/add aliases or add IDs 3–49. Run `python make_markers.py` again after editing. The UI has quick keys for the first nine, space permitting, but typed search works for all named objects.

## Local storage / privacy

The local `data/locations.sqlite3` file stores tag ID, last-observed timestamp, and normalized x/y camera position. `data/last_seen_<id>.jpg` stores the latest unobscured frame when a tag disappears. Delete the `data/` folder to wipe history. Don't point your camera at sensitive documents or people; snapshots capture the **whole frame**, not just the object.

## Project stages

| Stage | Capability | Status |
|---|---|---|
| V0.1 | Webcam sees printed tags; local last-seen memory; type-to-find overlay | Implemented; tested with a synthetic camera frame, **not verified with a physical webcam here** |
| V0.2 | Tagged-object last-seen snapshots in a polished interface; occlusion handling, calibrated stationary-camera test | Next |
| V0.3 | Recognize a small, user-enrolled set of **untagged** objects; quantify false positives and occlusion failures | Research / build |
| V0.4 | ESP32 pan/tilt LED points to the *last observed* x/y with a calibrated camera-to-desk mapping | Hardware extension |
| V1 | Voice query, robust object identity, motorized spotlight, full filming demo | Stretch goal |

### Honest constraints

- Tracking printed markers doesn't equal recognizing an arbitrary pen or USB drive.
- A single overhead camera cannot locate an object hidden before being seen, distinguish lookalike objects without identity cues, or see into drawers.
- Last-seen data may be stale; don't claim the object is **definitely** still there.
- Motorized spotlight requires calibration, servo limits, a safe low-power LED, and stable power. No high-power laser.

## Run tests without a camera

```powershell
python -m unittest discover -s tests -v
```
