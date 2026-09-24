# REAL-WORLD CTRL+F

**What if you could search your desk the way you search your laptop?**

A local, experimental physical-object finder: a stationary camera detects registered objects, remembers where each one was **last seen**, lets you search by name, and can optionally command a small pan/tilt LED to point at that location. A responsive browser studio shows the camera feed, live detections, local object library, last-visible image crops, and search results.

**Important: last observed location does not guarantee an object is still there.** A single camera cannot see inside drawers or through books. The software tracks printed ArUco markers under suitable conditions, and *optionally* recognizes textured objects enrolled by selecting their image region. The latter uses ORB feature matching, not a trained universal item detector: plain USB sticks and visually identical earphones can fail. No AI API key, account, or cloud connection is required for core visual search.

## Start on Windows (PowerShell)

Download or clone the project, then **enter the folder containing `server.py` and `requirements.txt`**. If you already have the repo checked out:

```powershell
cd "$env:USERPROFILE\Downloads\real-world-ctrl-f"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe make_markers.py
.\.venv\Scripts\python.exe server.py
```

Open **http://127.0.0.1:8765** in your browser. If your camera doesn't open, quit with `Ctrl+C` and retry with `--camera 1`. On some systems, Windows Settings → Privacy & security → Camera → allow desktop apps to access camera needs to be enabled. Shut down other apps using the same camera.

Use `python server.py --port 8766` if port 8765 is occupied. For a camera-window-only version, run `python app.py` instead of `server.py` (original V0.1 interface).

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python make_markers.py && python server.py`.

## Demo A: tags (most dependable starting point)

1. Print `markers/0_usb_drive.png`, `markers/1_keys.png`, and `markers/2_earphones.png` at approximately 4–6 cm wide. `make_markers.py` generates these locally. Keep the white margin visible.
2. Attach a corresponding card **to** the object or place it next to the object as a *proxy*. A separate card will only track the card, not the object.
3. Put the webcam on a stand looking down at a well-lit desk and **don't move the camera** after starting. A laptop camera looking straight ahead will work for experimentation but has a limited desk view.
4. Show a tag. Its box becomes visible and the item appears as “visible now.”
5. Search `where is my USB drive`, or click the item in the sidebar.
6. Cover the object and its tag without moving it. The camera can no longer see it; the UI displays a reticle at the **last observed** position and a crop of the last unobscured view.
7. Select “Forget object” to clear history; builtin tag cards remain available for later reuse.

## Demo B: enroll your own object (experimental untagged tracking)

1. Click **Enroll a new object**. Drag a rectangle directly on the live camera picture around an item, then enter its name and optional aliases.
2. Use a well-lit, distinctively textured, fairly flat item (printed notebook cover, decorated box, graphic coaster). Feature matching is usually less reliable on plain, reflective, rotating, or 3D objects.
3. Keep the same face of the object visible. Move it around and observe whether the detection box follows it. Cover it, then search for its last seen position.
4. If enrollment says “not enough distinctive details,” improve lighting, move the camera closer, or use the printed-marker workflow instead. **Enrollment success does not guarantee robust recognition**; evaluate with both positive and negative test objects.
5. Custom object crops are stored in `data/templates/`. Object names and aliases in `data/catalog.json`, history in `data/locations.sqlite3`, and last visible crops in `data/snapshots/`.

**Privacy:** The core app serves the webcam stream **only on your computer** at `127.0.0.1`; it does not upload frames. It automatically stores a cropped last-seen image locally when an observed object disappears. Don't aim the camera at sensitive information. To remove all local memory, stop the app and delete `data/`. A voice-search button is optional and depends on the browser's speech service, which **may send microphone audio to a vendor**; core text search needs no microphone.

## Optional pan/tilt spotlight hardware

This source tree includes the full optional software serial interface and ESP32 firmware, **not** a claim of tested physical robotics. Parts: an ESP32, two hobby servos, a pan/tilt bracket, a low-power LED and appropriate transistor/driver, and a suitable external 5V servo supply. Connect all grounds together. Don't run servos from the microcontroller's 3.3V pin; don't connect mains-powered lamps or lasers.

| ESP32 pin | Function |
|---|---|
| GPIO 18 | Pan servo signal |
| GPIO 19 | Tilt servo signal |
| GPIO 23 | LED driver input (through suitable driver) |
| GND | Common ground with external supply |

1. In Arduino IDE, install the **ESP32Servo** library and flash `firmware/spotlight/spotlight.ino` to your ESP32.
2. Find the board's serial port in Windows Device Manager. Stop Arduino Serial Monitor before using the Python application.
3. **Calibrate before operating:** edit `spotlight.json` with the actual servo angle pairs that aim at the four corners of your camera's fixed desk view: `tl`, `tr`, `bl`, `br`. The included values are placeholders; measure them on your own rig. The firmware further clamps movement to 30–150 degrees by default; adjust only after verifying safe limits.
4. Run `.\.venv\Scripts\python.exe server.py --serial-port COM3` (change port). In the studio, search an object and click **Point spotlight**. The command sends the remembered normalized x/y through a bilinear corner mapping to servo angles and turns on the LED. **Turn light off** sends a separate command.
5. The illumination is a demonstration of last-seen location, not an autonomous object-chasing robot. Without physical calibration, the LED may point somewhere else. Keep your eyes out of the beam and avoid leaving the rig running unattended.

## Technical structure

| File | Responsibility |
|---|---|
| `server.py` | Local web API, camera loop, persistent observations, enrollment, search and image crops |
| `vision.py` | Private visual templates and geometric ORB/Homography matching |
| `tracker.py` | ArUco detection, SQLite memory and local name matching |
| `spotlight.py` / `spotlight.json` | Optional serial pan/tilt + user-calibrated camera mapping |
| `firmware/spotlight/spotlight.ino` | ESP32 servo/LED firmware |
| `web/` | Responsive browser studio, object library and optional browser voice input |
| `app.py` | Original minimal OpenCV V0.1 desktop view |

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The automated tests cover synthetic printed-tag detection, SQLite persistence, browser routes, tag occlusion and snapshot retrieval, visual enrollment on a textured synthetic object, forgetting, and servo mapping. GitHub Actions runs camera-free tests across Python 3.11–3.13. Tests do not establish performance with your specific webcam, lighting, object types or real electronics; those require physical testing.

## Demo truthfulness

You can say: “A stationary webcam remembers where a registered object was last seen; the optional spotlight points to the remembered position.” Don't say: “It always knows where every lost item is.” If someone moves a covered object or the camera moves, the saved coordinates can become misleading. Future engineering work includes multi-camera mapping, measured recognition/false-positive rates on a real desk, and automatic calibration assistance.
