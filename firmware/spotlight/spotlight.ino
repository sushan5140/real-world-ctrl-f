/* Real-World Ctrl+F — optional ESP32 pan/tilt LED spotlight, firmware v2.
 *
 * Arduino IDE: install the "ESP32Servo" library, select your ESP32 board.
 * WIRING SAFETY
 *  - Never power servos from the ESP32 3.3 V pin. Use a separate 5 V supply
 *    (>= 2 A for two hobby servos) and connect its GND to the ESP32 GND.
 *  - Drive the LED through a transistor/MOSFET or LED driver, not the pin.
 *  - Low-power LED only: no lasers, no mains-powered lamps.
 *
 * SERIAL PROTOCOL (115200 baud, one command per line)
 *   ?               -> PONG CTRLF-SPOTLIGHT 2
 *   P,<pan>,<tilt>  -> OK P,<pan>,<tilt>   angles 0..180, clamped to the limits
 *   L,1  /  L,0     -> OK L,1 / OK L,0
 *   H               -> OK H   (home to 90/90 and light off)
 *   anything else   -> ERR <reason>
 * On boot the board prints: READY CTRLF-SPOTLIGHT 2
 *
 * SAFETY BEHAVIOUR
 *  - Servos move smoothly (limited speed) instead of jumping.
 *  - Angles are clamped to MIN_/MAX_ limits (keep in sync with the app's
 *    spotlight.json "limits"). Adjust only after checking your bracket.
 *  - The LED switches itself off if no command arrives for LED_TIMEOUT_MS.
 */
#include <ESP32Servo.h>

constexpr int PAN_PIN = 18;
constexpr int TILT_PIN = 19;
constexpr int LED_PIN = 23;
constexpr int MIN_PAN = 30, MAX_PAN = 150;
constexpr int MIN_TILT = 30, MAX_TILT = 150;
constexpr int HOME_PAN = 90, HOME_TILT = 90;
constexpr unsigned long STEP_INTERVAL_MS = 15;   // 2 degrees every 15 ms ~ 130 deg/s
constexpr int STEP_DEGREES = 2;
constexpr unsigned long LED_TIMEOUT_MS = 120000; // auto-off after 2 minutes idle
constexpr size_t MAX_LINE = 32;

Servo panServo, tiltServo;
int panNow = HOME_PAN, tiltNow = HOME_TILT;
int panTarget = HOME_PAN, tiltTarget = HOME_TILT;
bool ledOn = false;
unsigned long lastCommandMs = 0;
unsigned long lastStepMs = 0;
char line[MAX_LINE + 1];
size_t lineLength = 0;
bool lineOverflow = false;

void setLed(bool on) {
  ledOn = on;
  digitalWrite(LED_PIN, on ? HIGH : LOW);
}

// Strict unsigned integer parse; returns false on anything unexpected.
bool parseAngle(const char *text, int &value) {
  if (*text == '\0') return false;
  long result = 0;
  for (const char *p = text; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
    result = result * 10 + (*p - '0');
    if (result > 180) return false;
  }
  value = (int)result;
  return true;
}

void handleCommand(char *cmd) {
  lastCommandMs = millis();
  if (strcmp(cmd, "?") == 0) {
    Serial.println("PONG CTRLF-SPOTLIGHT 2");
  } else if (strcmp(cmd, "H") == 0) {
    panTarget = HOME_PAN;
    tiltTarget = HOME_TILT;
    setLed(false);
    Serial.println("OK H");
  } else if (strcmp(cmd, "L,1") == 0 || strcmp(cmd, "L,0") == 0) {
    setLed(cmd[2] == '1');
    Serial.println(ledOn ? "OK L,1" : "OK L,0");
  } else if (cmd[0] == 'P' && cmd[1] == ',') {
    char *first = cmd + 2;
    char *comma = strchr(first, ',');
    int pan, tilt;
    if (!comma) { Serial.println("ERR expected P,<pan>,<tilt>"); return; }
    *comma = '\0';
    if (!parseAngle(first, pan) || !parseAngle(comma + 1, tilt)) {
      Serial.println("ERR angles must be whole numbers 0-180");
      return;
    }
    panTarget = constrain(pan, MIN_PAN, MAX_PAN);
    tiltTarget = constrain(tilt, MIN_TILT, MAX_TILT);
    Serial.print("OK P,");
    Serial.print(panTarget);
    Serial.print(",");
    Serial.println(tiltTarget);
  } else {
    Serial.println("ERR unknown command");
  }
}

int stepToward(int now, int target) {
  if (now < target) return min(now + STEP_DEGREES, target);
  if (now > target) return max(now - STEP_DEGREES, target);
  return now;
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  setLed(false);                       // light off before anything else
  Serial.begin(115200);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN, 500, 2400);
  tiltServo.attach(TILT_PIN, 500, 2400);
  panServo.write(HOME_PAN);
  tiltServo.write(HOME_TILT);
  lastCommandMs = millis();
  Serial.println("READY CTRLF-SPOTLIGHT 2");
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[lineLength] = '\0';
      if (lineOverflow) Serial.println("ERR line too long");
      else if (lineLength > 0) handleCommand(line);
      lineLength = 0;
      lineOverflow = false;
    } else if (lineLength < MAX_LINE) {
      line[lineLength++] = c;
    } else {
      lineOverflow = true;             // discard the rest of this line
    }
  }

  unsigned long now = millis();
  if (now - lastStepMs >= STEP_INTERVAL_MS) {
    lastStepMs = now;
    int nextPan = stepToward(panNow, panTarget);
    int nextTilt = stepToward(tiltNow, tiltTarget);
    if (nextPan != panNow) { panNow = nextPan; panServo.write(panNow); }
    if (nextTilt != tiltNow) { tiltNow = nextTilt; tiltServo.write(tiltNow); }
  }

  if (ledOn && now - lastCommandMs >= LED_TIMEOUT_MS) {
    setLed(false);
    Serial.println("EVT LED_TIMEOUT");
  }
}
