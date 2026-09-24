/* Real-World Ctrl+F optional ESP32 pan/tilt LED.
 * Arduino IDE: install ESP32Servo; select your ESP32 board.
 * Never power servos from the ESP32's 3.3V pin. Use a suitable external 5V
 * supply, connect supply GND to ESP32 GND, and use a transistor/driver for LED.
 * Use a low-power LED (no lasers, no unshielded mains wiring).
 * Protocol: P,pan,tilt\n (0..180), L,1\n or L,0\n. 115200 baud.
 */
#include <ESP32Servo.h>
Servo panServo, tiltServo;
constexpr int PAN_PIN = 18;
constexpr int TILT_PIN = 19;
constexpr int LED_PIN = 23;
constexpr int MIN_PAN = 30, MAX_PAN = 150;
constexpr int MIN_TILT = 30, MAX_TILT = 150;
String command;
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN, 500, 2400);
  tiltServo.attach(TILT_PIN, 500, 2400);
  panServo.write(90);
  tiltServo.write(90);
}
void loop() {
  while (Serial.available()) {
    char c = char(Serial.read());
    if (c == '\n') {
      command.trim();
      if (command.startsWith("P,")) {
        int separator = command.indexOf(',', 2);
        if (separator > 2) {
          int pan = command.substring(2, separator).toInt();
          int tilt = command.substring(separator + 1).toInt();
          panServo.write(constrain(pan, MIN_PAN, MAX_PAN));
          tiltServo.write(constrain(tilt, MIN_TILT, MAX_TILT));
        }
      } else if (command == "L,1") digitalWrite(LED_PIN, HIGH);
      else if (command == "L,0") digitalWrite(LED_PIN, LOW);
      command = "";
    } else if (command.length() < 48) command += c;
    else command = "";
  }
}
