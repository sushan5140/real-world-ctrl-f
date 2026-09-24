// Drives the real firmware source with commands from stdin.
//   "#wait <ms>"  advance the clock (loop() runs every simulated millisecond)
//   anything else is sent to the firmware as one serial line
// At the end prints the firmware's serial output and "#state pan tilt led".
#include <iostream>
#include "Arduino.h"
#include "ESP32Servo.h"

unsigned long host_millis = 0;
int host_pins[64] = {0};
HostSerial Serial;

#include "../../firmware/spotlight/spotlight.ino"

static void run_for(unsigned long ms) {
  for (unsigned long i = 0; i < ms; ++i) { loop(); host_millis += 1; }
}

int main() {
  setup();
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.rfind("#wait ", 0) == 0) { run_for(std::stoul(line.substr(6))); continue; }
    for (char c : line) Serial.input.push_back(c);
    Serial.input.push_back('\n');
    run_for(5);
  }
  run_for(2000);
  std::cout << Serial.output << "#state " << panServo.angle << " " << tiltServo.angle << " "
            << host_pins[LED_PIN] << std::endl;
  return 0;
}
