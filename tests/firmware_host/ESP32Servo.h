// Host-side stand-in for the ESP32Servo library (tests only).
#pragma once
class Servo {
 public:
  int angle = -1;
  void setPeriodHertz(int) {}
  void attach(int, int, int) {}
  void write(int value) { angle = value; }
};
