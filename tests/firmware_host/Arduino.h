// Minimal host-side stand-in for the Arduino core, used only by the automated
// tests to compile and exercise firmware/spotlight/spotlight.ino on a PC.
// It is NOT a hardware simulator: timing, PWM and servo physics are not modelled.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>

#define HIGH 1
#define LOW 0
#define OUTPUT 1

extern unsigned long host_millis;
extern int host_pins[64];
inline unsigned long millis() { return host_millis; }
inline void pinMode(int, int) {}
inline void digitalWrite(int pin, int value) { host_pins[pin] = value; }
template <class T> T constrain(T x, T low, T high) { return x < low ? low : (x > high ? high : x); }
using std::max;
using std::min;

struct HostSerial {
  std::deque<char> input;
  std::string output;
  void begin(long) {}
  int available() { return (int)input.size(); }
  int read() { char c = input.front(); input.pop_front(); return (unsigned char)c; }
  void print(const char *s) { output += s; }
  void print(int v) { output += std::to_string(v); }
  void println(const char *s) { output += s; output += "\n"; }
  void println(int v) { output += std::to_string(v); output += "\n"; }
};
extern HostSerial Serial;
