# Detector Core

This folder contains the Raspberry Pi 5 + Hailo-8L detector-side code and deployment files.

## Contents

- `hailo_detector.cpp`
  Main inference loop, MJPEG preview, deterrent logic, and detector wiring.
- `gpio_trigger.hpp`
  PIR, LEDs, ultrasonic output, and GPIO-related helpers.
- `detection_event.hpp`
  Event schema and JSON serialization helpers.
- `sighting_tracker.*`
  Sighting lifecycle tracking for start, update, and end events.
- `event_sink.*`
  JSONL and Unix domain socket event output.
- `hailo-detector.service`
  Example systemd unit for Pi deployment.
- `tools/simulate_sprint0.cpp`
  Small local simulation helper for event-pipeline testing.

## Notes

- This directory is intentionally isolated so the GitHub root stays easy to understand.
- The detector sources are still plain C++ files with simple relative includes to keep Pi-side iteration easy.
