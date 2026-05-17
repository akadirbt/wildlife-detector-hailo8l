#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include "event_sink.hpp"
#include "sighting_tracker.hpp"

static void print_events(const std::vector<DetectionEvent>& events, EventSink& sink) {
    for (auto event : events) {
        std::cout << detection_event_to_json(event) << "\n";
        sink.enqueue(std::move(event));
    }
}

int main() {
    SightingTrackerConfig tracker_cfg;
    tracker_cfg.detector_id = "garden-pi-01";
    tracker_cfg.snapshot_root = "sim_detections";
    tracker_cfg.match_iou_threshold = 0.30f;
    tracker_cfg.update_interval_ms = 500;
    tracker_cfg.lost_timeout_ms = 2000;

    EventSinkConfig sink_cfg;
    sink_cfg.jsonl_root = "sim_detections";
    sink_cfg.enable_jsonl = true;
    sink_cfg.enable_uds = false;

    SightingTracker tracker(tracker_cfg);
    EventSink sink(sink_cfg);
    sink.start();

    const int frame_w = 1536;
    const int frame_h = 864;
    const auto t0 = std::chrono::steady_clock::now();

    print_events(
        tracker.update(
            {TrackedDetection{"deer", 0.82f, DetectionBox{408, 137, 592, 360}}},
            frame_w,
            frame_h,
            t0),
        sink);

    print_events(
        tracker.update(
            {TrackedDetection{"deer", 0.86f, DetectionBox{416, 141, 604, 364}}},
            frame_w,
            frame_h,
            t0 + std::chrono::milliseconds(600)),
        sink);

    print_events(
        tracker.update(
            {},
            frame_w,
            frame_h,
            t0 + std::chrono::milliseconds(2600)),
        sink);

    sink.stop();

    const auto stats = sink.stats();
    std::cout << "events_enqueued=" << stats.events_enqueued
              << " events_written=" << stats.events_written
              << " jsonl_errors=" << stats.jsonl_write_error_count
              << "\n";

    return 0;
}
