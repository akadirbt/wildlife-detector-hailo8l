#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "detection_event.hpp"

struct SightingTrackerConfig {
    std::string detector_id = "garden-pi-01";
    std::string source = "hailo_detector";
    std::string snapshot_root = "detections";
    float match_iou_threshold = 0.30f;
    int update_interval_ms = 500;
    int lost_timeout_ms = 2000;
};

struct TrackedDetection {
    std::string label;
    float confidence = 0.0f;
    DetectionBox box;
};

class SightingTracker {
public:
    explicit SightingTracker(SightingTrackerConfig config = {});

    std::vector<DetectionEvent> update(
        const std::vector<TrackedDetection>& detections,
        int frame_w,
        int frame_h,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    void reset();

private:
    struct ActiveSighting {
        std::string sighting_id;
        std::string label;
        std::string priority;
        DetectionBox box;
        float confidence = 0.0f;
        float best_confidence = 0.0f;
        int frame_w = 0;
        int frame_h = 0;
        std::string snapshot;
        std::chrono::steady_clock::time_point first_seen;
        std::chrono::steady_clock::time_point last_seen;
        std::chrono::steady_clock::time_point last_update_emitted;
        bool matched_this_update = false;
    };

    static float iou(const DetectionBox& a, const DetectionBox& b);
    static std::string priority_for_label(const std::string& label);
    static std::string today_folder();
    static std::string make_sighting_id(const std::string& label, uint64_t sequence);
    std::string make_snapshot_path(const std::string& sighting_id) const;

    DetectionEvent make_event(
        DetectionEventType type,
        const ActiveSighting& sighting,
        std::chrono::steady_clock::time_point now) const;

    int find_match(const TrackedDetection& detection) const;

    SightingTrackerConfig config_;
    std::vector<ActiveSighting> active_;
    uint64_t next_sequence_ = 1;
};
