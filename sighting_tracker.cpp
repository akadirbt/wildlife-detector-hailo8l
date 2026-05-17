#include "sighting_tracker.hpp"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <utility>

SightingTracker::SightingTracker(SightingTrackerConfig config)
    : config_(std::move(config)) {}

std::vector<DetectionEvent> SightingTracker::update(
    const std::vector<TrackedDetection>& detections,
    int frame_w,
    int frame_h,
    std::chrono::steady_clock::time_point now)
{
    std::vector<DetectionEvent> events;
    for (auto& sighting : active_) {
        sighting.matched_this_update = false;
    }

    for (const auto& detection : detections) {
        if (detection.label.empty() || detection.confidence <= 0.0f) {
            continue;
        }

        const int match_index = find_match(detection);
        if (match_index < 0) {
            ActiveSighting sighting;
            sighting.sighting_id = make_sighting_id(detection.label, next_sequence_++);
            sighting.label = detection.label;
            sighting.priority = priority_for_label(detection.label);
            sighting.box = detection.box;
            sighting.confidence = detection.confidence;
            sighting.best_confidence = detection.confidence;
            sighting.frame_w = frame_w;
            sighting.frame_h = frame_h;
            sighting.snapshot = make_snapshot_path(sighting.sighting_id);
            sighting.first_seen = now;
            sighting.last_seen = now;
            sighting.last_update_emitted = now;
            sighting.matched_this_update = true;

            active_.push_back(sighting);
            events.push_back(make_event(DetectionEventType::SightingStart, active_.back(), now));
            continue;
        }

        auto& sighting = active_[static_cast<size_t>(match_index)];
        sighting.box = detection.box;
        sighting.confidence = detection.confidence;
        sighting.frame_w = frame_w;
        sighting.frame_h = frame_h;
        sighting.last_seen = now;
        sighting.matched_this_update = true;

        if (detection.confidence > sighting.best_confidence) {
            sighting.best_confidence = detection.confidence;
        }

        const auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - sighting.last_update_emitted).count();
        if (since_update >= config_.update_interval_ms) {
            sighting.last_update_emitted = now;
            events.push_back(make_event(DetectionEventType::SightingUpdate, sighting, now));
        }
    }

    auto it = active_.begin();
    while (it != active_.end()) {
        const auto since_seen = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - it->last_seen).count();
        if (!it->matched_this_update && since_seen >= config_.lost_timeout_ms) {
            events.push_back(make_event(DetectionEventType::SightingEnd, *it, now));
            it = active_.erase(it);
        } else {
            ++it;
        }
    }

    return events;
}

void SightingTracker::reset() {
    active_.clear();
}

float SightingTracker::iou(const DetectionBox& a, const DetectionBox& b) {
    const int left = std::max(a.x1, b.x1);
    const int top = std::max(a.y1, b.y1);
    const int right = std::min(a.x2, b.x2);
    const int bottom = std::min(a.y2, b.y2);

    const int intersection_w = std::max(0, right - left);
    const int intersection_h = std::max(0, bottom - top);
    const float intersection = static_cast<float>(intersection_w * intersection_h);

    const float area_a = static_cast<float>(
        std::max(0, a.x2 - a.x1) * std::max(0, a.y2 - a.y1));
    const float area_b = static_cast<float>(
        std::max(0, b.x2 - b.x1) * std::max(0, b.y2 - b.y1));
    const float union_area = area_a + area_b - intersection;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return intersection / union_area;
}

std::string SightingTracker::priority_for_label(const std::string& label) {
    if (label == "bear") {
        return "critical";
    }
    if (label == "raccoon" || label == "fox" || label == "skunk" || label == "coyote") {
        return "priority";
    }
    return "routine";
}

std::string SightingTracker::today_folder() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm {};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &local_tm);
#endif

    char date[16] = {};
    std::strftime(date, sizeof(date), "%Y-%m-%d", &local_tm);
    return date;
}

std::string SightingTracker::make_sighting_id(const std::string& label, uint64_t sequence) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm {};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &local_tm);
#endif

    char timestamp[32] = {};
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d-%H%M%S", &local_tm);

    std::ostringstream out;
    out << timestamp << "-" << label << "-"
        << std::setfill('0') << std::setw(3) << sequence;
    return out.str();
}

std::string SightingTracker::make_snapshot_path(const std::string& sighting_id) const {
    std::ostringstream out;
    out << config_.snapshot_root << "/" << today_folder() << "/" << sighting_id << ".jpg";
    return out.str();
}

DetectionEvent SightingTracker::make_event(
    DetectionEventType type,
    const ActiveSighting& sighting,
    std::chrono::steady_clock::time_point) const
{
    DetectionEvent event;
    event.schema_version = 1;
    event.detector_id = config_.detector_id;
    event.source = config_.source;
    event.event_type = type;
    event.sighting_id = sighting.sighting_id;
    event.timestamp = current_iso8601_local_timestamp();
    event.label = sighting.label;
    event.confidence = sighting.confidence;
    event.priority = sighting.priority;
    event.frame_w = sighting.frame_w;
    event.frame_h = sighting.frame_h;
    event.box = sighting.box;
    event.snapshot = sighting.snapshot;
    return event;
}

int SightingTracker::find_match(const TrackedDetection& detection) const {
    int best_index = -1;
    float best_iou = 0.0f;
    for (size_t i = 0; i < active_.size(); ++i) {
        const auto& sighting = active_[i];
        if (sighting.label != detection.label) {
            continue;
        }

        const float overlap = iou(sighting.box, detection.box);
        if (overlap >= config_.match_iou_threshold && overlap > best_iou) {
            best_iou = overlap;
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}
