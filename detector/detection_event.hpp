#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

enum class DetectionEventType {
    SightingStart,
    SightingUpdate,
    SightingEnd,
};

struct DetectionBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};

struct DetectionEvent {
    int schema_version = 1;
    std::string detector_id = "garden-pi-01";
    std::string source = "hailo_detector";
    DetectionEventType event_type = DetectionEventType::SightingUpdate;
    std::string sighting_id;
    std::string timestamp;
    std::string label;
    float confidence = 0.0f;
    std::string priority = "routine";
    int frame_w = 0;
    int frame_h = 0;
    DetectionBox box;
    std::string snapshot;
};

inline const char* detection_event_type_to_string(DetectionEventType type) {
    switch (type) {
        case DetectionEventType::SightingStart:
            return "sighting_start";
        case DetectionEventType::SightingUpdate:
            return "sighting_update";
        case DetectionEventType::SightingEnd:
            return "sighting_end";
    }
    return "sighting_update";
}

inline std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                } else {
                    out << ch;
                }
                break;
        }
    }
    return out.str();
}

inline std::string current_iso8601_local_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm local_tm {};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &local_tm);
#endif

    char timestamp[40] = {};
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z", &local_tm);

    std::string result(timestamp);
    if (result.size() == 24) {
        result.insert(result.size() - 2, ":");
    }
    return result;
}

inline std::string detection_event_to_json(const DetectionEvent& event) {
    std::ostringstream out;
    out << "{"
        << "\"schema_version\":" << event.schema_version << ","
        << "\"detector_id\":\"" << json_escape(event.detector_id) << "\","
        << "\"source\":\"" << json_escape(event.source) << "\","
        << "\"event_type\":\"" << detection_event_type_to_string(event.event_type) << "\","
        << "\"sighting_id\":\"" << json_escape(event.sighting_id) << "\","
        << "\"timestamp\":\"" << json_escape(event.timestamp) << "\","
        << "\"label\":\"" << json_escape(event.label) << "\","
        << "\"confidence\":" << std::fixed << std::setprecision(3) << event.confidence << ","
        << "\"priority\":\"" << json_escape(event.priority) << "\","
        << "\"frame_w\":" << event.frame_w << ","
        << "\"frame_h\":" << event.frame_h << ","
        << "\"box\":{"
        << "\"x1\":" << event.box.x1 << ","
        << "\"y1\":" << event.box.y1 << ","
        << "\"x2\":" << event.box.x2 << ","
        << "\"y2\":" << event.box.y2
        << "},"
        << "\"snapshot\":\"" << json_escape(event.snapshot) << "\""
        << "}";
    return out.str();
}
