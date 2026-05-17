#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "detection_event.hpp"

struct EventSinkConfig {
    std::string jsonl_root = "detections";
    std::string uds_path = "/tmp/det.sock";
    bool enable_jsonl = true;
    bool enable_uds = true;
};

struct EventSinkStats {
    uint64_t events_enqueued = 0;
    uint64_t events_written = 0;
    uint64_t jsonl_write_error_count = 0;
    uint64_t uds_send_error_count = 0;
    uint64_t uds_connect_error_count = 0;
    uint64_t queue_depth = 0;
    bool uds_connected = false;
};

class EventSink {
public:
    explicit EventSink(EventSinkConfig config = {});
    ~EventSink();

    EventSink(const EventSink&) = delete;
    EventSink& operator=(const EventSink&) = delete;

    void start();
    void stop();
    void enqueue(DetectionEvent event);
    EventSinkStats stats() const;

private:
    void writer_loop();
    bool append_jsonl(const DetectionEvent& event, const std::string& line);
    bool send_uds(const std::string& line);
    bool ensure_uds_connected();
    void close_uds();

    EventSinkConfig config_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<DetectionEvent> queue_;
    std::thread writer_;
    bool running_ = false;
    bool stop_requested_ = false;

    std::atomic<uint64_t> events_enqueued_ {0};
    std::atomic<uint64_t> events_written_ {0};
    std::atomic<uint64_t> jsonl_write_error_count_ {0};
    std::atomic<uint64_t> uds_send_error_count_ {0};
    std::atomic<uint64_t> uds_connect_error_count_ {0};

    int uds_fd_ = -1;
    std::atomic<bool> uds_connected_ {false};
};
