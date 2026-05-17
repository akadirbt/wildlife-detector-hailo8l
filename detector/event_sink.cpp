#include "event_sink.hpp"

#include <filesystem>
#include <fstream>
#include <utility>

#if !defined(_WIN32)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

EventSink::EventSink(EventSinkConfig config)
    : config_(std::move(config)) {}

EventSink::~EventSink() {
    stop();
}

void EventSink::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) {
        return;
    }
    stop_requested_ = false;
    running_ = true;
    writer_ = std::thread([this]() { writer_loop(); });
}

void EventSink::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            return;
        }
        stop_requested_ = true;
    }
    cv_.notify_all();
    if (writer_.joinable()) {
        writer_.join();
    }
    close_uds();

    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
}

void EventSink::enqueue(DetectionEvent event) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(event));
        ++events_enqueued_;
    }
    cv_.notify_one();
}

EventSinkStats EventSink::stats() const {
    EventSinkStats out;
    out.events_enqueued = events_enqueued_.load();
    out.events_written = events_written_.load();
    out.jsonl_write_error_count = jsonl_write_error_count_.load();
    out.uds_send_error_count = uds_send_error_count_.load();
    out.uds_connect_error_count = uds_connect_error_count_.load();
    out.uds_connected = uds_connected_.load();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        out.queue_depth = queue_.size();
    }
    return out;
}

void EventSink::writer_loop() {
    while (true) {
        DetectionEvent event;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return stop_requested_ || !queue_.empty();
            });

            if (queue_.empty() && stop_requested_) {
                break;
            }

            event = std::move(queue_.front());
            queue_.pop_front();
        }

        const std::string line = detection_event_to_json(event);
        bool wrote_anywhere = false;

        if (config_.enable_jsonl) {
            wrote_anywhere = append_jsonl(event, line) || wrote_anywhere;
        }
        if (config_.enable_uds) {
            wrote_anywhere = send_uds(line) || wrote_anywhere;
        }

        if (wrote_anywhere) {
            ++events_written_;
        }
    }
}

bool EventSink::append_jsonl(const DetectionEvent& event, const std::string& line) {
    try {
        std::string date = "unknown-date";
        if (event.timestamp.size() >= 10) {
            date = event.timestamp.substr(0, 10);
        }

        const std::filesystem::path day_dir =
            std::filesystem::path(config_.jsonl_root) / date;
        std::filesystem::create_directories(day_dir);

        const std::filesystem::path path = day_dir / "events.jsonl";
        std::ofstream out(path, std::ios::app);
        if (!out.is_open()) {
            ++jsonl_write_error_count_;
            return false;
        }
        out << line << '\n';
        if (!out.good()) {
            ++jsonl_write_error_count_;
            return false;
        }
        return true;
    } catch (...) {
        ++jsonl_write_error_count_;
        return false;
    }
}

bool EventSink::send_uds(const std::string& line) {
#if defined(_WIN32)
    (void)line;
    return false;
#else
    if (!ensure_uds_connected()) {
        return false;
    }

    std::string payload = line;
    payload.push_back('\n');

    const ssize_t rc = send(
        uds_fd_,
        payload.data(),
        payload.size(),
        MSG_DONTWAIT | MSG_NOSIGNAL);
    if (rc < 0 || static_cast<size_t>(rc) != payload.size()) {
        ++uds_send_error_count_;
        close_uds();
        return false;
    }
    return true;
#endif
}

bool EventSink::ensure_uds_connected() {
#if defined(_WIN32)
    return false;
#else
    if (uds_connected_.load() && uds_fd_ >= 0) {
        return true;
    }

    uds_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (uds_fd_ < 0) {
        ++uds_connect_error_count_;
        return false;
    }

    const int flags = fcntl(uds_fd_, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(uds_fd_, F_SETFL, flags | O_NONBLOCK);
    }

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    if (config_.uds_path.size() >= sizeof(addr.sun_path)) {
        ++uds_connect_error_count_;
        close_uds();
        return false;
    }
    std::strncpy(addr.sun_path, config_.uds_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(uds_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (errno != EINPROGRESS && errno != EAGAIN) {
            ++uds_connect_error_count_;
            close_uds();
            return false;
        }
    }

    uds_connected_ = true;
    return true;
#endif
}

void EventSink::close_uds() {
#if !defined(_WIN32)
    if (uds_fd_ >= 0) {
        close(uds_fd_);
    }
#endif
    uds_fd_ = -1;
    uds_connected_ = false;
}
