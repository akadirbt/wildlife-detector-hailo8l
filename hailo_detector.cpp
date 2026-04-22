/*
 * ============================================================
 *  Wildlife Detector — Raspberry Pi 5 + Hailo-8L
 * ============================================================
 *
 *  THIS FILE IS INTENTIONALLY "SINGLE FILE AND EASY TO TWEAK".
 *  The goal is simple field use, not a big enterprise architecture.
 *
 *  QUICK START
 *  1. Build:
 *       cmake --build build --target hailo_detector -j4
 *
 *  2. Run:
 *       ./build/hailo_detector
 *
 *  3. Open live preview in a browser:
 *       http://127.0.0.1:8090
 *
 *  NORMAL DAILY USE
 *  - You do NOT need to type flags like --serve or --preview-raw.
 *  - Those are already enabled by default in the config section below.
 *  - In most cases you should only run the binary.
 *
 *  HARDWARE THIS FILE EXPECTS
 *  - Camera: IMX708 / Camera Module 3 over CSI
 *  - Hailo model: YOLOv5 HEF loaded from the path below
 *  - Audio amp: MAX98357A
 *      DIN  -> GPIO21
 *      BCLK -> GPIO18
 *      LRC  -> GPIO19
 *  - Ultrasonic piezo deterrent
 *      MOSFET gate -> GPIO13
 *  - PIR status LEDs
 *      Green -> GPIO17
 *      Red   -> GPIO22
 *  - Temperature ADC: ADS1115 on I2C-1
 *      SDA1 -> GPIO2
 *      SCL1 -> GPIO3
 *      TMP36 output -> ADS1115 A0
 *
 *  IF YOU WANT TO CHANGE ANYTHING
 *  - First look at the "EASY-TO-EDIT SETTINGS" section near the top.
 *  - Most common changes are there:
 *      model path
 *      preview port
 *      confidence threshold
 *      audio files path
 *      ADS1115 address / I2C device
 *      camera FPS
 *
 *  WHAT THIS PROGRAM DOES
 *  1. Opens the CSI camera
 *  2. Runs Hailo inference on the latest frame
 *  3. Draws detection boxes and overlay text
 *  4. Streams the preview in MJPEG over HTTP
 *  5. Plays an animal-specific deterrent sound
 *  6. Reads TMP36 temperature through ADS1115 and shows Fahrenheit
 * ============================================================
 */

#include <iostream>
#include <filesystem>
#include <fstream>
#include <array>
#include <deque>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <csignal>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <condition_variable>
#include <optional>
#include <sstream>
#include <unordered_map>

#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "hailo/hailort.hpp"
#include "hailo/inference_pipeline.hpp"
#include "gpio_trigger.hpp"

using namespace hailort;

// ============================================================
//  EASY-TO-EDIT SETTINGS
//
//  If you need to change a path, pin, port, threshold, or default
//  runtime behavior, start here first.
//
//  The values below are the built-in defaults used when you simply run:
//      ./build/hailo_detector
// ============================================================

// ------------------------------
// Hardware wiring notes
// ------------------------------
// These GPIO values are documented here so they are easy to find.
// Some of them are configured by Raspberry Pi OS overlays rather than
// directly toggled in this C++ file, but keeping them here makes the
// full system easier to understand.
static constexpr int CAMERA_CSI_USED = 1;        // Informational only
static constexpr int MAX98357A_DIN_GPIO = 21;    // I2S data out from Pi
static constexpr int MAX98357A_BCLK_GPIO = 18;   // I2S bit clock
static constexpr int MAX98357A_LRC_GPIO = 19;    // I2S word select / LRCLK
static constexpr int I2C_SDA_GPIO = 2;           // SDA1
static constexpr int I2C_SCL_GPIO = 3;           // SCL1
static constexpr int ADS1115_TMP36_CHANNEL = 0;  // TMP36 is connected to A0
static constexpr int RESERVED_PIR_GPIO = 23;     // PIR signal GPIO (informational only)
static constexpr int ULTRASONIC_PIEZO_GPIO = 13; // Ultrasonic piezo MOSFET gate
static constexpr int PIR_ACTIVE_LED_GPIO = 17;   // Green LED
static constexpr int PIR_WAITING_LED_GPIO = 22;  // Red LED

// ------------------------------
// Main runtime defaults
// ------------------------------
static const std::string HEF_PATH =
    "/home/ecet400/animal_yolov5s_9class/animal_yolov5s_9class_h8l.hef";
static const std::string SOUNDS_DIR =
    "/home/ecet400/hailo_detector/sounds_i2s";

// Model input resolution. These must match the trained network.
static const int INPUT_W = 640;
static const int INPUT_H = 640;

// Camera preview resolution when raw preview mode is enabled.
static const int RAW_CAMERA_W = 1536;
static const int RAW_CAMERA_H = 864;

// Most important detector tuning knob.
static const float DEFAULT_CONF_THRESH = 0.40f;

// In normal daily use we want the web preview and raw-aspect preview on
// by default so you can simply run the binary with no extra flags.
static constexpr bool DEFAULT_ENABLE_PREVIEW_SERVER = true;
static constexpr bool DEFAULT_ENABLE_RAW_PREVIEW = true;
static const std::string DEFAULT_PREVIEW_HOST = "0.0.0.0";
static constexpr int DEFAULT_PREVIEW_PORT = 8090;
static constexpr int DEFAULT_PREVIEW_PORT_TRIES = 10;
static constexpr int DEFAULT_CAMERA_FPS = 60;

// Camera autofocus defaults tuned for your current setup.
static const std::string DEFAULT_AF_MODE = "continuous";
static const std::string DEFAULT_AF_RANGE = "full";
static const std::string DEFAULT_AF_SPEED = "fast";

// Postprocess cleanup after the HEF's own NMS output.
static const int   NUM_CLASSES = 9;
static const int   MAX_NMS_BOXES_PER_CLASS = 100;
static const float SOFTWARE_NMS_IOU_THRESH = 0.45f;
static const int   SOFTWARE_MIN_BOX_SIZE = 12;
static const float SOFTWARE_MAX_ASPECT_RATIO = 6.0f;
static const int   SOFTWARE_MAX_DETECTIONS = 12;

// Terminal logging behavior for repeated detections.
static const float LOG_MATCH_IOU_THRESH = 0.60f;
static constexpr float LOG_REPEAT_INTERVAL_SEC = 2.0f;
static constexpr float LOG_STALE_TRACK_SEC = 1.5f;

// Temperature sensor settings.
static const char* ADS1115_I2C_DEVICE = "/dev/i2c-1";
static constexpr uint8_t ADS1115_I2C_ADDR = 0x48;
static constexpr int TEMP_SENSOR_READ_INTERVAL_MS = 500;
static constexpr int TEMP_SENSOR_SAMPLES_PER_READ = 3;
static constexpr int TEMP_SENSOR_SAMPLE_DELAY_MS = 8;
static constexpr int TEMP_SENSOR_READY_POLL_INTERVAL_MS = 2;
static constexpr int TEMP_SENSOR_READY_TIMEOUT_MS = 30;
static constexpr int TEMP_SENSOR_READ_RETRIES = 3;
static constexpr float TEMP_SENSOR_FILTER_ALPHA = 0.70f;
static constexpr float TEMP_SENSOR_FAST_TRACK_DELTA_F = 2.5f;

// Audio deterrent settings.
static constexpr float AUDIO_PLAYBACK_COOLDOWN_SEC = 6.0f;
static constexpr float PIR_MOTION_CHECK_SEC = 1.0f;
static constexpr float PIR_INFERENCE_WINDOW_SEC = 7.0f;
static constexpr float PIR_CONFIRMED_MISSING_SEC = 3.0f;
// Full path to aplay binary — avoids PATH issues when the program is run with
// sudo or from a restricted shell environment (e.g. systemd service).
static const std::string APLAY_BIN = "/usr/bin/aplay";

// Detection frame saving settings.
static constexpr bool DEFAULT_SAVE_DETECTION_FRAMES = true;
static const std::string DEFAULT_DETECTION_FRAME_DIR =
    "/home/ecet400/hailo_detector/detections";
static constexpr float DETECTION_FRAME_SAVE_INTERVAL_SEC = 1.0f;

static const std::vector<std::string> CLASS_NAMES = {
    "bear", "coyote", "deer", "fox", "possum",
    "raccoon", "skunk", "squirrel", "turkey"
};

static volatile bool g_running = true;


// ============================================================
//  DATA TYPES
// ============================================================

struct Detection {
    int   class_id;
    float score;
    int   x1, y1, x2, y2;
};

enum class PirGateState {
    Waiting,
    WaitPirLow,
    MotionCheck,
    InferenceWindow,
    ConfirmedHold,
};

static const char* pir_gate_state_name(PirGateState state) {
    switch (state) {
        case PirGateState::Waiting: return "WAITING";
        case PirGateState::WaitPirLow: return "WAIT_PIR_LOW";
        case PirGateState::MotionCheck: return "MOTION_CHECK";
        case PirGateState::InferenceWindow: return "INFERENCE_WINDOW";
        case PirGateState::ConfirmedHold: return "CONFIRMED_HOLD";
    }
    return "UNKNOWN";
}

class PirGateFsm {
public:
    PirGateFsm()
        : state_(PirGateState::Waiting)
        , state_started_at_(std::chrono::steady_clock::now())
        , last_detection_at_(state_started_at_)
        , last_pir_high_(false)
        , motion_check_samples_(0)
        , motion_check_high_samples_(0)
    {}

    void update_before_inference(bool pir_high) {
        const auto now = std::chrono::steady_clock::now();

        switch (state_) {
            case PirGateState::Waiting:
                if (pir_high && !last_pir_high_) {
                    enter(PirGateState::MotionCheck, now);
                    sample_motion_check(pir_high);
                }
                break;

            case PirGateState::WaitPirLow:
                if (!pir_high) {
                    enter(PirGateState::Waiting, now);
                    last_pir_high_ = false;
                }
                break;

            case PirGateState::MotionCheck:
                sample_motion_check(pir_high);
                if (elapsed_sec(now) >= PIR_MOTION_CHECK_SEC) {
                    const bool motion_confirmed =
                        motion_check_high_samples_ > 0;
                    enter(motion_confirmed
                              ? PirGateState::InferenceWindow
                              : PirGateState::Waiting,
                          now);
                }
                break;

            case PirGateState::InferenceWindow:
                if (elapsed_sec(now) >= PIR_INFERENCE_WINDOW_SEC) {
                    enter(PirGateState::WaitPirLow, now);
                }
                break;

            case PirGateState::ConfirmedHold:
                if (std::chrono::duration<float>(now - last_detection_at_).count() >=
                    PIR_CONFIRMED_MISSING_SEC) {
                    enter(PirGateState::WaitPirLow, now);
                }
                break;
        }

        last_pir_high_ = pir_high;
    }

    void update_after_inference(bool detected) {
        const auto now = std::chrono::steady_clock::now();
        if (!detected) {
            return;
        }

        last_detection_at_ = now;
        if (state_ == PirGateState::InferenceWindow) {
            enter(PirGateState::ConfirmedHold, now);
            last_detection_at_ = now;
        }
    }

    bool inference_enabled() const {
        return state_ == PirGateState::InferenceWindow ||
               state_ == PirGateState::ConfirmedHold;
    }

    bool waiting_indicator() const {
        return !inference_enabled();
    }

    PirGateState state() const {
        return state_;
    }

private:
    float elapsed_sec(std::chrono::steady_clock::time_point now) const {
        return std::chrono::duration<float>(now - state_started_at_).count();
    }

    void enter(PirGateState next, std::chrono::steady_clock::time_point now) {
        if (next == state_) {
            return;
        }

        std::cout << "[PIR_FSM] " << pir_gate_state_name(state_)
                  << " -> " << pir_gate_state_name(next) << "\n";

        state_ = next;
        state_started_at_ = now;
        if (state_ == PirGateState::MotionCheck) {
            motion_check_samples_ = 0;
            motion_check_high_samples_ = 0;
        }
        if (state_ == PirGateState::ConfirmedHold) {
            last_detection_at_ = now;
        }
    }

    void sample_motion_check(bool pir_high) {
        ++motion_check_samples_;
        if (pir_high) {
            ++motion_check_high_samples_;
        }
    }

    PirGateState state_;
    std::chrono::steady_clock::time_point state_started_at_;
    std::chrono::steady_clock::time_point last_detection_at_;
    bool last_pir_high_;
    int motion_check_samples_;
    int motion_check_high_samples_;
};

struct LoggedDetectionTrack {
    Detection detection;
    std::chrono::steady_clock::time_point last_seen_at;
    std::chrono::steady_clock::time_point last_logged_at;
};

struct RuntimeConfig {
    std::optional<std::string> image_path;
    std::optional<std::string> output_path;
    float conf_thresh = DEFAULT_CONF_THRESH;
    int fps = DEFAULT_CAMERA_FPS;
    int capture_width = INPUT_W;
    int capture_height = INPUT_H;
    std::string af_mode = DEFAULT_AF_MODE;
    std::string af_range = DEFAULT_AF_RANGE;
    std::string af_speed = DEFAULT_AF_SPEED;
    std::optional<float> lens_position;
    bool disable_ae = false;
    bool disable_awb = false;
    bool preview_raw = DEFAULT_ENABLE_RAW_PREVIEW;
    bool serve = DEFAULT_ENABLE_PREVIEW_SERVER;
    std::string serve_host = DEFAULT_PREVIEW_HOST;
    int serve_port = DEFAULT_PREVIEW_PORT;
    int serve_port_tries = DEFAULT_PREVIEW_PORT_TRIES;
    bool save_detection_frames = DEFAULT_SAVE_DETECTION_FRAMES;
    std::string detection_frame_dir = DEFAULT_DETECTION_FRAME_DIR;
};


// ============================================================
//  SIGNAL HANDLER
// ============================================================

static void handle_sigint(int) {
    g_running = false;
}

static std::string label_for_class(int class_id) {
    if (class_id >= 0 && class_id < static_cast<int>(CLASS_NAMES.size())) {
        return CLASS_NAMES[class_id];
    }
    return "cls" + std::to_string(class_id);
}

static void print_detection_line(const char* prefix, const Detection& det) {
    const std::string label = label_for_class(det.class_id);
    std::printf("%s %-12s %.2f  [%d,%d,%d,%d]\n",
                prefix,
                label.c_str(),
                det.score,
                det.x1, det.y1, det.x2, det.y2);
}

static float detection_iou(const Detection& a, const Detection& b) {
    const int left = std::max(a.x1, b.x1);
    const int top = std::max(a.y1, b.y1);
    const int right = std::min(a.x2, b.x2);
    const int bottom = std::min(a.y2, b.y2);

    const int intersection_w = std::max(0, right - left);
    const int intersection_h = std::max(0, bottom - top);
    const float intersection = static_cast<float>(intersection_w * intersection_h);

    const float area_a = static_cast<float>(std::max(0, a.x2 - a.x1) * std::max(0, a.y2 - a.y1));
    const float area_b = static_cast<float>(std::max(0, b.x2 - b.x1) * std::max(0, b.y2 - b.y1));
    const float union_area = area_a + area_b - intersection;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return intersection / union_area;
}

class DetectionEventLogger {
public:
    // We do not want to spam the terminal on every single frame.
    // This helper treats nearby repeated boxes as the same "ongoing event"
    // and only re-logs occasionally.
    void log(const std::vector<Detection>& detections) {
        const auto now = std::chrono::steady_clock::now();
        prune_stale_tracks(now);

        for (const auto& det : detections) {
            auto match_it = find_matching_track(det);
            if (match_it == tracks_.end()) {
                tracks_.push_back({det, now, now});
                print_detection_line("DETECTED:", det);
                continue;
            }

            match_it->detection = det;
            match_it->last_seen_at = now;

            const float since_last_log =
                std::chrono::duration<float>(now - match_it->last_logged_at).count();
            if (since_last_log >= LOG_REPEAT_INTERVAL_SEC) {
                match_it->last_logged_at = now;
                print_detection_line("TRACKING:", det);
            }
        }
    }

private:
    using TrackList = std::vector<LoggedDetectionTrack>;

    TrackList::iterator find_matching_track(const Detection& det) {
        TrackList::iterator best_match = tracks_.end();
        float best_iou = 0.0f;
        for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
            if (it->detection.class_id != det.class_id) {
                continue;
            }
            const float iou = detection_iou(it->detection, det);
            if (iou >= LOG_MATCH_IOU_THRESH && iou > best_iou) {
                best_iou = iou;
                best_match = it;
            }
        }
        return best_match;
    }

    void prune_stale_tracks(const std::chrono::steady_clock::time_point& now) {
        tracks_.erase(
            std::remove_if(
                tracks_.begin(),
                tracks_.end(),
                [&](const LoggedDetectionTrack& track) {
                    const float idle_for =
                        std::chrono::duration<float>(now - track.last_seen_at).count();
                    return idle_for >= LOG_STALE_TRACK_SEC;
                }),
            tracks_.end());
    }

    TrackList tracks_;
};

static std::string summarize_labels_for_filename(const std::vector<Detection>& detections);

class DetectionFrameSaver {
public:
    explicit DetectionFrameSaver(const RuntimeConfig& cfg)
        : enabled_(cfg.save_detection_frames)
        , output_dir_(cfg.detection_frame_dir) {
        if (!enabled_) {
            return;
        }

        try {
            std::filesystem::create_directories(output_dir_);
            std::cout << "[INFO] Detection frames will be saved to "
                      << output_dir_.string() << "\n";
        } catch (const std::exception& e) {
            enabled_ = false;
            std::cerr << "[WARN] Could not create detection frame directory "
                      << output_dir_.string() << ": " << e.what() << "\n";
        }
    }

    void maybe_save(
        const cv::Mat& rendered_frame,
        const std::vector<Detection>& detections,
        uint64_t frame_id,
        bool pir_gate_enabled)
    {
        if (!enabled_ || !pir_gate_enabled || rendered_frame.empty() || detections.empty()) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (last_saved_at_.has_value() &&
            std::chrono::duration<float>(now - *last_saved_at_).count() < DETECTION_FRAME_SAVE_INTERVAL_SEC) {
            return;
        }

        const std::filesystem::path out_path = output_dir_ / build_filename(detections, frame_id);
        if (!cv::imwrite(out_path.string(), rendered_frame)) {
            std::cerr << "[WARN] Failed to save detection frame: " << out_path << "\n";
            return;
        }

        last_saved_at_ = now;
        std::cout << "[SAVE] Detection frame: " << out_path << "\n";
    }

private:
    static std::string build_filename(const std::vector<Detection>& detections, uint64_t frame_id) {
        const auto now = std::chrono::system_clock::now();
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() % 1000;
        const std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm {};
        localtime_r(&now_time_t, &local_tm);

        char timestamp[32] = {};
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &local_tm);

        std::ostringstream oss;
        oss << timestamp
            << "_" << std::setfill('0') << std::setw(3) << millis
            << "_frame" << frame_id
            << "_" << summarize_labels_for_filename(detections)
            << ".jpg";
        return oss.str();
    }

    bool enabled_ = false;
    std::filesystem::path output_dir_;
    std::optional<std::chrono::steady_clock::time_point> last_saved_at_;
};

static void print_usage(const char* prog_name) {
    std::cout
        << "Usage: " << prog_name << " [optional overrides]\n"
        << "\n"
        << "Most users should simply run:\n"
        << "  " << prog_name << "\n"
        << "\n"
        << "Built-in defaults already enable:\n"
        << "  - live MJPEG preview\n"
        << "  - raw preview aspect ratio\n"
        << "  - camera mode\n"
        << "  - animal-specific audio deterrent\n"
        << "  - temperature overlay from ADS1115 + TMP36\n"
        << "\n"
        << "Optional overrides:\n"
        << "  --image <path>         Run a single image instead of camera mode\n"
        << "  --output <path>        Output image path for --image mode\n"
        << "  --conf <float>         Confidence threshold (default: " << DEFAULT_CONF_THRESH << ")\n"
        << "  --fps <int>            Requested camera FPS (default: " << DEFAULT_CAMERA_FPS << ")\n"
        << "  --capture-width <px>   Camera output width before inference (default: " << INPUT_W << ")\n"
        << "  --capture-height <px>  Camera output height before inference (default: " << INPUT_H << ")\n"
        << "  --af-mode <mode>       manual | auto | continuous (default: " << DEFAULT_AF_MODE << ")\n"
        << "  --af-range <range>     normal | macro | full (default: " << DEFAULT_AF_RANGE << ")\n"
        << "  --af-speed <speed>     normal | fast (default: " << DEFAULT_AF_SPEED << ")\n"
        << "  --lens-position <f>    Manual lens position when --af-mode manual is used\n"
        << "  --disable-ae           Disable auto exposure\n"
        << "  --disable-awb          Disable auto white balance\n"
        << "  --preview-raw          Force raw-aspect preview on\n"
        << "  --no-preview-raw       Force raw-aspect preview off\n"
        << "  --serve                Force preview server on\n"
        << "  --no-serve             Force preview server off\n"
        << "  --serve-host <host>    Preview bind host (default: " << DEFAULT_PREVIEW_HOST << ")\n"
        << "  --serve-port <port>    Preview start port (default: " << DEFAULT_PREVIEW_PORT << ")\n"
        << "  --save-detection-frames    Save rendered detection frames in camera mode\n"
        << "  --no-save-detection-frames Disable rendered detection frame saving\n"
        << "  --detection-dir <path>     Directory for saved detection frames (default: "
        << DEFAULT_DETECTION_FRAME_DIR << ")\n"
        << "  --help                 Show this help message\n";
}

static RuntimeConfig parse_args(int argc, char** argv) {
    RuntimeConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            cfg.image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            cfg.conf_thresh = std::stof(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            cfg.fps = std::stoi(argv[++i]);
        } else if (arg == "--capture-width" && i + 1 < argc) {
            cfg.capture_width = std::stoi(argv[++i]);
        } else if (arg == "--capture-height" && i + 1 < argc) {
            cfg.capture_height = std::stoi(argv[++i]);
        } else if (arg == "--af-mode" && i + 1 < argc) {
            cfg.af_mode = argv[++i];
        } else if (arg == "--af-range" && i + 1 < argc) {
            cfg.af_range = argv[++i];
        } else if (arg == "--af-speed" && i + 1 < argc) {
            cfg.af_speed = argv[++i];
        } else if (arg == "--lens-position" && i + 1 < argc) {
            cfg.lens_position = std::stof(argv[++i]);
        } else if (arg == "--disable-ae") {
            cfg.disable_ae = true;
        } else if (arg == "--disable-awb") {
            cfg.disable_awb = true;
        } else if (arg == "--preview-raw") {
            cfg.preview_raw = true;
        } else if (arg == "--no-preview-raw") {
            cfg.preview_raw = false;
        } else if (arg == "--serve") {
            cfg.serve = true;
        } else if (arg == "--no-serve") {
            cfg.serve = false;
        } else if (arg == "--serve-host" && i + 1 < argc) {
            cfg.serve_host = argv[++i];
        } else if (arg == "--serve-port" && i + 1 < argc) {
            cfg.serve_port = std::stoi(argv[++i]);
        } else if (arg == "--save-detection-frames") {
            cfg.save_detection_frames = true;
        } else if (arg == "--no-save-detection-frames") {
            cfg.save_detection_frames = false;
        } else if (arg == "--detection-dir" && i + 1 < argc) {
            cfg.detection_frame_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "[WARN] Unknown argument ignored: " << arg << "\n";
        }
    }
    return cfg;
}


// ============================================================
//  HELPERS
// ============================================================

// ============================================================
//  PREPROCESS
//  Converts OpenCV BGR frame to RGB uint8 [0,255]
//  This HEF expects UINT8 NHWC input.
// ============================================================

static void preprocess_frame(const cv::Mat& frame, std::vector<uint8_t>& out_data) {
    cv::Mat resized, rgb;

    // Resize to model input size if needed
    if (frame.cols != INPUT_W || frame.rows != INPUT_H)
        cv::resize(frame, resized, cv::Size(INPUT_W, INPUT_H));
    else
        resized = frame;

    // BGR → RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Flatten HWC to vector
    out_data.assign(rgb.data, rgb.data + INPUT_W * INPUT_H * 3);
}


// ============================================================
//  NMS OUTPUT DECODE
//  The new HEF already exposes yolov5_nms_postprocess, so we read
//  class-major boxes directly instead of decoding raw YOLO heads.
// ============================================================

static std::vector<Detection> decode_nms_output(
    const std::vector<float>& nms_tensor,
    float orig_w,
    float orig_h,
    float conf_thresh)
{
    std::vector<Detection> results;
    if (nms_tensor.empty()) {
        return results;
    }

    size_t offset = 0;
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        if (offset >= nms_tensor.size()) {
            break;
        }

        const int bbox_count = std::max(0, static_cast<int>(std::lround(nms_tensor[offset++])));
        if (bbox_count > MAX_NMS_BOXES_PER_CLASS) {
            std::cerr << "[WARN] Invalid bbox count " << bbox_count
                      << " for class " << class_id << "\n";
            return results;
        }

        for (int det_idx = 0; det_idx < bbox_count; ++det_idx) {
            if (offset + 5 > nms_tensor.size()) {
                std::cerr << "[WARN] Truncated NMS tensor while parsing class "
                          << class_id << "\n";
                return results;
            }

            const float y_min = nms_tensor[offset + 0];
            const float x_min = nms_tensor[offset + 1];
            const float y_max = nms_tensor[offset + 2];
            const float x_max = nms_tensor[offset + 3];
            const float score = nms_tensor[offset + 4];
            offset += 5;

            if (score < conf_thresh) {
                continue;
            }

            const int x1 = std::max(0, std::min(static_cast<int>(orig_w) - 1, static_cast<int>(x_min * orig_w)));
            const int y1 = std::max(0, std::min(static_cast<int>(orig_h) - 1, static_cast<int>(y_min * orig_h)));
            const int x2 = std::max(0, std::min(static_cast<int>(orig_w) - 1, static_cast<int>(x_max * orig_w)));
            const int y2 = std::max(0, std::min(static_cast<int>(orig_h) - 1, static_cast<int>(y_max * orig_h)));

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            results.push_back({class_id, score, x1, y1, x2, y2});
        }
    }

    std::sort(results.begin(), results.end(),
        [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });

    return results;
}

static cv::Scalar color_for_class(int class_id) {
    static const std::vector<cv::Scalar> palette = {
        {60, 130, 245}, {30, 180, 220}, {0, 180, 120}, {30, 120, 255}, {220, 180, 0},
        {200, 100, 255}, {255, 255, 255}, {120, 220, 120}, {140, 140, 255},
    };
    if (class_id >= 0 && class_id < static_cast<int>(palette.size())) {
        return palette[class_id];
    }
    return {0, 200, 255};
}

static bool box_is_reasonable(const Detection& det) {
    const int width = det.x2 - det.x1;
    const int height = det.y2 - det.y1;
    if (width < SOFTWARE_MIN_BOX_SIZE || height < SOFTWARE_MIN_BOX_SIZE) {
        return false;
    }

    const float safe_width = static_cast<float>(std::max(width, 1));
    const float safe_height = static_cast<float>(std::max(height, 1));
    const float aspect_ratio = std::max(safe_width / safe_height, safe_height / safe_width);
    return aspect_ratio <= SOFTWARE_MAX_ASPECT_RATIO;
}

static std::vector<Detection> software_postprocess(const std::vector<Detection>& detections) {
    std::vector<Detection> filtered;
    filtered.reserve(detections.size());

    for (const auto& det : detections) {
        if (box_is_reasonable(det)) {
            filtered.push_back(det);
        }
    }

    std::vector<Detection> merged;
    merged.reserve(filtered.size());

    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<Detection> class_detections;

        for (const auto& det : filtered) {
            if (det.class_id != class_id) {
                continue;
            }
            boxes.emplace_back(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
            scores.push_back(det.score);
            class_detections.push_back(det);
        }

        if (boxes.empty()) {
            continue;
        }

        std::vector<int> kept_indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.0f, SOFTWARE_NMS_IOU_THRESH, kept_indices);
        for (const int index : kept_indices) {
            if (index >= 0 && index < static_cast<int>(class_detections.size())) {
                merged.push_back(class_detections[index]);
            }
        }
    }

    std::sort(merged.begin(), merged.end(),
        [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });

    if (static_cast<int>(merged.size()) > SOFTWARE_MAX_DETECTIONS) {
        merged.resize(SOFTWARE_MAX_DETECTIONS);
    }

    return merged;
}

static void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        const cv::Scalar color = color_for_class(det.class_id);
        cv::rectangle(image, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), color, 2);

        const std::string label =
            CLASS_NAMES[det.class_id] + " " + cv::format("%.2f", det.score);
        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
        const int top = std::max(0, det.y1 - text_size.height - baseline - 6);
        cv::rectangle(image,
                      cv::Point(det.x1, top),
                      cv::Point(det.x1 + text_size.width + 8, top + text_size.height + baseline + 6),
                      color,
                      cv::FILLED);
        cv::putText(image,
                    label,
                    cv::Point(det.x1 + 4, top + text_size.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(20, 20, 20),
                    2,
                    cv::LINE_AA);
    }
}

static std::string summarize_labels(const std::vector<Detection>& detections) {
    std::vector<bool> seen(CLASS_NAMES.size(), false);
    std::vector<std::string> labels;
    labels.reserve(detections.size());

    for (const auto& det : detections) {
        if (det.class_id < 0 || det.class_id >= static_cast<int>(CLASS_NAMES.size())) {
            continue;
        }
        if (seen[det.class_id]) {
            continue;
        }
        seen[det.class_id] = true;
        labels.push_back(CLASS_NAMES[det.class_id]);
    }

    if (labels.empty()) {
        return "none";
    }

    std::ostringstream oss;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << labels[i];
    }
    return oss.str();
}

static std::string summarize_labels_for_filename(const std::vector<Detection>& detections) {
    std::string label_summary = summarize_labels(detections);
    for (char& ch : label_summary) {
        const bool is_ascii_alpha =
            (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
        const bool is_ascii_digit = (ch >= '0' && ch <= '9');
        if (!is_ascii_alpha && !is_ascii_digit) {
            ch = '_';
        }
    }

    std::string compact;
    compact.reserve(label_summary.size());
    bool previous_was_underscore = false;
    for (char ch : label_summary) {
        if (ch == '_') {
            if (previous_was_underscore) {
                continue;
            }
            previous_was_underscore = true;
        } else {
            previous_was_underscore = false;
        }
        compact.push_back(ch);
    }

    while (!compact.empty() && compact.front() == '_') {
        compact.erase(compact.begin());
    }
    while (!compact.empty() && compact.back() == '_') {
        compact.pop_back();
    }

    if (compact.empty()) {
        return "unknown";
    }
    return compact;
}

static void draw_summary(
    cv::Mat& image,
    int frame_count,
    const std::vector<Detection>& detections,
    float fps,
    std::optional<float> temperature_f)
{
    std::vector<std::string> lines = {
        "Frame: " + std::to_string(frame_count),
        "Detected: " + summarize_labels(detections),
        "FPS: " + cv::format("%.1f", fps),
    };
    if (temperature_f.has_value()) {
        lines.push_back("Temp: " + cv::format("%.1f F", *temperature_f));
    } else {
        lines.push_back("Temp: n/a");
    }

    int y = 28;
    for (const auto& line : lines) {
        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.72, 2, &baseline);
        const cv::Rect bg(10, y - text_size.height - 8, text_size.width + 16, text_size.height + baseline + 12);
        cv::rectangle(image, bg, cv::Scalar(20, 20, 20), cv::FILLED);
        cv::putText(image,
                    line,
                    cv::Point(bg.x + 8, y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.72,
                    cv::Scalar(255, 255, 255),
                    2,
                    cv::LINE_AA);
        y += text_size.height + baseline + 18;
    }
}

// ============================================================
//  TEMPERATURE SENSOR: ADS1115 + TMP36
//
//  What this class does:
//  - Opens ADS1115 on I2C-1
//  - Reads channel A0
//  - Converts TMP36 voltage to Celsius, then Fahrenheit
//  - Smooths the result so the overlay is easier to read
// ============================================================
class Ads1115Tmp36Sensor {
public:
    Ads1115Tmp36Sensor() {
        fd_ = open(ADS1115_I2C_DEVICE, O_RDWR);
        if (fd_ < 0) {
            std::cerr << "[WARN] Temperature sensor unavailable: cannot open "
                      << ADS1115_I2C_DEVICE
                      << ". Enable I2C and reboot if needed.\n";
            return;
        }

        if (ioctl(fd_, I2C_SLAVE, ADS1115_I2C_ADDR) < 0) {
            std::cerr << "[WARN] Temperature sensor unavailable: cannot talk to ADS1115 at 0x"
                      << std::hex << static_cast<int>(ADS1115_I2C_ADDR) << std::dec << ".\n";
            close(fd_);
            fd_ = -1;
            return;
        }

        available_ = true;
        std::cout << "[INFO] Temperature sensor ready on " << ADS1115_I2C_DEVICE
                  << " (ADS1115 0x48, TMP36 on A0)\n";

        worker_ = std::thread([this]() { sampling_loop(); });
    }

    ~Ads1115Tmp36Sensor() {
        stop_worker_ = true;
        if (worker_.joinable()) {
            worker_.join();
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    std::optional<float> latest_fahrenheit() {
        std::lock_guard<std::mutex> lock(temp_mutex_);
        return filtered_temp_f_;
    }

private:
    static constexpr uint8_t REG_CONVERSION = 0x00;
    static constexpr uint8_t REG_CONFIG = 0x01;
    static constexpr uint16_t CONFIG_OS_BIT = 0x8000;
    static constexpr uint16_t CONFIG_SINGLE_SHOT_A0 =
        0x8000 | // start single conversion
        0x4000 | // AIN0 versus GND
        0x0400 | // +/-2.048V full scale
        0x0100 | // single-shot mode
        0x0080 | // 128 samples per second
        0x0003;  // disable comparator

    bool write_register_u16(uint8_t reg, uint16_t value) {
        const uint8_t buffer[3] = {
            reg,
            static_cast<uint8_t>((value >> 8) & 0xFF),
            static_cast<uint8_t>(value & 0xFF)
        };
        return write(fd_, buffer, sizeof(buffer)) == static_cast<ssize_t>(sizeof(buffer));
    }

    std::optional<uint16_t> read_register_u16(uint8_t reg) {
        if (write(fd_, &reg, 1) != 1) {
            return std::nullopt;
        }

        uint8_t buffer[2] = {0, 0};
        if (read(fd_, buffer, sizeof(buffer)) != static_cast<ssize_t>(sizeof(buffer))) {
            return std::nullopt;
        }

        return static_cast<uint16_t>((buffer[0] << 8) | buffer[1]);
    }

    std::optional<float> read_stable_fahrenheit() {
        std::vector<float> samples;
        samples.reserve(TEMP_SENSOR_SAMPLES_PER_READ);

        for (int sample_index = 0; sample_index < TEMP_SENSOR_SAMPLES_PER_READ; ++sample_index) {
            const auto sample = read_single_sample_fahrenheit();
            if (sample.has_value()) {
                samples.push_back(*sample);
            }

            if (sample_index + 1 < TEMP_SENSOR_SAMPLES_PER_READ) {
                std::this_thread::sleep_for(std::chrono::milliseconds(TEMP_SENSOR_SAMPLE_DELAY_MS));
            }
        }

        if (samples.empty()) {
            return std::nullopt;
        }

        std::sort(samples.begin(), samples.end());
        if (samples.size() >= 3) {
            const float middle_sum = std::accumulate(samples.begin() + 1, samples.end() - 1, 0.0f);
            return middle_sum / static_cast<float>(samples.size() - 2);
        }

        return std::accumulate(samples.begin(), samples.end(), 0.0f) /
               static_cast<float>(samples.size());
    }

    std::optional<float> read_single_sample_fahrenheit() {
        for (int attempt = 0; attempt < TEMP_SENSOR_READ_RETRIES; ++attempt) {
            if (!write_register_u16(REG_CONFIG, CONFIG_SINGLE_SHOT_A0)) {
                continue;
            }

            if (!wait_for_conversion_ready()) {
                continue;
            }

            const auto raw_u16 = read_register_u16(REG_CONVERSION);
            if (!raw_u16.has_value()) {
                continue;
            }

            const int16_t raw = static_cast<int16_t>(*raw_u16);
            const float volts = static_cast<float>(raw) * 2.048f / 32768.0f;
            const float temp_c = (volts - 0.5f) * 100.0f;
            const float temp_f = (temp_c * 9.0f / 5.0f) + 32.0f;
            return temp_f;
        }

        warn_once("failed to read ADS1115 conversion");
        return std::nullopt;
    }

    bool wait_for_conversion_ready() {
        auto waited_ms = 0;
        while (waited_ms <= TEMP_SENSOR_READY_TIMEOUT_MS) {
            const auto config = read_register_u16(REG_CONFIG);
            if (config.has_value() && ((*config & CONFIG_OS_BIT) != 0)) {
                return true;
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(TEMP_SENSOR_READY_POLL_INTERVAL_MS));
            waited_ms += TEMP_SENSOR_READY_POLL_INTERVAL_MS;
        }

        return false;
    }

    void sampling_loop() {
        while (!stop_worker_) {
            const auto temp = read_stable_fahrenheit();
            if (temp.has_value()) {
                std::lock_guard<std::mutex> lock(temp_mutex_);
                if (filtered_temp_f_.has_value()) {
                    const float delta_f = std::fabs(*temp - *filtered_temp_f_);
                    if (delta_f >= TEMP_SENSOR_FAST_TRACK_DELTA_F) {
                        filtered_temp_f_ = *temp;
                    } else {
                        filtered_temp_f_ =
                            ((1.0f - TEMP_SENSOR_FILTER_ALPHA) * *filtered_temp_f_) +
                            (TEMP_SENSOR_FILTER_ALPHA * *temp);
                    }
                } else {
                    filtered_temp_f_ = temp;
                }
            }

            const auto sleep_chunk = std::chrono::milliseconds(100);
            auto remaining = std::chrono::milliseconds(TEMP_SENSOR_READ_INTERVAL_MS);
            while (!stop_worker_ && remaining.count() > 0) {
                const auto this_sleep = std::min(remaining, sleep_chunk);
                std::this_thread::sleep_for(this_sleep);
                remaining -= this_sleep;
            }
        }
    }

    void warn_once(const std::string& message) {
        if (warned_read_failure_) {
            return;
        }
        warned_read_failure_ = true;
        std::cerr << "[WARN] Temperature sensor read failed: " << message << "\n";
    }

    int fd_ = -1;
    bool available_ = false;
    bool warned_read_failure_ = false;
    std::atomic<bool> stop_worker_ {false};
    std::thread worker_;
    std::mutex temp_mutex_;
    std::optional<float> filtered_temp_f_;
};

static void draw_pir_status(cv::Mat& image, bool waiting) {
    const std::string text = waiting ? "PIR: waiting" : "PIR: active";
    const cv::Scalar bg = waiting ? cv::Scalar(30, 60, 160) : cv::Scalar(40, 120, 40);

    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.72, 2, &baseline);
    const int x = std::max(10, image.cols - text_size.width - 26);
    const int y = 28;
    const cv::Rect rect(
        x,
        y - text_size.height - 8,
        text_size.width + 16,
        text_size.height + baseline + 12);
    cv::rectangle(image, rect, bg, cv::FILLED);
    cv::putText(image,
                text,
                cv::Point(rect.x + 8, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.72,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);
}

static std::string sound_name_for_label(const std::string& label) {
    // Heuristic deterrent mapping using the four available sounds.
    // Prefer human voice for large/confident mammals, dog bark for
    // nuisance mid-size mammals, hawk for small prey, and wolf howl
    // as a stronger predator cue for deer.
    if (label == "bear") {
        return "human.wav";
    }
    if (label == "coyote") {
        return "human.wav";
    }
    if (label == "deer") {
        return "wolf_howl.wav";
    }
    if (label == "fox") {
        return "dog_bark.wav";
    }
    if (label == "possum") {
        return "dog_bark.wav";
    }
    if (label == "raccoon") {
        return "dog_bark.wav";
    }
    if (label == "skunk") {
        return "human.wav";
    }
    if (label == "squirrel") {
        return "hawk.wav";
    }
    if (label == "turkey") {
        return "human.wav";
    }
    return "";
}

static std::optional<std::string> first_detected_label(const std::vector<Detection>& detections) {
    for (const auto& detection : detections) {
        if (detection.class_id < 0 || detection.class_id >= static_cast<int>(CLASS_NAMES.size())) {
            continue;
        }
        return CLASS_NAMES[detection.class_id];
    }
    return std::nullopt;
}

// ============================================================
//  AUDIO DETERRENT
//
//  This class maps detected animals to sound files and plays them
//  asynchronously so inference does not stop while audio is playing.
// ============================================================
class AudioDeterrent {
public:
    explicit AudioDeterrent(std::string sounds_dir)
        : sounds_dir_(std::move(sounds_dir)) {
        playback_device_ = detect_playback_device();
        if (!playback_device_.has_value()) {
            std::cerr << "[WARN] ALSA playback device not detected. "
                      << "Audio playback will fail until MAX98357A/I2S and /dev/snd are available.\n";
        } else {
            std::cout << "[INFO] Audio output device: " << *playback_device_ << "\n";
        }

        for (const auto& label : CLASS_NAMES) {
            const std::string sound_name = sound_name_for_label(label);
            if (sound_name.empty()) {
                continue;
            }
            const std::filesystem::path sound_path = std::filesystem::path(sounds_dir_) / sound_name;
            if (std::filesystem::exists(sound_path)) {
                label_to_sound_[label] = sound_path.string();
            } else {
                std::cerr << "[WARN] Missing sound for " << label << ": " << sound_path << "\n";
            }
        }

        if (!label_to_sound_.empty()) {
            std::cout << "[INFO] Audio deterrent loaded from " << sounds_dir_ << "\n";
        }
    }

    ~AudioDeterrent() {
        if (playback_thread_.joinable()) {
            playback_thread_.join();
        }
    }

    void maybe_play_for_detections(const std::vector<Detection>& detections) {
        for (const auto& detection : detections) {
            if (detection.class_id < 0 || detection.class_id >= static_cast<int>(CLASS_NAMES.size())) {
                continue;
            }
            const std::string& label = CLASS_NAMES[detection.class_id];
            auto it = label_to_sound_.find(label);
            if (it == label_to_sound_.end()) {
                continue;
            }
            maybe_play_for_label(label, it->second);
            return;
        }
    }

private:
    static std::optional<std::string> playback_device_from_env() {
        if (const char* env_value = std::getenv("WILDLIFE_AUDIO_DEVICE")) {
            if (env_value[0] != '\0') {
                return std::string(env_value);
            }
        }
        return std::nullopt;
    }

    static std::string capture_command_output(const char* cmd) {
        std::string output;
        std::array<char, 256> buffer{};
        FILE* pipe = popen(cmd, "r");
        if (nullptr == pipe) {
            return output;
        }

        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            output += buffer.data();
        }
        pclose(pipe);
        return output;
    }

    static std::optional<std::string> parse_max98357a_card_index(const std::string& text) {
        std::istringstream lines(text);
        std::string line;
        while (std::getline(lines, line)) {
            if (line.find("MAX98357A") == std::string::npos) {
                continue;
            }

            const std::string marker = "card ";
            const auto marker_pos = line.find(marker);
            if (marker_pos != std::string::npos) {
                size_t number_begin = marker_pos + marker.size();
                size_t number_end = number_begin;
                while (number_end < line.size() &&
                       std::isdigit(static_cast<unsigned char>(line[number_end]))) {
                    ++number_end;
                }

                if (number_end > number_begin) {
                    return "plughw:" + line.substr(number_begin, number_end - number_begin) + ",0";
                }
            }

            size_t number_begin = 0;
            while (number_begin < line.size() &&
                   std::isspace(static_cast<unsigned char>(line[number_begin]))) {
                ++number_begin;
            }

            size_t number_end = number_begin;
            while (number_end < line.size() &&
                   std::isdigit(static_cast<unsigned char>(line[number_end]))) {
                ++number_end;
            }

            if (number_end > number_begin) {
                return "plughw:" + line.substr(number_begin, number_end - number_begin) + ",0";
            }
        }

        return std::nullopt;
    }

    static std::optional<std::string> detect_playback_device() {
        if (const auto env_device = playback_device_from_env(); env_device.has_value()) {
            return env_device;
        }

        if (!std::filesystem::exists("/dev/snd")) {
            return std::nullopt;
        }

        const std::string aplay_list = capture_command_output((APLAY_BIN + " -l 2>/dev/null").c_str());
        if (const auto max98357a = parse_max98357a_card_index(aplay_list); max98357a.has_value()) {
            return max98357a;
        }

        // aplay -l ran but did not find MAX98357A — try /proc/asound/cards as a
        // second source before giving up.
        std::ifstream proc_cards("/proc/asound/cards");
        if (proc_cards.is_open()) {
            std::ostringstream proc_text;
            proc_text << proc_cards.rdbuf();
            if (const auto max98357a = parse_max98357a_card_index(proc_text.str());
                max98357a.has_value()) {
                return max98357a;
            }
        }

        // MAX98357A was not found by card number, but fall back to the card-name
        // form so ALSA can locate it regardless of what card index it gets
        // assigned.  Never return "default" — on this system "default" maps to
        // HDMI (card 0) and produces error 524 instead of playing through the
        // amplifier.
        if (!aplay_list.empty() || std::filesystem::exists("/proc/asound/cards")) {
            std::cerr << "[AUDIO] MAX98357A not found in ALSA card list; "
                      << "falling back to plughw:MAX98357A,0\n";
            return "plughw:MAX98357A,0";
        }

        return std::nullopt;
    }

    void maybe_play_for_label(const std::string& label, const std::string& sound_path) {
        const auto now = std::chrono::steady_clock::now();
        const std::string sound_filename =
            std::filesystem::path(sound_path).filename().string();

        std::lock_guard<std::mutex> lock(playback_mutex_);
        if (!playback_device_.has_value()) {
            return;
        }
        if (playback_active_) {
            return;
        }

        // Cooldown is keyed on sound file name (not label), and measured from
        // when the sound FINISHED playing — not when it started.  This prevents
        // the same wav from replaying immediately via a different label, and
        // avoids the bug where a long clip (e.g. hawk.wav = 16 s) exhausts the
        // 6-second cooldown while still playing, causing every subsequent
        // detection to be silently dropped until the clip ends.
        const auto ended_it = sound_last_ended_at_.find(sound_filename);
        if (ended_it != sound_last_ended_at_.end()) {
            const float since_ended =
                std::chrono::duration<float>(now - ended_it->second).count();
            if (since_ended < AUDIO_PLAYBACK_COOLDOWN_SEC) {
                return;
            }
        }

        if (playback_thread_.joinable()) {
            playback_thread_.join();
        }

        playback_active_ = true;
        playback_thread_ = std::thread([this, label, sound_path, sound_filename]() {
            std::cout << "[AUDIO] Playing " << sound_filename
                      << " for " << label << " via " << *playback_device_ << "\n";

            const std::string cmd =
                APLAY_BIN + " -q -D " + *playback_device_ + " -B 500000 \"" + sound_path + "\"";
            const int rc = std::system(cmd.c_str());
            if (rc != 0) {
                std::cerr << "[AUDIO] Playback failed for " << label
                          << " with code " << rc << "\n";
            }

            std::lock_guard<std::mutex> guard(playback_mutex_);
            sound_last_ended_at_[sound_filename] = std::chrono::steady_clock::now();
            playback_active_ = false;
        });
    }

    std::string sounds_dir_;
    std::unordered_map<std::string, std::string> label_to_sound_;
    std::optional<std::string> playback_device_;
    std::mutex playback_mutex_;
    std::thread playback_thread_;
    bool playback_active_ = false;
    // Cooldown tracking: sound file name → time the clip finished playing.
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> sound_last_ended_at_;
};

static bool run_inference_once(
    InferVStreams& infer_pipeline,
    const std::string& input_name,
    const std::string& output_name,
    size_t output_frame_size,
    const cv::Mat& frame,
    float conf_thresh,
    std::vector<Detection>& detections_out)
{
    std::vector<uint8_t> input_data(INPUT_W * INPUT_H * 3);
    std::vector<float> nms_output(output_frame_size / sizeof(float));

    preprocess_frame(frame, input_data);

    std::map<std::string, MemoryView> input_data_map = {
        {input_name, MemoryView(input_data.data(), input_data.size() * sizeof(uint8_t))}
    };
    std::map<std::string, MemoryView> output_data_map = {
        {output_name, MemoryView(nms_output.data(), output_frame_size)}
    };

    const auto status = infer_pipeline.infer(input_data_map, output_data_map, 1);
    if (status != HAILO_SUCCESS) {
        std::cerr << "[ERROR] InferVStreams inference failed: " << status << "\n";
        return false;
    }

    detections_out = decode_nms_output(
        nms_output,
        static_cast<float>(frame.cols),
        static_cast<float>(frame.rows),
        conf_thresh);
    detections_out = software_postprocess(detections_out);
    return true;
}

static bool run_single_image(
    const RuntimeConfig& cfg,
    InferVStreams& infer_pipeline,
    const std::string& input_name,
    const std::string& output_name,
    size_t output_frame_size)
{
    cv::Mat frame = cv::imread(*cfg.image_path);
    if (frame.empty()) {
        std::cerr << "[ERROR] Could not read image: " << *cfg.image_path << "\n";
        return false;
    }

    std::vector<Detection> detections;
    if (!run_inference_once(
            infer_pipeline,
            input_name,
            output_name,
            output_frame_size,
            frame,
            cfg.conf_thresh,
            detections)) {
        return false;
    }

    std::cout << "[INFO] Confidence threshold: " << cfg.conf_thresh << "\n";
    std::cout << "[INFO] Image detections: " << detections.size() << "\n";
    for (const auto& det : detections) {
        print_detection_line("DETECTED:", det);
    }

    cv::Mat rendered = frame.clone();
    draw_detections(rendered, detections);
    const std::string out_path = cfg.output_path.value_or(
        "/home/ecet400/hailo_detector/test/cpp_image_test_det.jpg");
    if (!cv::imwrite(out_path, rendered)) {
        std::cerr << "[ERROR] Failed to save output image: " << out_path << "\n";
        return false;
    }
    std::cout << "[INFO] Saved output image to " << out_path << "\n";
    return true;
}


// ============================================================
//  FRAME READER
//  Background thread continuously captures from libcamerasrc.
//  Inference thread picks up the latest frame — same pattern
//  as Kria FrameReader, pipeline string is the only difference.
// ============================================================

class FrameReader {
public:
    explicit FrameReader(const RuntimeConfig& cfg) : stop_(false), frame_id_(0) {
        const int output_width = cfg.preview_raw ? RAW_CAMERA_W : cfg.capture_width;
        const int output_height = cfg.preview_raw ? RAW_CAMERA_H : cfg.capture_height;

        auto try_gstreamer_pipeline = [&](int fps) -> bool {
            const std::string pipeline = build_libcamera_pipeline(cfg, output_width, output_height, fps);
            cap_.open(pipeline, cv::CAP_GSTREAMER);
            if (cap_.isOpened()) {
                camera_description_ = "libcamerasrc @ " + std::to_string(fps) + "fps";
                opened_width_ = output_width;
                opened_height_ = output_height;
                return true;
            }
            std::cerr << "[WARN] Could not open camera pipeline.\n";
            std::cerr << "Pipeline: " << pipeline << "\n";
            return false;
        };

        if (!try_gstreamer_pipeline(cfg.fps) && cfg.fps > 30) {
            std::cerr << "[INFO] Retrying camera at 30fps.\n";
            try_gstreamer_pipeline(30);
        }

        if (!cap_.isOpened()) {
            std::cerr << "[INFO] Retrying camera through /dev/video0 (V4L2 fallback).\n";
            cap_.open(0, cv::CAP_V4L2);
            if (cap_.isOpened()) {
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, output_width);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, output_height);
                cap_.set(cv::CAP_PROP_FPS, std::min(cfg.fps, 30));
                camera_description_ = "/dev/video0 via V4L2";
                opened_width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
                opened_height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
            }
        }

        if (!cap_.isOpened()) {
            std::cerr
                << "[ERROR] No camera source opened. Check that rpicam-hello --list-cameras "
                << "shows the camera and that /dev/video*, /dev/media*, and /dev/dma_heap "
                << "are visible to this process.\n";
            return;
        }

        std::cout << "[INFO] Camera opened (" << camera_description_
                  << " -> " << opened_width_ << "x" << opened_height_ << ").\n";
        if (cfg.preview_raw) {
            std::cout << "[INFO] Raw preview mode enabled. Inference still resizes internally to "
                      << INPUT_W << "x" << INPUT_H << ".\n";
        }

        worker_ = std::thread([this]() {
            cv::Mat raw;
            while (!stop_) {
                if (cap_.read(raw) && !raw.empty()) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    raw.copyTo(latest_frame_);
                    ++frame_id_;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
        });
    }

    ~FrameReader() {
        stop_ = true;
        if (worker_.joinable())
            worker_.join();
    }

    bool is_open() const { return cap_.isOpened(); }

    bool get_latest_frame(cv::Mat& out, uint64_t& id) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (latest_frame_.empty()) return false;
        out = latest_frame_.clone();
        id  = frame_id_;
        return true;
    }

private:
    static std::string build_libcamera_pipeline(
        const RuntimeConfig& cfg,
        int output_width,
        int output_height,
        int fps)
    {
        std::ostringstream props;
        props
            << "ae-enable=" << (cfg.disable_ae ? "false" : "true") << " "
            << "awb-enable=" << (cfg.disable_awb ? "false" : "true") << " "
            << "af-mode=" << cfg.af_mode << " "
            << "af-range=" << cfg.af_range << " "
            << "af-speed=" << cfg.af_speed;
        if (cfg.af_mode == "manual" && cfg.lens_position.has_value()) {
            props << " lens-position=" << *cfg.lens_position;
        }

        return
            "libcamerasrc " + props.str() + " ! "
            "video/x-raw,width=" + std::to_string(RAW_CAMERA_W) +
            ",height=" + std::to_string(RAW_CAMERA_H) +
            ",framerate=" + std::to_string(fps) + "/1,format=RGBx ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,width=" + std::to_string(output_width) +
            ",height=" + std::to_string(output_height) + ",format=BGR ! "
            "appsink max-buffers=1 drop=true sync=false";
    }

    cv::VideoCapture   cap_;
    std::thread        worker_;
    std::atomic<bool>  stop_;
    mutable std::mutex mtx_;
    cv::Mat            latest_frame_;
    uint64_t           frame_id_;
    std::string        camera_description_;
    int                opened_width_ = 0;
    int                opened_height_ = 0;
};

class LivePreviewServer {
public:
    LivePreviewServer(std::string host, int start_port, int max_port_tries, int max_encode_width)
        : host_(std::move(host))
        , port_(start_port)
        , max_port_tries_(max_port_tries)
        , max_encode_width_(max_encode_width) {}

    ~LivePreviewServer() {
        stop();
    }

    bool start() {
        if (!bind_server()) {
            return false;
        }

        running_ = true;
        accept_thread_ = std::thread([this]() { accept_loop(); });
        return true;
    }

    void stop() {
        const bool was_running = running_.exchange(false);
        if (server_fd_ >= 0) {
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            server_fd_ = -1;
        }
        if (accept_thread_.joinable()) {
            accept_thread_.join();
        }
        std::lock_guard<std::mutex> lock(client_threads_mutex_);
        for (auto& thread : client_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        client_threads_.clear();
        if (was_running) {
            frame_cv_.notify_all();
        }
    }

    int port() const {
        return port_;
    }

    void update_frame(const cv::Mat& frame_bgr) {
        cv::Mat preview_frame = frame_bgr;
        if (max_encode_width_ > 0 && frame_bgr.cols > max_encode_width_) {
            const int scaled_height = static_cast<int>(
                std::lround(static_cast<double>(frame_bgr.rows) * max_encode_width_ / frame_bgr.cols));
            cv::resize(
                frame_bgr,
                preview_frame,
                cv::Size(max_encode_width_, scaled_height),
                0.0,
                0.0,
                cv::INTER_AREA);
        }

        std::vector<uchar> encoded;
        if (!cv::imencode(".jpg", preview_frame, encoded, {cv::IMWRITE_JPEG_QUALITY, 75})) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_jpeg_.assign(encoded.begin(), encoded.end());
            ++latest_frame_index_;
        }
        frame_cv_.notify_all();
    }

private:
    bool bind_server() {
        for (int try_idx = 0; try_idx < max_port_tries_; ++try_idx) {
            const int candidate_port = port_ + try_idx;
            const int fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                continue;
            }

            int reuse = 1;
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

            sockaddr_in addr {};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(static_cast<uint16_t>(candidate_port));

            if (host_ == "0.0.0.0") {
                addr.sin_addr.s_addr = htonl(INADDR_ANY);
            } else if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
                std::cerr << "[ERROR] Invalid preview host: " << host_ << "\n";
                close(fd);
                return false;
            }

            if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
                listen(fd, 4) == 0) {
                server_fd_ = fd;
                port_ = candidate_port;
                return true;
            }

            close(fd);
        }

        std::cerr << "[ERROR] Could not bind preview server to ports "
                  << port_ << "-" << (port_ + max_port_tries_ - 1) << "\n";
        return false;
    }

    void accept_loop() {
        while (running_) {
            const int client_fd = accept(server_fd_, nullptr, nullptr);
            if (client_fd < 0) {
                if (!running_) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            std::lock_guard<std::mutex> lock(client_threads_mutex_);
            client_threads_.emplace_back([this, client_fd]() {
                handle_client(client_fd);
            });
        }
    }

    static bool send_all(int fd, const void* data, size_t len) {
        const char* ptr = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < len) {
            const ssize_t rc = send(fd, ptr + sent, len - sent, MSG_NOSIGNAL);
            if (rc <= 0) {
                return false;
            }
            sent += static_cast<size_t>(rc);
        }
        return true;
    }

    static bool send_string(int fd, const std::string& text) {
        return send_all(fd, text.data(), text.size());
    }

    void handle_client(int client_fd) {
        char request_buffer[2048] = {0};
        const ssize_t bytes_read = recv(client_fd, request_buffer, sizeof(request_buffer) - 1, 0);
        std::string request = bytes_read > 0 ? std::string(request_buffer, request_buffer + bytes_read) : "";

        std::string path = "/";
        const auto line_end = request.find("\r\n");
        const std::string first_line = request.substr(0, line_end);
        std::istringstream iss(first_line);
        std::string method;
        iss >> method >> path;

        if (path == "/" || path.empty()) {
            const std::string html =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html; charset=utf-8\r\n"
                "Cache-Control: no-store\r\n"
                "Connection: close\r\n\r\n"
                "<html><body style='margin:0;background:#111;color:#eee;font-family:sans-serif'>"
                "<div style='padding:12px'>Live preview: <code>/stream.mjpg</code></div>"
                "<img src='/stream.mjpg' style='display:block;max-width:100vw;height:auto'/>"
                "</body></html>";
            send_string(client_fd, html);
            close(client_fd);
            return;
        }

        if (path != "/stream.mjpg") {
            const std::string not_found =
                "HTTP/1.1 404 Not Found\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                "Connection: close\r\n\r\n"
                "Not found\n";
            send_string(client_fd, not_found);
            close(client_fd);
            return;
        }

        const std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Cache-Control: no-store\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        if (!send_string(client_fd, header)) {
            close(client_fd);
            return;
        }

        uint64_t last_sent_index = 0;
        while (running_) {
            std::vector<uint8_t> frame;
            uint64_t frame_index = 0;

            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                frame_cv_.wait_for(lock, std::chrono::milliseconds(50), [this, last_sent_index]() {
                    return !running_ || (!latest_jpeg_.empty() && latest_frame_index_ != last_sent_index);
                });
                if (!running_) {
                    break;
                }
                frame = latest_jpeg_;
                frame_index = latest_frame_index_;
            }

            if (frame.empty() || frame_index == last_sent_index) {
                continue;
            }
            last_sent_index = frame_index;

            const std::string part_header =
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: " + std::to_string(frame.size()) + "\r\n\r\n";
            if (!send_string(client_fd, part_header) ||
                !send_all(client_fd, frame.data(), frame.size()) ||
                !send_string(client_fd, "\r\n")) {
                break;
            }
        }

        close(client_fd);
    }

    std::string host_;
    int port_;
    int max_port_tries_;
    int max_encode_width_;
    int server_fd_ = -1;
    std::atomic<bool> running_ {false};
    std::thread accept_thread_;

    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::vector<uint8_t> latest_jpeg_;
    uint64_t latest_frame_index_ = 0;

    std::mutex client_threads_mutex_;
    std::vector<std::thread> client_threads_;
};


// ============================================================
//  MAIN
// ============================================================

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    // Important: the program is designed to work with no arguments.
    // parse_args only applies optional overrides on top of the defaults
    // defined in the EASY-TO-EDIT SETTINGS section near the top.
    const RuntimeConfig runtime_config = parse_args(argc, argv);

    // --- Step 1: Load HEF and create Hailo network ---
    std::cout << "[1/3] Loading Hailo model: " << HEF_PATH << "\n";

    auto hef_result = Hef::create(HEF_PATH);
    if (!hef_result) {
        std::cerr << "[ERROR] Failed to load HEF: " << hef_result.status() << "\n";
        return 1;
    }
    auto hef = hef_result.release();

    auto vdevice_result = VDevice::create();
    if (!vdevice_result) {
        std::cerr << "[ERROR] Failed to create VDevice: " << vdevice_result.status() << "\n";
        return 1;
    }
    auto vdevice = vdevice_result.release();

    auto network_groups_result = vdevice->configure(hef);
    if (!network_groups_result) {
        std::cerr << "[ERROR] Failed to configure network: " << network_groups_result.status() << "\n";
        return 1;
    }
    auto& network_groups = network_groups_result.value();

    auto& network_group = network_groups[0];

    auto params_result = network_group->make_input_vstream_params(
        true, HAILO_FORMAT_TYPE_UINT8,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!params_result) {
        std::cerr << "[ERROR] Failed to make input params\n";
        return 1;
    }

    auto out_params_result = network_group->make_output_vstream_params(
        false, HAILO_FORMAT_TYPE_FLOAT32,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!out_params_result) {
        std::cerr << "[ERROR] Failed to make output params\n";
        return 1;
    }

    auto infer_pipeline_result = InferVStreams::create(*network_group, params_result.value(), out_params_result.value());
    if (!infer_pipeline_result) {
        std::cerr << "[ERROR] Failed to create InferVStreams pipeline: "
                  << infer_pipeline_result.status() << "\n";
        return 1;
    }
    auto infer_pipeline = infer_pipeline_result.release();

    std::unique_ptr<ActivatedNetworkGroup> activated_network_group;
    if (!network_group->is_scheduled()) {
        auto activation_result = network_group->activate();
        if (activation_result) {
            activated_network_group = activation_result.release();
        } else {
            std::cerr << "[ERROR] Failed to activate network group: "
                      << activation_result.status() << "\n";
            return 1;
        }
    }

    const std::string input_name = params_result.value().begin()->first;
    const std::string output_name = out_params_result.value().begin()->first;
    auto output_vstream_ref_result = infer_pipeline.get_output_by_name(output_name);
    if (!output_vstream_ref_result) {
        std::cerr << "[ERROR] Failed to access output vstream: "
                  << output_vstream_ref_result.status() << "\n";
        return 1;
    }
    const size_t output_frame_size = output_vstream_ref_result.value().get().get_frame_size();

    const auto nms_threshold_status = infer_pipeline.set_nms_score_threshold(runtime_config.conf_thresh);
    if (nms_threshold_status != HAILO_SUCCESS && nms_threshold_status != HAILO_INVALID_OPERATION) {
        std::cerr << "[WARN] Failed to set NMS score threshold: " << nms_threshold_status << "\n";
    }

    std::cout << "[INFO] Hailo-8L ready. "
              << infer_pipeline.get_input_vstreams().size()  << " input(s), "
              << infer_pipeline.get_output_vstreams().size() << " output(s).\n";
    for (const auto& out_vs_ref : infer_pipeline.get_output_vstreams()) {
        const auto& out_vs = out_vs_ref.get();
        auto info = out_vs.get_info();
        std::cout << "[INFO] Output stream: " << info.name
                  << " (" << out_vs.get_frame_size() / sizeof(float) << " float values)\n";
    }

    if (runtime_config.image_path.has_value()) {
        const bool ok = run_single_image(
            runtime_config,
            infer_pipeline,
            input_name,
            output_name,
            output_frame_size);
        return ok ? 0 : 1;
    }

    AudioDeterrent audio_deterrent(SOUNDS_DIR);
    Ads1115Tmp36Sensor temperature_sensor;

    // --- Step 2: GPIO init ---
    GpioTrigger gpio;
    bool pir_gate_enabled = false;
    if (!gpio.init()) {
        std::cerr << "[WARN] GPIO init failed. Running without PIR gate.\n";
    } else {
        gpio.start();
        gpio.set_waiting_indicator(true);
        pir_gate_enabled = true;
        std::cout << "[INFO] GPIO ready. Waiting for PIR motion...\n";
    }

    // --- Step 3: Open camera ---
    std::cout << "[3/4] Opening camera...\n";
    FrameReader reader(runtime_config);

    if (!reader.is_open()) {
        std::cerr << "[ERROR] Camera could not be opened.\n";
        return 1;
    }

    std::unique_ptr<LivePreviewServer> preview_server;
    if (runtime_config.serve) {
        const int preview_max_encode_width = 0;
        preview_server = std::make_unique<LivePreviewServer>(
            runtime_config.serve_host,
            runtime_config.serve_port,
            runtime_config.serve_port_tries,
            preview_max_encode_width);
        if (!preview_server->start()) {
            return 1;
        }

        const std::string preview_host =
            (runtime_config.serve_host == "0.0.0.0") ? "127.0.0.1" : runtime_config.serve_host;
        std::cout << "[INFO] Live preview: http://" << preview_host << ":"
                  << preview_server->port() << " (or replace host with the Pi IP)\n";
    }

    // Wait for first frame
    std::cout << "[INFO] Waiting for first frame...\n";
    for (int i = 0; i < 50 && g_running; i++) {
        cv::Mat tmp; uint64_t id = 0;
        if (reader.get_latest_frame(tmp, id)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // --- Step 4: Inference loop ---
    std::cout << "[4/4] Inference running. Press Ctrl+C to stop.\n\n";
    std::cout << "[INFO] Confidence threshold: " << runtime_config.conf_thresh << "\n";

    int      frame_count   = 0;
    uint64_t last_processed_frame_id = 0;
    float    last_fps      = 0.0f;
    DetectionEventLogger detection_logger;
    DetectionFrameSaver detection_frame_saver(runtime_config);
    PirGateFsm pir_fsm;
    auto     start_time    = std::chrono::steady_clock::now();
    float    active_elapsed_sec = 0.0f;
    std::optional<std::chrono::steady_clock::time_point> active_session_started_at;
    std::deque<std::chrono::steady_clock::time_point> recent_processed_frame_times;
    auto     last_preview_push = std::chrono::steady_clock::time_point::min();
    const auto raw_preview_interval = std::chrono::milliseconds(66);
    static constexpr size_t FPS_WINDOW_SIZE = 30;

    while (g_running) {
        cv::Mat  frame;
        uint64_t current_frame_id = 0;

        if (!reader.get_latest_frame(frame, current_frame_id)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        if (current_frame_id == last_processed_frame_id) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        last_processed_frame_id = current_frame_id;

        if (pir_gate_enabled) {
            pir_fsm.update_before_inference(gpio.pir_high());
        }

        if (pir_gate_enabled && !pir_fsm.inference_enabled()) {
            const auto now = std::chrono::steady_clock::now();
            if (active_session_started_at.has_value()) {
                active_elapsed_sec += std::chrono::duration<float>(
                    now - *active_session_started_at).count();
                active_session_started_at.reset();
                recent_processed_frame_times.clear();
                last_fps = 0.0f;
            }

            gpio.set_waiting_indicator(pir_fsm.waiting_indicator());
            if (preview_server) {
                cv::Mat waiting_frame = frame.clone();
                const std::optional<float> temperature_f = temperature_sensor.latest_fahrenheit();
                draw_summary(waiting_frame, frame_count, {}, last_fps, temperature_f);
                draw_pir_status(waiting_frame, true);

                if (!runtime_config.preview_raw ||
                    last_preview_push == std::chrono::steady_clock::time_point::min() ||
                    (now - last_preview_push) >= raw_preview_interval) {
                    preview_server->update_frame(waiting_frame);
                    last_preview_push = now;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        if (pir_gate_enabled) {
            gpio.set_waiting_indicator(pir_fsm.waiting_indicator());
        }
        if (!active_session_started_at.has_value()) {
            active_session_started_at = std::chrono::steady_clock::now();
        }

        auto t1 = std::chrono::steady_clock::now();
        std::vector<Detection> detections;
        const bool infer_ok = run_inference_once(
            infer_pipeline,
            input_name,
            output_name,
            output_frame_size,
            frame,
            runtime_config.conf_thresh,
            detections);
        auto t2 = std::chrono::steady_clock::now();
        auto t3 = std::chrono::steady_clock::now();
        if (!infer_ok) {
            g_running = false;
            break;
        }
        auto t4 = std::chrono::steady_clock::now();

        // Print only new or periodically refreshed detection events.
        detection_logger.log(detections);
        if (pir_gate_enabled) {
            pir_fsm.update_after_inference(!detections.empty());
        }

        audio_deterrent.maybe_play_for_detections(detections);
        if (pir_gate_enabled) {
            const auto label = first_detected_label(detections);
            if (label.has_value()) {
                gpio.maybe_play_ultrasonic_for_label(*label);
            }
        }

        cv::Mat rendered = frame.clone();
        draw_detections(rendered, detections);

        // -- FPS report every 30 frames --
        frame_count++;
        const auto frame_finished_at = std::chrono::steady_clock::now();
        recent_processed_frame_times.push_back(frame_finished_at);
        while (recent_processed_frame_times.size() > FPS_WINDOW_SIZE) {
            recent_processed_frame_times.pop_front();
        }
        if (recent_processed_frame_times.size() >= 2) {
            const float rolling_elapsed_sec = std::chrono::duration<float>(
                recent_processed_frame_times.back() - recent_processed_frame_times.front()).count();
            if (rolling_elapsed_sec > 0.0f) {
                last_fps = static_cast<float>(recent_processed_frame_times.size() - 1) /
                           rolling_elapsed_sec;
            }
        }

        const std::optional<float> temperature_f = temperature_sensor.latest_fahrenheit();
        draw_summary(rendered, frame_count, detections, last_fps, temperature_f);
        if (pir_gate_enabled) {
            draw_pir_status(rendered, false);
        }

        detection_frame_saver.maybe_save(
            rendered,
            detections,
            current_frame_id,
            pir_gate_enabled);

        if (preview_server) {
            const auto now = std::chrono::steady_clock::now();
            if (!runtime_config.preview_raw ||
                last_preview_push == std::chrono::steady_clock::time_point::min() ||
                (now - last_preview_push) >= raw_preview_interval) {
                preview_server->update_frame(rendered);
                last_preview_push = now;
            }
        }

        if (frame_count % 30 == 0) {
            std::printf("--- %.1f FPS | pre=%.1f ms  hailo=%.1f ms  post=%.1f ms ---\n",
                        last_fps,
                        std::chrono::duration<float, std::milli>(t2 - t1).count(),
                        std::chrono::duration<float, std::milli>(t3 - t2).count(),
                        std::chrono::duration<float, std::milli>(t4 - t3).count());
        }
    }

    if (active_session_started_at.has_value()) {
        active_elapsed_sec += std::chrono::duration<float>(
            std::chrono::steady_clock::now() - *active_session_started_at).count();
    }

    const float total_elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time).count();
    const float average_active_fps =
        (active_elapsed_sec > 0.0f) ? (static_cast<float>(frame_count) / active_elapsed_sec) : 0.0f;

    std::printf("\nStopped. %d frames processed, active average %.1f FPS over %.1fs active time (%.1fs total uptime).\n",
                frame_count,
                average_active_fps,
                active_elapsed_sec,
                total_elapsed);

    return 0;
}
