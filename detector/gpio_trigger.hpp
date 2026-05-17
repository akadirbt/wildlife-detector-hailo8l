#pragma once
/*
 * ============================================================
 *  gpio_trigger.hpp — PIR Motion Sensing + Deterrent Control
 *  Raspberry Pi 5 + Hailo Wildlife Detector
 *  libgpiod v2 API
 *
 *  Overview:
 *    This module keeps the proven PIR gating logic and now also adds:
 *      - red/green status LEDs
 *      - a lightweight ultrasonic piezo burst output
 *
 *  Hardware connections (BCM pin numbers):
 *    PIR Signal       -> GPIO23  (input, active HIGH when motion detected)
 *    Ultrasonic Gate  -> GPIO13  (output to logic-level MOSFET gate)
 *    Green LED        -> GPIO17  (ON while PIR window is active)
 *    Red LED          -> GPIO22  (ON while waiting for PIR motion)
 *    AMP SD pin       -> GPIO26  (output, HIGH = amplifier enabled)
 *
 *  Notes:
 *    - The main detector still uses AudioDeterrent for normal speaker
 *      playback; GPIO amp toggling remains optional.
 * ============================================================
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <gpiod.h>


// ============================================================
//  CONFIG
// ============================================================

static constexpr unsigned int PIR_SIGNAL_PIN      = 23;  // BCM pin for PIR signal
static constexpr unsigned int ULTRASONIC_PIN      = 13;  // BCM pin for piezo MOSFET gate
static constexpr unsigned int GREEN_LED_PIN       = 17;  // BCM pin for green LED
static constexpr unsigned int RED_LED_PIN         = 22;  // BCM pin for red LED
static constexpr unsigned int AMP_ENABLE_PIN      = 26;  // BCM pin for MAX98357A SD
static constexpr bool LEDS_ACTIVE_LOW             = false;

// How long to suppress some repeated deterrent actions (seconds).
static constexpr float COOLDOWN_SEC = 10.0f;

// Poll PIR often enough that software does not add noticeable lag.
static constexpr int PIR_POLL_INTERVAL_MS = 20;

// How long to keep the amplifier enabled after triggering (seconds).
static constexpr float DETERRENT_DURATION = 3.0f;
static constexpr int ULTRASONIC_TOTAL_MS =
    static_cast<int>(DETERRENT_DURATION * 1000.0f);

// Keep GPIO26 asserted so the MAX98357A SD/EN pin does not mute the amp
// while ALSA + AudioDeterrent plays animal-specific sounds.
static constexpr bool ENABLE_GPIO_AMP_CONTROL = true;

// Full path to the deterrent sound file on the Raspberry Pi filesystem.
static const std::string DETERRENT_SOUND = "/home/ecet400/sounds/deterrent.wav";

// GPIO chip device node. This setup exposes the 40-pin header on gpiochip0.
static const char* DEFAULT_GPIO_CHIP_PATH = "/dev/gpiochip0";
static const char* DEFAULT_PWM_CHIP_PATH = "/sys/class/pwm/pwmchip0";
static constexpr unsigned int DEFAULT_ULTRASONIC_PWM_CHANNEL = 1;  // GPIO13 = PWM1

struct UltrasonicProfile {
    int frequency_hz;
    int burst_on_ms;
    int burst_off_ms;
    int total_ms;
};

static inline std::optional<UltrasonicProfile> ultrasonic_profile_for_label(
    const std::string& label)
{
    if (label == "bear" || label == "coyote" || label == "deer") {
        return UltrasonicProfile {17000, ULTRASONIC_TOTAL_MS, 0, ULTRASONIC_TOTAL_MS};
    }
    if (label == "fox" || label == "possum" || label == "raccoon" || label == "skunk") {
        return UltrasonicProfile {20000, ULTRASONIC_TOTAL_MS, 0, ULTRASONIC_TOTAL_MS};
    }
    if (label == "squirrel") {
        return UltrasonicProfile {23000, ULTRASONIC_TOTAL_MS, 0, ULTRASONIC_TOTAL_MS};
    }
    if (label == "turkey") {
        return UltrasonicProfile {30000, ULTRASONIC_TOTAL_MS, 0, ULTRASONIC_TOTAL_MS};
    }
    return std::nullopt;
}


class GpioTrigger {
public:
    GpioTrigger()
        : chip_(nullptr)
        , pir_request_(nullptr)
        , green_led_request_(nullptr)
        , red_led_request_(nullptr)
        , amp_request_(nullptr)
        , pir_active_(false)
        , stop_(false)
        , last_pir_value_(GPIOD_LINE_VALUE_ERROR)
        , ultrasonic_pwm_channel_(DEFAULT_ULTRASONIC_PWM_CHANNEL)
        , ultrasonic_pwm_ready_(false)
        , ultrasonic_pwm_exported_by_app_(false)
        , ultrasonic_active_(false)
        , deterrent_active_(false)
    {}

    ~GpioTrigger() {
        stop_ = true;
        if (poll_thread_.joinable()) {
            poll_thread_.join();
        }
        if (ultrasonic_thread_.joinable()) {
            ultrasonic_thread_.join();
        }
        if (deterrent_thread_.joinable()) {
            deterrent_thread_.join();
        }
        cleanup();
    }

    bool init() {
        const std::string chip_path = gpio_chip_path();
        chip_ = gpiod_chip_open(chip_path.c_str());
        if (!chip_) {
            std::cerr << "[GPIO] Failed to open " << chip_path
                      << ". Check libgpiod installation and chip path";
            if (chip_path == DEFAULT_GPIO_CHIP_PATH) {
                std::cerr << " (or set WILDLIFE_GPIO_CHIP=/dev/gpiochipN)";
            }
            std::cerr << ".\n";
            return false;
        }

        gpiod_request_config* pir_cfg      = gpiod_request_config_new();
        gpiod_line_config*    pir_line_cfg = gpiod_line_config_new();
        gpiod_line_settings*  pir_settings = gpiod_line_settings_new();

        gpiod_request_config_set_consumer(pir_cfg, "wildlife_pir");
        gpiod_line_settings_set_direction(pir_settings, GPIOD_LINE_DIRECTION_INPUT);
        gpiod_line_settings_set_bias(pir_settings, GPIOD_LINE_BIAS_PULL_DOWN);
        gpiod_line_config_add_line_settings(pir_line_cfg, &PIR_SIGNAL_PIN, 1, pir_settings);

        pir_request_ = gpiod_chip_request_lines(chip_, pir_cfg, pir_line_cfg);

        gpiod_line_settings_free(pir_settings);
        gpiod_line_config_free(pir_line_cfg);
        gpiod_request_config_free(pir_cfg);

        if (!pir_request_) {
            std::cerr << "[GPIO] Failed to request PIR signal line (GPIO"
                      << PIR_SIGNAL_PIN << ")\n";
            return false;
        }

        ultrasonic_pwm_ready_ = init_ultrasonic_pwm();
        if (!ultrasonic_pwm_ready_) {
            std::cerr << "[GPIO] Hardware PWM on GPIO" << ULTRASONIC_PIN
                      << " is unavailable. Ultrasonic output disabled.\n";
        }

        green_led_request_ = request_output_line(
            GREEN_LED_PIN,
            "wildlife_green_led",
            led_value(false));
        if (!green_led_request_) {
            std::cerr << "[GPIO] Failed to request green LED line (GPIO"
                      << GREEN_LED_PIN << "). Continuing without LED output.\n";
        }

        red_led_request_ = request_output_line(
            RED_LED_PIN,
            "wildlife_red_led",
            led_value(true));
        if (!red_led_request_) {
            std::cerr << "[GPIO] Failed to request red LED line (GPIO"
                      << RED_LED_PIN << "). Continuing without LED output.\n";
        }

        if (ENABLE_GPIO_AMP_CONTROL) {
            gpiod_request_config* amp_cfg      = gpiod_request_config_new();
            gpiod_line_config*    amp_line_cfg = gpiod_line_config_new();
            gpiod_line_settings*  amp_settings = gpiod_line_settings_new();

            gpiod_request_config_set_consumer(amp_cfg, "wildlife_amp");
            gpiod_line_settings_set_direction(amp_settings, GPIOD_LINE_DIRECTION_OUTPUT);
            gpiod_line_settings_set_output_value(
                amp_settings,
                GPIOD_LINE_VALUE_ACTIVE);
            gpiod_line_config_add_line_settings(
                amp_line_cfg,
                &AMP_ENABLE_PIN,
                1,
                amp_settings);

            amp_request_ = gpiod_chip_request_lines(chip_, amp_cfg, amp_line_cfg);

            gpiod_line_settings_free(amp_settings);
            gpiod_line_config_free(amp_line_cfg);
            gpiod_request_config_free(amp_cfg);

            if (!amp_request_) {
                std::cerr << "[GPIO] Failed to request AMP enable line (GPIO"
                          << AMP_ENABLE_PIN << "). Continuing with PIR only.\n";
            }
        } else {
            std::cout << "[GPIO] AMP GPIO control disabled; leaving GPIO"
                      << AMP_ENABLE_PIN << " untouched.\n";
        }

        std::cout << "[GPIO] Initialized. PIR=GPIO" << PIR_SIGNAL_PIN;
        std::cout << " CHIP=" << chip_path;
        if (amp_request_) {
            std::cout << " AMP=GPIO" << AMP_ENABLE_PIN << "(enabled)";
        } else {
            std::cout << " AMP=disabled";
        }
        std::cout << "\n";
        return true;
    }

    void start() {
        poll_thread_ = std::thread([this]() {
            while (!stop_) {
                const gpiod_line_value value =
                    gpiod_line_request_get_value(pir_request_, PIR_SIGNAL_PIN);

                if (value == GPIOD_LINE_VALUE_ERROR) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(PIR_POLL_INTERVAL_MS));
                    continue;
                }

                if (value != last_pir_value_) {
                    std::cout << "[GPIO] PIR GPIO" << PIR_SIGNAL_PIN
                              << " = "
                              << (value == GPIOD_LINE_VALUE_ACTIVE ? "HIGH" : "LOW")
                              << "\n";
                    last_pir_value_ = value;
                }

                const bool is_active = value == GPIOD_LINE_VALUE_ACTIVE;
                pir_active_.store(is_active);

                std::this_thread::sleep_for(std::chrono::milliseconds(PIR_POLL_INTERVAL_MS));
            }
        });
    }

    void set_waiting_indicator(bool waiting) {
        std::lock_guard<std::mutex> lock(indicator_mutex_);
        if (last_waiting_state_.has_value() && *last_waiting_state_ == waiting) {
            return;
        }

        if (green_led_request_) {
            gpiod_line_request_set_value(
                green_led_request_,
                GREEN_LED_PIN,
                led_value(!waiting));
        }
        if (red_led_request_) {
            gpiod_line_request_set_value(
                red_led_request_,
                RED_LED_PIN,
                led_value(waiting));
        }
        last_waiting_state_ = waiting;
    }

    bool maybe_play_ultrasonic_for_label(const std::string& label) {
        const auto profile = ultrasonic_profile_for_label(label);
        if (!profile.has_value() || !ultrasonic_pwm_ready_) {
            return false;
        }

        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(ultrasonic_mutex_);
        if (ultrasonic_active_.load()) {
            return false;
        }
        if (last_ultrasonic_time_.has_value() &&
            std::chrono::duration<float>(now - *last_ultrasonic_time_).count() < COOLDOWN_SEC) {
            return false;
        }

        if (ultrasonic_thread_.joinable()) {
            ultrasonic_thread_.join();
        }

        ultrasonic_active_.store(true);
        last_ultrasonic_time_ = now;
        ultrasonic_thread_ = std::thread([this, label, profile = *profile]() {
            if (!configure_ultrasonic_pwm(profile.frequency_hz)) {
                ultrasonic_active_.store(false);
                return;
            }

            std::cout << "[ULTRASONIC] " << label
                      << " -> " << profile.frequency_hz
                      << " Hz PWM burst on GPIO" << ULTRASONIC_PIN << "\n";

            const auto finished_at =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(profile.total_ms);
            if (profile.burst_off_ms <= 0) {
                set_ultrasonic_enabled(true);
                std::this_thread::sleep_until(finished_at);
                set_ultrasonic_enabled(false);
                ultrasonic_active_.store(false);
                return;
            }

            while (!stop_ && std::chrono::steady_clock::now() < finished_at) {
                play_ultrasonic_burst(profile);
                if (stop_ || std::chrono::steady_clock::now() >= finished_at) {
                    break;
                }
                set_ultrasonic_enabled(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(profile.burst_off_ms));
            }

            set_ultrasonic_enabled(false);
            ultrasonic_active_.store(false);
        });

        return true;
    }

    bool activate_deterrent() {
        const auto now = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(deterrent_mutex_);

            if (deterrent_active_.load()) {
                return false;
            }

            if (last_deterrent_time_.has_value() &&
                std::chrono::duration<float>(now - *last_deterrent_time_).count() < COOLDOWN_SEC) {
                return false;
            }

            if (deterrent_thread_.joinable()) {
                deterrent_thread_.join();
            }

            deterrent_active_.store(true);
            last_deterrent_time_ = now;
            deterrent_thread_ = std::thread([this]() {
                std::cout << "[DETERRENT] Activating speaker...\n";

                if (amp_request_) {
                    gpiod_line_request_set_value(
                        amp_request_,
                        AMP_ENABLE_PIN,
                        GPIOD_LINE_VALUE_ACTIVE);
                }

                const std::string cmd = "aplay -q -D plughw:MAX98357A,0 -B 500000 " + DETERRENT_SOUND;
                const int rc = std::system(cmd.c_str());
                if (rc != 0) {
                    std::cerr << "[DETERRENT] aplay failed with code " << rc << "\n";
                }

                if (amp_request_) {
                    gpiod_line_request_set_value(
                        amp_request_,
                        AMP_ENABLE_PIN,
                        GPIOD_LINE_VALUE_INACTIVE);
                }

                deterrent_active_.store(false);
                std::cout << "[DETERRENT] Done.\n";
            });
        }

        return true;
    }

    bool pir_high() const {
        return pir_active_.load();
    }

private:
    static std::string gpio_chip_path() {
        if (const char* env_path = std::getenv("WILDLIFE_GPIO_CHIP")) {
            if (env_path[0] != '\0') {
                return env_path;
            }
        }
        return DEFAULT_GPIO_CHIP_PATH;
    }

    static gpiod_line_value led_value(bool on) {
        if (LEDS_ACTIVE_LOW) {
            return on ? GPIOD_LINE_VALUE_INACTIVE : GPIOD_LINE_VALUE_ACTIVE;
        }
        return on ? GPIOD_LINE_VALUE_ACTIVE : GPIOD_LINE_VALUE_INACTIVE;
    }

    static bool write_text_file(const std::filesystem::path& path, const std::string& value) {
        errno = 0;
        std::ofstream stream(path);
        if (!stream.is_open()) {
            std::cerr << "[ULTRASONIC] Failed to open " << path
                      << ": " << std::strerror(errno) << "\n";
            return false;
        }
        stream << value;
        if (!stream.good()) {
            std::cerr << "[ULTRASONIC] Failed to write " << path
                      << ": " << std::strerror(errno) << "\n";
        }
        return stream.good();
    }

    static bool wait_for_path(const std::filesystem::path& path, int timeout_ms) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (std::filesystem::exists(path)) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return std::filesystem::exists(path);
    }

    static std::optional<unsigned int> parse_uint_env(const char* name) {
        const char* env_value = std::getenv(name);
        if (!env_value || env_value[0] == '\0') {
            return std::nullopt;
        }
        try {
            return static_cast<unsigned int>(std::stoul(env_value));
        } catch (const std::exception&) {
            std::cerr << "[ULTRASONIC] Ignoring invalid " << name
                      << "=" << env_value << "\n";
            return std::nullopt;
        }
    }

    static std::optional<unsigned int> pwm_npwm(const std::filesystem::path& chip_path) {
        std::ifstream stream(chip_path / "npwm");
        unsigned int value = 0;
        if (stream >> value) {
            return value;
        }
        return std::nullopt;
    }

    static std::filesystem::path discover_pwm_chip_path(unsigned int channel) {
        if (const char* env_path = std::getenv("WILDLIFE_PWM_CHIP")) {
            if (env_path[0] != '\0') {
                return env_path;
            }
        }

        const std::filesystem::path default_path(DEFAULT_PWM_CHIP_PATH);
        if (std::filesystem::exists(default_path)) {
            return default_path;
        }

        const std::filesystem::path pwm_root("/sys/class/pwm");
        if (!std::filesystem::exists(pwm_root)) {
            return default_path;
        }

        for (const auto& entry : std::filesystem::directory_iterator(pwm_root)) {
            if (!entry.is_directory()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind("pwmchip", 0) != 0) {
                continue;
            }
            const auto npwm = pwm_npwm(entry.path());
            if (!npwm.has_value() || channel < *npwm) {
                return entry.path();
            }
        }

        return default_path;
    }

    std::filesystem::path ultrasonic_pwm_channel_path() const {
        return ultrasonic_pwm_chip_path_ /
               ("pwm" + std::to_string(ultrasonic_pwm_channel_));
    }

    bool init_ultrasonic_pwm() {
        if (const auto env_channel = parse_uint_env("WILDLIFE_PWM_CHANNEL")) {
            ultrasonic_pwm_channel_ = *env_channel;
        }
        ultrasonic_pwm_chip_path_ = discover_pwm_chip_path(ultrasonic_pwm_channel_);

        const std::filesystem::path chip_path = ultrasonic_pwm_chip_path_;
        if (!std::filesystem::exists(chip_path)) {
            std::cerr << "[ULTRASONIC] PWM chip path missing: " << chip_path
                      << " (enable a PWM overlay, or set WILDLIFE_PWM_CHIP)\n";
            return false;
        }

        const std::filesystem::path channel_path = ultrasonic_pwm_channel_path();
        if (!std::filesystem::exists(channel_path)) {
            const std::filesystem::path export_path = chip_path / "export";
            if (!write_text_file(export_path, std::to_string(ultrasonic_pwm_channel_))) {
                return false;
            }
            ultrasonic_pwm_exported_by_app_ = true;
            if (!wait_for_path(channel_path, 500)) {
                std::cerr << "[ULTRASONIC] PWM channel did not appear after export: "
                          << channel_path << "\n";
                return false;
            }
        }

        // Ensure the channel starts disabled.
        set_ultrasonic_enabled(false);
        return true;
    }

    bool configure_ultrasonic_pwm(int frequency_hz) {
        const auto channel_path = ultrasonic_pwm_channel_path();
        if (!std::filesystem::exists(channel_path)) {
            std::cerr << "[ULTRASONIC] PWM channel path missing: " << channel_path << "\n";
            return false;
        }

        const int period_ns = std::max(1, static_cast<int>(1000000000LL / frequency_hz));
        const int duty_ns = period_ns / 2;

        set_ultrasonic_enabled(false);
        if (!write_text_file(channel_path / "period", std::to_string(period_ns))) {
            return false;
        }
        if (!write_text_file(channel_path / "duty_cycle", std::to_string(duty_ns))) {
            return false;
        }
        return true;
    }

    bool set_ultrasonic_enabled(bool enabled) {
        const auto channel_path = ultrasonic_pwm_channel_path();
        if (!std::filesystem::exists(channel_path)) {
            return false;
        }

        return write_text_file(channel_path / "enable", enabled ? "1" : "0");
    }

    gpiod_line_request* request_output_line(
        unsigned int pin,
        const char* consumer,
        gpiod_line_value initial_value)
    {
        gpiod_request_config* cfg = gpiod_request_config_new();
        gpiod_line_config* line_cfg = gpiod_line_config_new();
        gpiod_line_settings* settings = gpiod_line_settings_new();

        if (!cfg || !line_cfg || !settings) {
            gpiod_request_config_free(cfg);
            gpiod_line_config_free(line_cfg);
            gpiod_line_settings_free(settings);
            return nullptr;
        }

        gpiod_request_config_set_consumer(cfg, consumer);
        gpiod_line_settings_set_direction(settings, GPIOD_LINE_DIRECTION_OUTPUT);
        gpiod_line_settings_set_output_value(settings, initial_value);
        gpiod_line_config_add_line_settings(line_cfg, &pin, 1, settings);

        gpiod_line_request* request = gpiod_chip_request_lines(chip_, cfg, line_cfg);

        gpiod_line_settings_free(settings);
        gpiod_line_config_free(line_cfg);
        gpiod_request_config_free(cfg);
        return request;
    }

    void play_ultrasonic_burst(const UltrasonicProfile& profile) {
        set_ultrasonic_enabled(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(profile.burst_on_ms));
        set_ultrasonic_enabled(false);
    }

    void cleanup() {
        if (red_led_request_) {
            gpiod_line_request_set_value(
                red_led_request_,
                RED_LED_PIN,
                led_value(false));
            gpiod_line_request_release(red_led_request_);
            red_led_request_ = nullptr;
        }
        if (green_led_request_) {
            gpiod_line_request_set_value(
                green_led_request_,
                GREEN_LED_PIN,
                led_value(false));
            gpiod_line_request_release(green_led_request_);
            green_led_request_ = nullptr;
        }
        if (ultrasonic_pwm_ready_) {
            set_ultrasonic_enabled(false);
            if (ultrasonic_pwm_exported_by_app_) {
                const std::filesystem::path unexport_path =
                    ultrasonic_pwm_chip_path_ / "unexport";
                write_text_file(unexport_path, std::to_string(ultrasonic_pwm_channel_));
            }
            ultrasonic_pwm_ready_ = false;
            ultrasonic_pwm_exported_by_app_ = false;
        }
        if (amp_request_) {
            gpiod_line_request_set_value(
                amp_request_,
                AMP_ENABLE_PIN,
                GPIOD_LINE_VALUE_INACTIVE);
            gpiod_line_request_release(amp_request_);
            amp_request_ = nullptr;
        }
        if (pir_request_) {
            gpiod_line_request_release(pir_request_);
            pir_request_ = nullptr;
        }
        if (chip_) {
            gpiod_chip_close(chip_);
            chip_ = nullptr;
        }
    }

    gpiod_chip*         chip_;
    gpiod_line_request* pir_request_;
    gpiod_line_request* green_led_request_;
    gpiod_line_request* red_led_request_;
    gpiod_line_request* amp_request_;

    std::atomic<bool> pir_active_;
    std::atomic<bool> stop_;
    std::atomic<gpiod_line_value> last_pir_value_;
    std::filesystem::path ultrasonic_pwm_chip_path_;
    unsigned int ultrasonic_pwm_channel_;
    bool ultrasonic_pwm_ready_;
    bool ultrasonic_pwm_exported_by_app_;
    std::atomic<bool> ultrasonic_active_;
    std::atomic<bool> deterrent_active_;

    std::thread poll_thread_;
    std::thread ultrasonic_thread_;
    std::thread deterrent_thread_;
    std::mutex  indicator_mutex_;
    std::mutex  ultrasonic_mutex_;
    std::mutex  deterrent_mutex_;

    std::optional<bool> last_waiting_state_;
    std::optional<std::chrono::steady_clock::time_point> last_ultrasonic_time_;
    std::optional<std::chrono::steady_clock::time_point> last_deterrent_time_;
};
