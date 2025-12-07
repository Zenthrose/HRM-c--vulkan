#include "hrm_gui.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif
#include <signal.h>
#include <cstring>
#include <cstdlib>

HRMGUI::HRMGUI(std::shared_ptr<ResourceAwareHRM> hrm_system)
    : hrm_system_(hrm_system),
      current_page_(GUIPage::MAIN_MENU),
      theme_(GUITheme::DARK),
      window_title_("HRM - Hierarchical Reasoning Module"),
      status_bar_text_("Ready"),
      awaiting_response_(false),
      selected_menu_item_(0),
      redraw_needed_(true),
      last_update_(std::chrono::system_clock::now()) {
    initialize_menus();
    setup_terminal();
}

HRMGUI::~HRMGUI() {
    restore_terminal();
}

void HRMGUI::run() {
    clear_screen();
    redraw_needed_ = true;

    while (true) {
        if (redraw_needed_) {
            draw_header();
            switch (current_page_) {
                case GUIPage::MAIN_MENU:
                    handle_main_menu();
                    break;
                case GUIPage::CHAT_INTERFACE:
                    handle_chat_interface();
                    break;
                case GUIPage::SYSTEM_STATUS:
                    handle_system_status();
                    break;
                case GUIPage::MEMORY_MANAGEMENT:
                    handle_memory_management();
                    break;
                case GUIPage::CLOUD_STORAGE:
                    handle_cloud_storage();
                    break;
                case GUIPage::SETTINGS:
                    handle_settings();
                    break;
                case GUIPage::ABOUT:
                    handle_about();
                    break;
            }
            draw_footer();
            redraw_needed_ = false;
        }

        // Handle input
        std::string input = get_input();
        if (!input.empty()) {
            process_input(input);
        }

        // Small delay to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void HRMGUI::switch_page(GUIPage page) {
    current_page_ = page;
    selected_menu_item_ = 0;
    redraw_needed_ = true;
}

GUIPage HRMGUI::get_current_page() const {
    return current_page_;
}

void HRMGUI::set_theme(GUITheme theme) {
    theme_ = theme;
    redraw_needed_ = true;
}

void HRMGUI::set_window_title(const std::string& title) {
    window_title_ = title;
    redraw_needed_ = true;
}

void HRMGUI::set_status_bar_text(const std::string& text) {
    status_bar_text_ = text;
    redraw_needed_ = true;
}

void HRMGUI::process_input(const std::string& input) {
    switch (current_page_) {
        case GUIPage::MAIN_MENU:
            handle_main_menu_input(input);
            break;
        case GUIPage::CHAT_INTERFACE:
            handle_chat_input(input);
            break;
        case GUIPage::SYSTEM_STATUS:
            handle_system_status_input(input);
            break;
        case GUIPage::MEMORY_MANAGEMENT:
            handle_memory_management_input(input);
            break;
        case GUIPage::CLOUD_STORAGE:
            handle_cloud_storage_input(input);
            break;
        case GUIPage::SETTINGS:
            handle_settings_input(input);
            break;
        default:
            break;
    }
}

void HRMGUI::handle_special_keys(int key_code) {
    // Handle special keys like arrows, enter, etc.
    switch (key_code) {
        case 27: // ESC
            switch_page(GUIPage::MAIN_MENU);
            break;
        case 10: // Enter
            // Handle enter key based on current page
            break;
        default:
            break;
    }
}

void HRMGUI::draw_header() {
    clear_screen();
    move_cursor(0, 0);

    int width = get_terminal_width();
    std::string header = center_text(window_title_, width);
    set_text_color(get_theme_color("header"));
    std::cout << header << std::endl;

    // Draw separator line
    set_text_color(get_theme_color("separator"));
    std::cout << std::string(width, '=') << std::endl;
    reset_text_color();
}

void HRMGUI::draw_menu(const std::vector<GUIMenuItem>& items) {
    int y = 3; // Start after header
    for (size_t i = 0; i < items.size(); ++i) {
        move_cursor(2, y + i);
        if (static_cast<int>(i) == selected_menu_item_) {
            set_text_color(get_theme_color("selected"));
            std::cout << "> " << items[i].label;
            reset_text_color();
        } else {
            std::cout << "  " << items[i].label;
        }
        if (!items[i].description.empty()) {
            move_cursor(30, y + i);
            set_text_color(get_theme_color("description"));
            std::cout << items[i].description;
            reset_text_color();
        }
    }
}

void HRMGUI::draw_chat_interface() {
    int height = get_terminal_height();
    int chat_height = height - 8; // Leave space for input and status

    // Draw chat history
    int y = 3;
    size_t start_idx = (chat_history_.size() > static_cast<size_t>(chat_height)) ?
                       chat_history_.size() - chat_height : 0;

    for (size_t i = start_idx; i < chat_history_.size() && y < height - 3; ++i) {
        move_cursor(2, y++);
        const auto& msg = chat_history_[i];
        set_text_color(msg.is_user ? get_theme_color("user_message") : get_theme_color("system_message"));
        std::cout << "[" << get_timestamp_string(msg.timestamp) << "] ";
        std::cout << msg.sender << ": " << wrap_text(msg.message, get_terminal_width() - 25);
        reset_text_color();
    }

    // Draw input area
    move_cursor(2, height - 2);
    set_text_color(get_theme_color("input_prompt"));
    std::cout << "You: ";
    reset_text_color();
    std::cout << current_input_;
}

void HRMGUI::draw_system_status() {
    update_system_status();

    int y = 3;
    move_cursor(2, y++);
    set_text_color(get_theme_color("title"));
    std::cout << "System Status";
    reset_text_color();

    // CPU/GPU usage
    auto usage = hrm_system_->get_current_resource_usage();
    move_cursor(2, y++);
    std::cout << "CPU Usage: " << format_resource_usage(usage);

    // Memory stats - using resource usage for now
    move_cursor(2, y++);
    std::cout << "Memory: " << format_memory_stats(usage);

    // Other system info
    move_cursor(2, y++);
    std::cout << "Uptime: " << "N/A"; // TODO: Implement uptime tracking
}

void HRMGUI::draw_memory_management() {
    int y = 3;
    move_cursor(2, y++);
    set_text_color(get_theme_color("title"));
    std::cout << "Memory Management";
    reset_text_color();

    // Get memory stats
    auto mem_stats = hrm_system_->get_memory_compaction_stats();

    move_cursor(2, y++);
    std::cout << "Current Memory Usage: " << (mem_stats.count("memory_current_usage") ?
        mem_stats.at("memory_current_usage") + " bytes" : "Unknown");
    move_cursor(2, y++);
    std::cout << "Compacted Memory: " << (mem_stats.count("memory_compacted_size") ?
        mem_stats.at("memory_compacted_size") + " bytes" : "Unknown");
    move_cursor(2, y++);
    std::cout << "Avg Compression Ratio: " << (mem_stats.count("memory_avg_compression_ratio") ?
        mem_stats.at("memory_avg_compression_ratio") : "Unknown");

    y++;
    move_cursor(2, y++);
    set_text_color(get_theme_color("subtitle"));
    std::cout << "Options:";
    reset_text_color();

    move_cursor(2, y++);
    std::cout << "1. View Detailed Memory Statistics";
    move_cursor(2, y++);
    std::cout << "2. Perform Memory Compaction";
    move_cursor(2, y++);
    std::cout << "3. List Memory Compactions";
    move_cursor(2, y++);
    std::cout << "4. Cloud Storage Operations";
    move_cursor(2, y++);
    std::cout << "5. Back to Main Menu";
}

void HRMGUI::draw_settings() {
    int y = 3;
    move_cursor(2, y++);
    set_text_color(get_theme_color("title"));
    std::cout << "Settings";
    reset_text_color();

    move_cursor(2, y++);
    std::cout << "1. Theme: " << (theme_ == GUITheme::DARK ? "Dark" : theme_ == GUITheme::LIGHT ? "Light" : "Auto");
    move_cursor(2, y++);
    std::cout << "2. Window Title: " << window_title_;
    move_cursor(2, y++);
    std::cout << "3. Back to Main Menu";
}

void HRMGUI::draw_about() {
    int y = 3;
    move_cursor(2, y++);
    set_text_color(get_theme_color("title"));
    std::cout << "About HRM - Hierarchical Reasoning Module";
    reset_text_color();

    move_cursor(2, y++);
    std::cout << "Version: 1.0.0 (Development)";
    move_cursor(2, y++);
    std::cout << "Build: " << __DATE__ << " " << __TIME__;

    y++;
    move_cursor(2, y++);
    set_text_color(get_theme_color("subtitle"));
    std::cout << "System Information:";
    reset_text_color();

    // Get system info from resource monitor
    auto status = hrm_system_->get_resource_aware_status();
    move_cursor(2, y++);
    std::cout << "CPU Cores: " << (status.count("cpu_cores") ? status.at("cpu_cores") : "Unknown");
    move_cursor(2, y++);
    std::cout << "RAM: " << (status.count("total_memory_mb") ? status.at("total_memory_mb") + " MB" : "Unknown");
    move_cursor(2, y++);
    std::cout << "GPU: " << (status.count("vulkan_available") ? (status.at("vulkan_available") == "true" ? "Vulkan Compatible" : "No Vulkan") : "Unknown");

    // Get system uptime
    uint64_t uptime_seconds = 0;
#ifdef _WIN32
    uptime_seconds = GetTickCount64() / 1000;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        uptime_seconds = info.uptime;
    }
#endif
    uint64_t days = uptime_seconds / 86400;
    uint64_t hours = (uptime_seconds % 86400) / 3600;
    uint64_t minutes = (uptime_seconds % 3600) / 60;
    move_cursor(2, y++);
    std::cout << "System Uptime: " << days << "d " << hours << "h " << minutes << "m";

    y++;
    move_cursor(2, y++);
    set_text_color(get_theme_color("subtitle"));
    std::cout << "Capabilities:";
    reset_text_color();

    move_cursor(2, y++);
    std::cout << "• Self-Modifying Code Engine";
    move_cursor(2, y++);
    std::cout << "• Vulkan-Accelerated Neural Networks";
    move_cursor(2, y++);
    std::cout << "• Character-Level Language Processing";
    move_cursor(2, y++);
    std::cout << "• Resource-Aware Task Management";
    move_cursor(2, y++);
    std::cout << "• Cross-Platform Compatibility";

    y++;
    move_cursor(2, y++);
    set_text_color(get_theme_color("subtitle"));
    std::cout << "Press any key to return to main menu";
    reset_text_color();
}

void HRMGUI::draw_footer() {
    int height = get_terminal_height();
    move_cursor(0, height - 1);
    set_text_color(get_theme_color("status_bar"));
    std::cout << std::string(get_terminal_width(), ' ') << "\r";
    std::cout << status_bar_text_;
    reset_text_color();
}

void HRMGUI::add_chat_message(const GUIChatMessage& message) {
    chat_history_.push_back(message);
    redraw_needed_ = true;
}

void HRMGUI::clear_chat_history() {
    chat_history_.clear();
    redraw_needed_ = true;
}

std::vector<GUIChatMessage> HRMGUI::get_chat_history() const {
    return chat_history_;
}

void HRMGUI::show_message_box(const std::string& title, const std::string& message) {
    // Simple message box implementation
    clear_screen();
    int width = get_terminal_width();
    int height = get_terminal_height();

    int box_width = std::min<int>(60, width - 4);
    int box_height = 8;
    int x = (width - box_width) / 2;
    int y = (height - box_height) / 2;

    draw_box(x, y, box_width, box_height, title);

    move_cursor(x + 2, y + 2);
    std::cout << wrap_text(message, box_width - 4);

    move_cursor(x + 2, y + box_height - 2);
    std::cout << "Press any key to continue...";
    std::cin.get();
    redraw_needed_ = true;
}

bool HRMGUI::show_confirmation_dialog(const std::string& message) {
    // Simple confirmation dialog
    clear_screen();
    int width = get_terminal_width();
    int height = get_terminal_height();

    int box_width = std::min<int>(50, width - 4);
    int box_height = 6;
    int x = (width - box_width) / 2;
    int y = (height - box_height) / 2;

    draw_box(x, y, box_width, box_height, "Confirm");

    move_cursor(x + 2, y + 2);
    std::cout << wrap_text(message, box_width - 4);

    move_cursor(x + 2, y + box_height - 2);
    std::cout << "(y/n): ";

    char response;
    std::cin >> response;
    redraw_needed_ = true;
    return (response == 'y' || response == 'Y');
}

std::string HRMGUI::show_input_dialog(const std::string& prompt) {
    clear_screen();
    int width = get_terminal_width();
    int height = get_terminal_height();

    int box_width = std::min<int>(60, width - 4);
    int box_height = 6;
    int x = (width - box_width) / 2;
    int y = (height - box_height) / 2;

    draw_box(x, y, box_width, box_height, "Input");

    move_cursor(x + 2, y + 2);
    std::cout << wrap_text(prompt, box_width - 4);

    move_cursor(x + 2, y + box_height - 2);
    std::cout << "> ";

    std::string input;
    std::getline(std::cin, input);
    redraw_needed_ = true;
    return input;
}

// Private helper methods

void HRMGUI::clear_screen() {
    std::cout << "\033[2J\033[H";
}

void HRMGUI::move_cursor(int x, int y) {
    std::cout << "\033[" << (y + 1) << ";" << (x + 1) << "H";
}

void HRMGUI::set_text_color(const std::string& color) {
    if (color == "red") std::cout << "\033[31m";
    else if (color == "green") std::cout << "\033[32m";
    else if (color == "yellow") std::cout << "\033[33m";
    else if (color == "blue") std::cout << "\033[34m";
    else if (color == "magenta") std::cout << "\033[35m";
    else if (color == "cyan") std::cout << "\033[36m";
    else if (color == "white") std::cout << "\033[37m";
    else if (color == "bright_red") std::cout << "\033[91m";
    else if (color == "bright_green") std::cout << "\033[92m";
    else if (color == "bright_yellow") std::cout << "\033[93m";
    else if (color == "bright_blue") std::cout << "\033[94m";
    else if (color == "bright_magenta") std::cout << "\033[95m";
    else if (color == "bright_cyan") std::cout << "\033[96m";
    else if (color == "bright_white") std::cout << "\033[97m";
    else std::cout << "\033[0m"; // Default/reset
}

void HRMGUI::reset_text_color() {
    std::cout << "\033[0m";
}

std::string HRMGUI::get_theme_color(const std::string& element) {
    if (theme_ == GUITheme::DARK) {
        if (element == "header") return "bright_cyan";
        if (element == "separator") return "blue";
        if (element == "selected") return "bright_yellow";
        if (element == "description") return "cyan";
        if (element == "title") return "bright_green";
        if (element == "user_message") return "bright_blue";
        if (element == "system_message") return "bright_magenta";
        if (element == "input_prompt") return "bright_yellow";
        if (element == "status_bar") return "blue";
    } else {
        // Light theme colors
        if (element == "header") return "blue";
        if (element == "separator") return "cyan";
        if (element == "selected") return "red";
        if (element == "description") return "magenta";
        if (element == "title") return "green";
        if (element == "user_message") return "blue";
        if (element == "system_message") return "magenta";
        if (element == "input_prompt") return "red";
        if (element == "status_bar") return "cyan";
    }
    return "white";
}

void HRMGUI::draw_box(int x, int y, int width, int height, const std::string& title) {
    // Draw top border
    move_cursor(x, y);
    std::cout << "+";
    if (!title.empty()) {
        std::cout << " " << title << " ";
        int remaining = width - title.length() - 4;
        std::cout << std::string(remaining, '-');
    } else {
        std::cout << std::string(width - 2, '-');
    }
    std::cout << "+";

    // Draw sides
    for (int i = 1; i < height - 1; ++i) {
        move_cursor(x, y + i);
        std::cout << "|" << std::string(width - 2, ' ') << "|";
    }

    // Draw bottom border
    move_cursor(x, y + height - 1);
    std::cout << "+" << std::string(width - 2, '-') << "+";
}

void HRMGUI::draw_progress_bar(int x, int y, int width, double progress, const std::string& label) {
    move_cursor(x, y);
    if (!label.empty()) {
        std::cout << label << ": ";
        x += label.length() + 2;
        width -= label.length() + 2;
    }

    int filled = static_cast<int>(progress * width);
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < filled) std::cout << "█";
        else std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(progress * 100) << "%";
}

void HRMGUI::initialize_menus() {
    create_main_menu();
    create_settings_menu();
}

void HRMGUI::create_main_menu() {
    main_menu_items_ = {
        {"Chat Interface", "Start chatting with HRM", [this]() { switch_page(GUIPage::CHAT_INTERFACE); }, 'c'},
        {"System Status", "View system resource usage", [this]() { switch_page(GUIPage::SYSTEM_STATUS); }, 's'},
        {"Memory Management", "Manage memory and compaction", [this]() { switch_page(GUIPage::MEMORY_MANAGEMENT); }, 'm'},
        {"Settings", "Configure HRM settings", [this]() { switch_page(GUIPage::SETTINGS); }, 't'},
        {"About", "About HRM", [this]() { switch_page(GUIPage::ABOUT); }, 'a'},
        {"Exit", "Exit the application", [this]() { exit(0); }, 'x'}
    };
}

void HRMGUI::create_settings_menu() {
    settings_menu_items_ = {
        {"Theme", "Change GUI theme", [this]() { /* TODO */ }, 't'},
        {"Window Title", "Change window title", [this]() { /* TODO */ }, 'w'},
        {"Back", "Return to main menu", [this]() { switch_page(GUIPage::MAIN_MENU); }, 'b'}
    };
}

void HRMGUI::handle_main_menu() {
    draw_menu(main_menu_items_);
}

void HRMGUI::handle_chat_interface() {
    draw_chat_interface();
}

void HRMGUI::handle_system_status() {
    draw_system_status();
}

void HRMGUI::handle_memory_management() {
    draw_memory_management();
}

void HRMGUI::handle_settings() {
    draw_settings();
}

void HRMGUI::handle_about() {
    draw_about();
}

void HRMGUI::send_chat_message(const std::string& message) {
    GUIChatMessage user_msg{"You", message, std::chrono::system_clock::now(), true, 1.0};
    add_chat_message(user_msg);

    awaiting_response_ = true;
    set_status_bar_text("Processing...");

    // Send to HRM system
    auto result = hrm_system_->communicate(message);

    process_chat_response(result);
}

void HRMGUI::process_chat_response(const CommunicationResult& result) {
    GUIChatMessage system_msg{"HRM", result.response, std::chrono::system_clock::now(), false, result.confidence_score};
    add_chat_message(system_msg);

    awaiting_response_ = false;
    set_status_bar_text("Ready");
}

void HRMGUI::update_system_status() {
    // Update timestamp for refresh
    last_update_ = std::chrono::system_clock::now();
}

std::string HRMGUI::format_resource_usage(const ResourceUsage& usage) {
    std::stringstream ss;
    ss << "CPU: " << usage.cpu_usage_percent << "%";
    // Note: No GPU usage in ResourceUsage struct
    return ss.str();
}

std::string HRMGUI::format_memory_stats(const ResourceUsage& usage) {
    std::stringstream ss;
    ss << "Total: " << (usage.total_memory_bytes / 1024 / 1024) << " MB";
    ss << " Used: " << (usage.used_memory_bytes / 1024 / 1024) << " MB";
    ss << " (" << usage.memory_usage_percent << "%)";
    return ss.str();
}

void HRMGUI::show_memory_compaction_options() {
    // TODO: Implement memory compaction options display
}

void HRMGUI::show_cloud_storage_options() {
    switch_page(GUIPage::CLOUD_STORAGE);
}

void HRMGUI::handle_cloud_storage() {
    draw_cloud_storage();
}

void HRMGUI::draw_cloud_storage() {
    int y = 3;
    move_cursor(2, y++);
    set_text_color(get_theme_color("title"));
    std::cout << "Cloud Storage Operations";
    reset_text_color();

    // Get cloud stats
    auto cloud_stats = hrm_system_->get_cloud_storage_stats();

    move_cursor(2, y++);
    std::cout << "Cloud Storage: " << (cloud_stats.count("cloud_enabled") && cloud_stats.at("cloud_enabled") == "true" ?
        "Enabled" : "Disabled");
    move_cursor(2, y++);
    std::cout << "Provider: " << (cloud_stats.count("cloud_provider") ?
        cloud_stats.at("cloud_provider") : "None");

    y++;
    move_cursor(2, y++);
    set_text_color(get_theme_color("subtitle"));
    std::cout << "Options:";
    reset_text_color();

    move_cursor(2, y++);
    std::cout << "1. List Cloud Storage";
    move_cursor(2, y++);
    std::cout << "2. Upload Data to Cloud";
    move_cursor(2, y++);
    std::cout << "3. Download Data from Cloud";
    move_cursor(2, y++);
    std::cout << "4. Back to Memory Management";
}

void HRMGUI::perform_memory_compaction() {
    set_status_bar_text("Performing memory compaction...");
    redraw_needed_ = true;

    // Perform compaction in background
    std::thread([this]() {
        bool success = hrm_system_->perform_memory_compaction();
        if (success) {
            show_message_box("Memory Compaction", "Memory compaction completed successfully!");
        } else {
            show_message_box("Memory Compaction", "Memory compaction failed or no compaction needed.");
        }
        set_status_bar_text("Ready");
        redraw_needed_ = true;
    }).detach();
}

void HRMGUI::upload_to_cloud() {
    // TODO: Implement cloud upload
}

void HRMGUI::download_from_cloud() {
    // TODO: Implement cloud download
}

void HRMGUI::show_current_settings() {
    // TODO: Implement settings display
}

void HRMGUI::change_setting(const std::string& setting_name) {
    // TODO: Implement setting change
}

std::string HRMGUI::get_timestamp_string(const std::chrono::system_clock::time_point& time) {
    auto time_t = std::chrono::system_clock::to_time_t(time);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    return ss.str();
}

std::string HRMGUI::wrap_text(const std::string& text, size_t width) {
    std::string result;
    size_t pos = 0;
    while (pos < text.length()) {
        size_t end = std::min<size_t>(pos + width, text.length());
        if (end < text.length()) {
            // Find last space within width
            size_t last_space = text.rfind(' ', end);
            if (last_space > pos && last_space < end) {
                end = last_space;
            }
        }
        result += text.substr(pos, end - pos);
        if (end < text.length()) result += "\n";
        pos = end;
        if (pos < text.length() && text[pos] == ' ') ++pos;
    }
    return result;
}

std::string HRMGUI::center_text(const std::string& text, size_t width) {
    if (text.length() >= width) return text;
    size_t padding = (width - text.length()) / 2;
    return std::string(padding, ' ') + text + std::string(width - text.length() - padding, ' ');
}

void HRMGUI::sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

void HRMGUI::setup_terminal() {
#ifdef _WIN32
    // Windows console setup
    HANDLE hConsole = GetStdHandle(STD_INPUT_HANDLE);
    GetConsoleMode(hConsole, &original_console_mode_);
    SetConsoleMode(hConsole, ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
    // Hide cursor - Windows equivalent
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsole, &cursorInfo);
    cursorInfo.bVisible = FALSE;
    SetConsoleCursorInfo(hConsole, &cursorInfo);
#else
    // Save current terminal settings
    tcgetattr(STDIN_FILENO, &original_termios_);

    // Set up new terminal settings for raw mode
    struct termios new_termios = original_termios_;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

    // Hide cursor
    std::cout << "\033[?25l";
    std::cout.flush();
#endif
}

void HRMGUI::restore_terminal() {
#ifdef _WIN32
    // Restore Windows console settings
    HANDLE hConsole = GetStdHandle(STD_INPUT_HANDLE);
    SetConsoleMode(hConsole, original_console_mode_);
    // Show cursor
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsole, &cursorInfo);
    cursorInfo.bVisible = TRUE;
    SetConsoleCursorInfo(hConsole, &cursorInfo);
#else
    // Restore original terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);

    // Show cursor
    std::cout << "\033[?25h";
    std::cout.flush();
#endif
}

int HRMGUI::get_terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col;
#endif
}

int HRMGUI::get_terminal_height() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_row;
#endif
}

bool HRMGUI::is_terminal_resized() {
    // TODO: Implement terminal resize detection
    return false;
}

std::string HRMGUI::get_input() {
#ifdef _WIN32
    // Windows non-blocking input
    if (_kbhit()) {
        char ch = _getch();
        if (ch == '\n' || ch == '\r') {
            std::string input = current_input_;
            current_input_.clear();
            return input;
        } else if (ch == 127 || ch == 8) { // Backspace
            if (!current_input_.empty()) {
                current_input_.pop_back();
                redraw_needed_ = true;
            }
        } else if (ch >= 32 && ch <= 126) { // Printable characters
            current_input_ += ch;
            redraw_needed_ = true;
        } else {
            handle_special_keys(ch);
        }
    }
#else
    // Non-blocking input reading
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 10000; // 10ms timeout

    int retval = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
    if (retval > 0) {
        char ch;
        if (read(STDIN_FILENO, &ch, 1) > 0) {
            if (ch == '\n' || ch == '\r') {
                std::string input = current_input_;
                current_input_.clear();
                return input;
            } else if (ch == 127 || ch == 8) { // Backspace
                if (!current_input_.empty()) {
                    current_input_.pop_back();
                    redraw_needed_ = true;
                }
            } else if (ch >= 32 && ch <= 126) { // Printable characters
                current_input_ += ch;
                redraw_needed_ = true;
            } else {
                handle_special_keys(ch);
            }
        }
    }
#endif
    return "";
}

void HRMGUI::handle_main_menu_input(const std::string& input) {
    if (input.empty()) return;

    char choice = tolower(input[0]);
    for (const auto& item : main_menu_items_) {
        if (item.shortcut_key == choice) {
            item.action();
            return;
        }
    }

    // Handle numeric selection
    try {
        int index = std::stoi(input) - 1;
        if (index >= 0 && index < static_cast<int>(main_menu_items_.size())) {
            main_menu_items_[index].action();
        }
    } catch (...) {
        // Invalid input
    }
}

void HRMGUI::handle_chat_input(const std::string& input) {
    if (!input.empty()) {
        send_chat_message(input);
    }
}

void HRMGUI::handle_system_status_input(const std::string& input) {
    if (input == "b" || input == "back") {
        switch_page(GUIPage::MAIN_MENU);
    }
}

void HRMGUI::handle_memory_management_input(const std::string& input) {
    if (input == "1") {
        // View detailed memory statistics
        auto mem_stats = hrm_system_->get_memory_compaction_stats();
        std::string stats = "Detailed Memory Statistics:\n\n";
        for (const auto& stat : mem_stats) {
            stats += stat.first + ": " + stat.second + "\n";
        }
        show_message_box("Detailed Memory Statistics", stats);
    } else if (input == "2") {
        perform_memory_compaction();
    } else if (input == "3") {
        // List memory compactions
        auto compactions = hrm_system_->list_memory_compactions();
        std::string list = "Memory Compactions:\n\n";
        if (compactions.empty()) {
            list += "No compactions found.\n";
        } else {
            for (const auto& id : compactions) {
                list += "- " + id + "\n";
            }
        }
        show_message_box("Memory Compactions", list);
    } else if (input == "4") {
        show_cloud_storage_options();
    } else if (input == "5" || input == "b" || input == "back") {
        switch_page(GUIPage::MAIN_MENU);
    }
}

void HRMGUI::handle_cloud_storage_input(const std::string& input) {
    if (input == "1") {
        // List cloud storage
        auto items = hrm_system_->list_cloud_storage();
        std::string list = "Cloud Storage Items:\n\n";
        if (items.empty()) {
            list += "No items found in cloud storage.\n";
        } else {
            for (const auto& item : items) {
                list += "- " + item + "\n";
            }
        }
        show_message_box("Cloud Storage List", list);
    } else if (input == "2") {
        // Upload to cloud
        std::string data_id = show_input_dialog("Enter data ID to upload:");
        if (!data_id.empty()) {
            set_status_bar_text("Uploading to cloud...");
            std::thread([this, data_id]() {
                bool success = hrm_system_->upload_to_cloud(data_id);
                if (success) {
                    show_message_box("Cloud Upload", "Data uploaded successfully!");
                } else {
                    show_message_box("Cloud Upload", "Upload failed.");
                }
                set_status_bar_text("Ready");
                redraw_needed_ = true;
            }).detach();
        }
    } else if (input == "3") {
        // Download from cloud
        std::string data_id = show_input_dialog("Enter data ID to download:");
        if (!data_id.empty()) {
            set_status_bar_text("Downloading from cloud...");
            std::thread([this, data_id]() {
                bool success = hrm_system_->download_from_cloud(data_id);
                if (success) {
                    show_message_box("Cloud Download", "Data downloaded successfully!");
                } else {
                    show_message_box("Cloud Download", "Download failed.");
                }
                set_status_bar_text("Ready");
                redraw_needed_ = true;
            }).detach();
        }
    } else if (input == "4" || input == "b" || input == "back") {
        switch_page(GUIPage::MEMORY_MANAGEMENT);
    }
}

void HRMGUI::handle_settings_input(const std::string& input) {
    if (input == "1") {
        // Change theme
        theme_ = (theme_ == GUITheme::DARK) ? GUITheme::LIGHT : GUITheme::DARK;
        redraw_needed_ = true;
    } else if (input == "2") {
        // Change window title
        std::string new_title = show_input_dialog("Enter new window title:");
        if (!new_title.empty()) {
            set_window_title(new_title);
        }
    } else if (input == "3" || input == "b" || input == "back") {
        switch_page(GUIPage::MAIN_MENU);
    }
}