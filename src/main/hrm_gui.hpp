#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <termios.h>
#endif
#include "../hrm/resource_aware_hrm.hpp"

enum class GUITheme {
    DARK,
    LIGHT,
    AUTO
};

enum class GUIPage {
    MAIN_MENU,
    CHAT_INTERFACE,
    SYSTEM_STATUS,
    MEMORY_MANAGEMENT,
    CLOUD_STORAGE,
    SETTINGS,
    ABOUT
};

struct GUIMenuItem {
    std::string label;
    std::string description;
    std::function<void()> action;
    char shortcut_key;
};

struct GUIChatMessage {
    std::string sender;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
    bool is_user;
    double confidence_score;
};

class NyxGUI {
public:
    NyxGUI(std::shared_ptr<ResourceAwareHRM> hrm_system);
    ~NyxGUI();

    // Main GUI loop
    void run();

    // Page management
    void switch_page(GUIPage page);
    GUIPage get_current_page() const;

    // UI customization
    void set_theme(GUITheme theme);
    void set_window_title(const std::string& title);
    void set_status_bar_text(const std::string& text);

    // Input handling
    void process_input(const std::string& input);
    void handle_special_keys(int key_code);

    // Drawing functions
    void draw_header();
    void draw_menu(const std::vector<GUIMenuItem>& items);
    void draw_chat_interface();
    void draw_system_status();
    void draw_memory_management();
    void draw_cloud_storage();
    void draw_settings();
    void draw_about();
    void draw_footer();

    // Chat functionality
    void add_chat_message(const GUIChatMessage& message);
    void clear_chat_history();
    std::vector<GUIChatMessage> get_chat_history() const;

    // Utility functions
    void show_message_box(const std::string& title, const std::string& message);
    bool show_confirmation_dialog(const std::string& message);
    std::string show_input_dialog(const std::string& prompt);

private:
    std::shared_ptr<ResourceAwareHRM> hrm_system_;
    GUIPage current_page_;
    GUITheme theme_;
    std::string window_title_;
    std::string status_bar_text_;

    // Chat interface
    std::vector<GUIChatMessage> chat_history_;
    std::string current_input_;
    bool awaiting_response_;

    // Menu system
    std::vector<GUIMenuItem> main_menu_items_;
    std::vector<GUIMenuItem> settings_menu_items_;

    // UI state
    int selected_menu_item_;
    bool redraw_needed_;
    std::chrono::system_clock::time_point last_update_;

    // Terminal state
#ifdef _WIN32
    DWORD original_console_mode_;
#else
    struct termios original_termios_;
#endif

    // Drawing helpers
    void clear_screen();
    void move_cursor(int x, int y);
    void set_text_color(const std::string& color);
    void reset_text_color();
    std::string get_theme_color(const std::string& element);
    void draw_box(int x, int y, int width, int height, const std::string& title = "");
    void draw_progress_bar(int x, int y, int width, double progress, const std::string& label = "");

    // Menu creation
    void initialize_menus();
    void create_main_menu();
    void create_settings_menu();

    // Page handlers
    void handle_main_menu();
    void handle_chat_interface();
    void handle_system_status();
    void handle_memory_management();
    void handle_cloud_storage();
    void handle_about();
    void handle_settings();

    // Chat processing
    void send_chat_message(const std::string& message);
    void process_chat_response(const CommunicationResult& result);

    // System monitoring
    void update_system_status();
    std::string format_resource_usage(const ResourceUsage& usage);
    std::string format_memory_stats(const ResourceUsage& usage);

    // Memory management
    void show_memory_compaction_options();
    void show_cloud_storage_options();
    void perform_memory_compaction();
    void upload_to_cloud();
    void download_from_cloud();

    // Settings management
    void show_current_settings();
    void change_setting(const std::string& setting_name);

    // Utility functions
    std::string get_timestamp_string(const std::chrono::system_clock::time_point& time);
    std::string wrap_text(const std::string& text, size_t width);
    std::string center_text(const std::string& text, size_t width);
    void sleep_ms(int milliseconds);

    // Platform-specific functions
    void setup_terminal();
    void restore_terminal();
    int get_terminal_width();
    int get_terminal_height();
    bool is_terminal_resized();

    // Input handling
    std::string get_input();
    void handle_main_menu_input(const std::string& input);
    void handle_chat_input(const std::string& input);
    void handle_system_status_input(const std::string& input);
    void handle_memory_management_input(const std::string& input);
    void handle_cloud_storage_input(const std::string& input);
    void handle_settings_input(const std::string& input);
};