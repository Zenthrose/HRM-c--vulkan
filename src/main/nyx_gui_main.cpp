#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <memory>

#include "../system/hardware_profiler.hpp"
#include "../hrm/resource_aware_hrm.hpp"

// Simple worker class for background operations
class HrmWorker : public QObject {
    Q_OBJECT

public:
    HrmWorker(std::shared_ptr<ResourceAwareHRM> hrm) : hrm_(hrm) {}

public slots:
    void processMessage(const QString& message) {
        // Simple placeholder response
        QString response = "Nyx: I acknowledge your message: " + message;
        emit messageProcessed(response, 0.8f);
    }

signals:
    void messageProcessed(const QString& response, float confidence);
    void errorOccurred(const QString& error);

private:
    std::shared_ptr<ResourceAwareHRM> hrm_;
};

// Main window class
class NyxMainWindow : public QMainWindow {
    Q_OBJECT

public:
    NyxMainWindow(std::shared_ptr<ResourceAwareHRM> hrm, QWidget* parent = nullptr)
        : QMainWindow(parent), hrm_(hrm) {

        setupUI();
        setupWorker();
        setupTimers();
        setWindowTitle("Nyx - Primordial Goddess of Night");
        resize(1200, 800);
    }

private slots:
    void sendMessage() {
        QString message = messageInput_->text().trimmed();
        if (!message.isEmpty()) {
            addToChatHistory("You: " + message);
            messageInput_->clear();

            // Show typing indicator
            addToChatHistory("Nyx is thinking...");

            // Process in background thread
            QMetaObject::invokeMethod(worker_, "processMessage",
                                    Qt::QueuedConnection,
                                    Q_ARG(QString, message));
        }
    }

    void onMessageProcessed(const QString& response, float confidence) {
        // Remove typing indicator
        removeLastChatLine();

        // Add actual response
        QString confidenceText = QString(" (confidence: %1%)").arg(confidence * 100, 0, 'f', 1);
        addToChatHistory("Nyx: " + response + confidenceText);
    }

    void onErrorOccurred(const QString& error) {
        removeLastChatLine();
        addToChatHistory("Error: " + error);
    }

    void updateStatus() {
        try {
            auto usage = hrm_->get_current_resource_usage();
            statusLabel_->setText(QString("CPU: %1% | Memory: %2% | Status: Active")
                                .arg(usage.cpu_usage_percent, 0, 'f', 1)
                                .arg(usage.memory_usage_percent, 0, 'f', 1));
        } catch (const std::exception& e) {
            statusLabel_->setText("Status: Error - " + QString::fromStdString(e.what()));
        }
    }

private:
    void setupUI() {
        // Create central widget and main layout
        QWidget* centralWidget = new QWidget;
        setCentralWidget(centralWidget);

        QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

        // Create tab widget for different sections
        tabWidget_ = new QTabWidget;
        mainLayout->addWidget(tabWidget_);

        // Chat tab
        setupChatTab();

        // Status tab
        setupStatusTab();

        // Memory tab
        setupMemoryTab();

        // Settings tab
        setupSettingsTab();

        // Status bar
        statusLabel_ = new QLabel("Initializing...");
        statusBar()->addWidget(statusLabel_);

        // Menu bar
        setupMenuBar();
    }

    void setupChatTab() {
        QWidget* chatTab = new QWidget;
        QVBoxLayout* layout = new QVBoxLayout(chatTab);

        // Chat history
        chatHistory_ = new QTextEdit;
        chatHistory_->setReadOnly(true);
        layout->addWidget(chatHistory_);

        // Input area
        QHBoxLayout* inputLayout = new QHBoxLayout;

        messageInput_ = new QLineEdit;
        messageInput_->setPlaceholderText("Type your message to Nyx...");
        connect(messageInput_, &QLineEdit::returnPressed, this, &NyxMainWindow::sendMessage);
        inputLayout->addWidget(messageInput_);

        QPushButton* sendButton = new QPushButton("Send");
        connect(sendButton, &QPushButton::clicked, this, &NyxMainWindow::sendMessage);
        inputLayout->addWidget(sendButton);

        layout->addLayout(inputLayout);

        tabWidget_->addTab(chatTab, "Chat");
    }

    void setupStatusTab() {
        QWidget* statusTab = new QWidget;
        QVBoxLayout* layout = new QVBoxLayout(statusTab);

        // CPU Usage Chart
        cpuChart_ = new QChart();
        cpuChart_->setTitle("CPU Usage (%)");
        cpuSeries_ = new QLineSeries();
        cpuChart_->addSeries(cpuSeries_);
        cpuChart_->createDefaultAxes();
        cpuChart_->axes(Qt::Vertical).first()->setRange(0, 100);

        QChartView* cpuChartView = new QChartView(cpuChart_);
        cpuChartView->setMinimumHeight(200);
        layout->addWidget(cpuChartView);

        // Memory Usage Chart
        memoryChart_ = new QChart();
        memoryChart_->setTitle("Memory Usage (%)");
        memorySeries_ = new QLineSeries();
        memoryChart_->addSeries(memorySeries_);
        memoryChart_->createDefaultAxes();
        memoryChart_->axes(Qt::Vertical).first()->setRange(0, 100);

        QChartView* memoryChartView = new QChartView(memoryChart_);
        memoryChartView->setMinimumHeight(200);
        layout->addWidget(memoryChartView);

        tabWidget_->addTab(statusTab, "System Status");
    }

    void setupMemoryTab() {
        QWidget* memoryTab = new QWidget;
        QVBoxLayout* layout = new QVBoxLayout(memoryTab);

        // Memory compaction controls
        QGroupBox* compactionGroup = new QGroupBox("Memory Compaction");
        QVBoxLayout* compactionLayout = new QVBoxLayout(compactionGroup);

        QPushButton* compactButton = new QPushButton("Perform Memory Compaction");
        compactionLayout->addWidget(compactButton);

        compactionProgress_ = new QProgressBar;
        compactionProgress_->setVisible(false);
        compactionLayout->addWidget(compactionProgress_);

        layout->addWidget(compactionGroup);

        // Cloud storage controls
        QGroupBox* cloudGroup = new QGroupBox("Cloud Storage");
        QVBoxLayout* cloudLayout = new QVBoxLayout(cloudGroup);

        QPushButton* uploadButton = new QPushButton("Upload to Cloud");
        QPushButton* downloadButton = new QPushButton("Download from Cloud");
        cloudLayout->addWidget(uploadButton);
        cloudLayout->addWidget(downloadButton);

        layout->addWidget(cloudGroup);
        layout->addStretch();

        tabWidget_->addTab(memoryTab, "Memory & Cloud");
    }

    void setupSettingsTab() {
        QWidget* settingsTab = new QWidget;
        QVBoxLayout* layout = new QVBoxLayout(settingsTab);

        QGroupBox* generalGroup = new QGroupBox("General Settings");
        QFormLayout* generalLayout = new QFormLayout(generalGroup);

        // Theme selection
        QComboBox* themeCombo = new QComboBox;
        themeCombo->addItems({"Dark", "Light", "Auto"});
        themeCombo->setCurrentText("Dark");
        generalLayout->addRow("Theme:", themeCombo);

        layout->addWidget(generalGroup);

        QGroupBox* systemGroup = new QGroupBox("System Settings");
        QFormLayout* systemLayout = new QFormLayout(systemGroup);

        // Resource limits
        QSpinBox* cpuLimitSpin = new QSpinBox;
        cpuLimitSpin->setRange(10, 100);
        cpuLimitSpin->setValue(80);
        cpuLimitSpin->setSuffix("%");
        systemLayout->addRow("CPU Limit:", cpuLimitSpin);

        QSpinBox* memoryLimitSpin = new QSpinBox;
        memoryLimitSpin->setRange(100, 4096);
        memoryLimitSpin->setValue(1024);
        memoryLimitSpin->setSuffix(" MB");
        systemLayout->addRow("Memory Limit:", memoryLimitSpin);

        layout->addWidget(systemGroup);
        layout->addStretch();

        tabWidget_->addTab(settingsTab, "Settings");
    }

    void setupMenuBar() {
        QMenuBar* menuBar = this->menuBar();

        // File menu
        QMenu* fileMenu = menuBar->addMenu("&File");
        QAction* exitAction = fileMenu->addAction("E&xit");
        connect(exitAction, &QAction::triggered, this, &QWidget::close);

        // View menu
        QMenu* viewMenu = menuBar->addMenu("&View");
        QAction* statusAction = viewMenu->addAction("&Status");
        connect(statusAction, &QAction::triggered, [this]() { tabWidget_->setCurrentIndex(1); });

        // Help menu
        QMenu* helpMenu = menuBar->addMenu("&Help");
        QAction* aboutAction = helpMenu->addAction("&About");
        connect(aboutAction, &QAction::triggered, this, &NyxMainWindow::showAbout);
    }

    void setupWorker() {
        // Create worker thread for HRM operations
        workerThread_ = new QThread;
        worker_ = new HrmWorker(hrm_);
        worker_->moveToThread(workerThread_);

        // Connect signals
        connect(worker_, &HrmWorker::messageProcessed, this, &NyxMainWindow::onMessageProcessed);
        connect(worker_, &HrmWorker::errorOccurred, this, &NyxMainWindow::onErrorOccurred);

        workerThread_->start();
    }

    void setupTimers() {
        // Status update timer (every 2 seconds)
        statusTimer_ = new QTimer(this);
        connect(statusTimer_, &QTimer::timeout, this, &NyxMainWindow::updateStatus);
        statusTimer_->start(2000);

        // Chart update timer (every 5 seconds)
        chartTimer_ = new QTimer(this);
        connect(chartTimer_, &QTimer::timeout, this, &NyxMainWindow::updateCharts);
        chartTimer_->start(5000);
    }

    void addToChatHistory(const QString& message) {
        chatHistory_->append(message);
        // Auto-scroll to bottom
        QTextCursor cursor = chatHistory_->textCursor();
        cursor.movePosition(QTextCursor::End);
        chatHistory_->setTextCursor(cursor);
    }

    void removeLastChatLine() {
        QString text = chatHistory_->toPlainText();
        QStringList lines = text.split('\n');
        if (!lines.isEmpty()) {
            lines.removeLast();
            chatHistory_->setPlainText(lines.join('\n'));
        }
    }

    void updateCharts() {
        try {
            auto usage = hrm_->get_current_resource_usage();

            // Update CPU chart
            cpuSeries_->append(QPointF(cpuSeries_->count(), usage.cpu_usage_percent));
            if (cpuSeries_->count() > 50) { // Keep last 50 points
                cpuSeries_->remove(0);
            }

            // Update memory chart
            memorySeries_->append(QPointF(memorySeries_->count(), usage.memory_usage_percent));
            if (memorySeries_->count() > 50) {
                memorySeries_->remove(0);
            }

            // Update axes
            cpuChart_->axes(Qt::Horizontal).first()->setRange(0, cpuSeries_->count());
            memoryChart_->axes(Qt::Horizontal).first()->setRange(0, memorySeries_->count());

        } catch (const std::exception& e) {
            qWarning() << "Failed to update charts:" << e.what();
        }
    }

    void showAbout() {
        QMessageBox::about(this, "About Nyx",
            "Nyx - Primordial Goddess of Night\n"
            "A self-evolving AI system with Vulkan acceleration\n\n"
            "Features:\n"
            "• Real-time conversation\n"
            "• Self-modification and learning\n"
            "• Resource-aware operation\n"
            "• Vulkan GPU acceleration\n"
            "• Memory compaction and cloud storage");
    }

private:
    std::shared_ptr<ResourceAwareHRM> hrm_;

    // UI Components
    QTabWidget* tabWidget_;
    QTextEdit* chatHistory_;
    QLineEdit* messageInput_;
    QLabel* statusLabel_;
    QProgressBar* compactionProgress_;

    // Charts
    QChart* cpuChart_;
    QLineSeries* cpuSeries_;
    QChart* memoryChart_;
    QLineSeries* memorySeries_;

    // Background processing
    QThread* workerThread_;
    HrmWorker* worker_;

    // Timers
    QTimer* statusTimer_;
    QTimer* chartTimer_;
};

// Main application
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("Nyx");
    app.setApplicationVersion("1.0");
    app.setOrganizationName("Nyx AI");

    try {
        // Initialize hardware profiling
        HardwareProfiler hw_profiler;
        HardwareCapabilities hw_caps = hw_profiler.profile_system();

        // Initialize HRM system
        auto hrm_config = createDefaultHRMConfig(hw_caps);
        auto hrm = std::make_shared<ResourceAwareHRM>(hrm_config);

        // Create and show main window
        NyxMainWindow window(hrm);
        window.show();

        return app.exec();

    } catch (const std::exception& e) {
        QMessageBox::critical(nullptr, "Nyx Error",
            QString("Failed to initialize Nyx: %1").arg(e.what()));
        return 1;
    }
}

#include "nyx_gui_main.moc"