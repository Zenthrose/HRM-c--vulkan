// C++ examples for reasoning and problem solving
// HRM self-modification and repair capabilities

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <unordered_map>

/**
 * Binary search algorithm - demonstrates algorithmic reasoning
 * Used in HRM for efficient data retrieval and code analysis
 */
int binary_search(const std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

/**
 * Fibonacci sequence - recursive reasoning
 * Demonstrates recursive problem solving in HRM code generation
 */
int fibonacci_recursive(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2);
}

/**
 * Fibonacci sequence - iterative approach
 * More efficient for large n - HRM optimization reasoning
 */
int fibonacci_iterative(int n) {
    if (n <= 1) {
        return n;
    }

    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

/**
 * Primality testing - mathematical reasoning
 * Used in HRM for cryptographic operations and code validation
 */
bool is_prime(int num) {
    if (num <= 1) {
        return false;
    }
    if (num <= 3) {
        return true;
    }
    if (num % 2 == 0 || num % 3 == 0) {
        return false;
    }

    for (int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) {
            return false;
        }
    }

    return true;
}

/**
 * Greatest Common Divisor - Euclidean algorithm
 * Demonstrates mathematical reasoning and recursion
 * HRM uses this for memory management and resource allocation
 */
int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * Sieve of Eratosthenes - efficient prime generation
 * Algorithmic reasoning for optimization
 * HRM applies similar sieving for code optimization
 */
std::vector<int> sieve_of_eratosthenes(int n) {
    std::vector<bool> is_prime(n + 1, true);
    std::vector<int> primes;

    is_prime[0] = is_prime[1] = false;

    for (int i = 2; i * i <= n; ++i) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }

    for (int i = 2; i <= n; ++i) {
        if (is_prime[i]) {
            primes.push_back(i);
        }
    }

    return primes;
}

/**
 * Smart pointer example - RAII pattern
 * Essential for HRM memory management and self-modification safety
 */
class ResourceManager {
private:
    std::unique_ptr<int[]> data;
    size_t size;

public:
    ResourceManager(size_t s) : size(s), data(std::make_unique<int[]>(s)) {
        std::cout << "ResourceManager created with size " << size << std::endl;
    }

    ~ResourceManager() {
        std::cout << "ResourceManager destroyed, cleaning up " << size << " elements" << std::endl;
    }

    void initialize() {
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int>(i * i);
        }
    }

    int get(size_t index) const {
        if (index < size) {
            return data[index];
        }
        return -1;
    }
};

/**
 * Template metaprogramming example
 * HRM uses templates for compile-time code generation and optimization
 */
template<int N>
struct Fibonacci {
    static const int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};

template<>
struct Fibonacci<0> {
    static const int value = 0;
};

template<>
struct Fibonacci<1> {
    static const int value = 1;
};

/**
 * Lambda function example - functional programming in C++
 * HRM uses lambdas for dynamic code generation and callbacks
 */
auto create_multiplier(int factor) {
    return [factor](int x) {
        return x * factor;
    };
}

/**
 * Exception handling - error management in HRM
 */
class HRMException : public std::exception {
private:
    std::string message;

public:
    HRMException(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
};

void safe_operation(int value) {
    if (value < 0) {
        throw HRMException("Negative value not allowed in HRM operation");
    }
    std::cout << "Operation successful with value: " << value << std::endl;
}

/**
 * Thread-safe singleton pattern
 * HRM uses singletons for managing global resources
 */
class HRMCore {
private:
    static std::unique_ptr<HRMCore> instance;
    static std::mutex mutex;

    HRMCore() {
        std::cout << "HRM Core initialized" << std::endl;
    }

public:
    static HRMCore& getInstance() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!instance) {
            instance = std::unique_ptr<HRMCore>(new HRMCore());
        }
        return *instance;
    }

    void process_request(const std::string& request) {
        std::cout << "HRM processing: " << request << std::endl;
    }
};

std::unique_ptr<HRMCore> HRMCore::instance = nullptr;
std::mutex HRMCore::mutex;

/**
 * Observer pattern - event handling in HRM
 */
class Observer {
public:
    virtual void update(const std::string& message) = 0;
    virtual ~Observer() = default;
};

class Subject {
private:
    std::vector<std::shared_ptr<Observer>> observers;

public:
    void attach(std::shared_ptr<Observer> observer) {
        observers.push_back(observer);
    }

    void notify(const std::string& message) {
        for (auto& observer : observers) {
            observer->update(message);
        }
    }
};

class HRMMonitor : public Observer {
public:
    void update(const std::string& message) override {
        std::cout << "HRM Monitor: " << message << std::endl;
    }
};

/**
 * Code analysis example - HRM self-inspection capability
 */
class CodeAnalyzer {
private:
    std::unordered_map<std::string, int> keyword_counts;

public:
    void analyze_code(const std::string& code) {
        // Simple keyword counting for demonstration
        std::vector<std::string> keywords = {
            "class", "struct", "void", "int", "float", "double",
            "if", "else", "for", "while", "return", "const"
        };

        for (const auto& keyword : keywords) {
            size_t pos = 0;
            int count = 0;
            while ((pos = code.find(keyword, pos)) != std::string::npos) {
                ++count;
                pos += keyword.length();
            }
            keyword_counts[keyword] = count;
        }
    }

    void print_analysis() const {
        std::cout << "Code Analysis Results:" << std::endl;
        for (const auto& pair : keyword_counts) {
            std::cout << pair.first << ": " << pair.second << " occurrences" << std::endl;
        }
    }
};

// Example usage and reasoning
int main() {
    std::cout << "C++ Algorithm Demonstrations for HRM Self-Modification" << std::endl;
    std::cout << "======================================================" << std::endl;

    // Binary search demonstration
    std::vector<int> arr = {1, 3, 5, 7, 9, 11, 13, 15};
    int target = 7;
    int result = binary_search(arr, target);
    std::cout << "Binary search: " << target << " found at index " << result << std::endl;

    // Fibonacci comparison
    int n = 10;
    int recursive_result = fibonacci_recursive(n);
    int iterative_result = fibonacci_iterative(n);
    std::cout << "Fibonacci " << n << ": recursive=" << recursive_result
              << ", iterative=" << iterative_result << std::endl;

    // Compile-time Fibonacci using templates
    std::cout << "Compile-time Fibonacci " << n << ": " << Fibonacci<10>::value << std::endl;

    // Prime checking
    std::vector<int> test_numbers = {2, 3, 4, 5, 16, 17, 18, 19, 23};
    for (int num : test_numbers) {
        std::cout << num << " is " << (is_prime(num) ? "prime" : "not prime") << std::endl;
    }

    // GCD calculation
    std::cout << "GCD of 48 and 18 is " << gcd(48, 18) << std::endl;

    // Prime generation
    std::vector<int> primes = sieve_of_eratosthenes(50);
    std::cout << "Primes up to 50: ";
    for (int prime : primes) {
        std::cout << prime << " ";
    }
    std::cout << std::endl;

    // Smart pointer demonstration
    {
        ResourceManager rm(5);
        rm.initialize();
        std::cout << "Resource value at index 3: " << rm.get(3) << std::endl;
    } // Automatic cleanup here

    // Lambda function demonstration
    auto doubler = create_multiplier(2);
    auto tripler = create_multiplier(3);
    std::cout << "5 doubled: " << doubler(5) << std::endl;
    std::cout << "5 tripled: " << tripler(5) << std::endl;

    // Exception handling
    try {
        safe_operation(10);
        safe_operation(-5); // This will throw
    } catch (const HRMException& e) {
        std::cout << "Caught HRM exception: " << e.what() << std::endl;
    }

    // Singleton pattern
    HRMCore& core = HRMCore::getInstance();
    core.process_request("Analyze system performance");

    // Observer pattern
    Subject subject;
    auto monitor = std::make_shared<HRMMonitor>();
    subject.attach(monitor);
    subject.notify("System optimization completed");

    // Code analysis
    CodeAnalyzer analyzer;
    std::string sample_code = R"(
        class Example {
        public:
            void process() {
                if (condition) {
                    for (int i = 0; i < 10; i++) {
                        // do something
                    }
                }
            }
        };
    )";
    analyzer.analyze_code(sample_code);
    analyzer.print_analysis();

    std::cout << std::endl;
    std::cout << "These C++ patterns enable HRM to:" << std::endl;
    std::cout << "• Manage memory safely with smart pointers" << std::endl;
    std::cout << "• Generate code at compile-time with templates" << std::endl;
    std::cout << "• Handle errors gracefully with exceptions" << std::endl;
    std::cout << "• Implement thread-safe resource management" << std::endl;
    std::cout << "• Analyze and modify its own code structure" << std::endl;
    std::cout << "• Use functional programming for dynamic behavior" << std::endl;

    return 0;
}
