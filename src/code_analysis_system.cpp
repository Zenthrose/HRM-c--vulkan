#include "code_analysis_system.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <unordered_map>

CodeAnalysisSystem::CodeAnalysisSystem(const std::string& project_root)
    : project_root_(project_root) {
    std::cout << "Initializing Code Analysis System..." << std::endl;
    discover_source_files();
    initialize_bug_patterns();
    std::cout << "Code Analysis System initialized with " << source_files_.size() << " source files" << std::endl;
}

CodeAnalysisResult CodeAnalysisSystem::analyze_codebase() {
    CodeAnalysisResult result;
    result.overall_quality_score = 1.0f;

    for (const auto& file_path : source_files_) {
        auto file_result = analyze_file(file_path);
        result.issues.insert(result.issues.end(), file_result.issues.begin(), file_result.issues.end());

        // Aggregate statistics
        for (const auto& pair : file_result.issue_counts_by_type) {
            result.issue_counts_by_type[pair.first] += pair.second;
        }
        for (const auto& pair : file_result.issue_counts_by_severity) {
            result.issue_counts_by_severity[pair.first] += pair.second;
        }
    }

    result.overall_quality_score = calculate_code_quality_score(result.issues);
    result.recommendations = generate_recommendations(result);

    return result;
}

CodeAnalysisResult CodeAnalysisSystem::analyze_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file " << file_path << std::endl;
        return CodeAnalysisResult{};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string code = buffer.str();

    return analyze_code_snippet(code, file_path);
}

CodeAnalysisResult CodeAnalysisSystem::analyze_code_snippet(const std::string& code, const std::string& file_path) {
    CodeAnalysisResult result;

    // Run all detection methods
    auto memory_issues = detect_memory_issues(code, file_path);
    auto logic_errors = detect_logic_errors(code, file_path);
    auto performance_issues = detect_performance_issues(code, file_path);
    auto security_vulnerabilities = detect_security_vulnerabilities(code, file_path);
    auto style_issues = detect_style_issues(code, file_path);

    // Combine all issues
    result.issues.insert(result.issues.end(), memory_issues.begin(), memory_issues.end());
    result.issues.insert(result.issues.end(), logic_errors.begin(), logic_errors.end());
    result.issues.insert(result.issues.end(), performance_issues.begin(), performance_issues.end());
    result.issues.insert(result.issues.end(), security_vulnerabilities.begin(), security_vulnerabilities.end());
    result.issues.insert(result.issues.end(), style_issues.begin(), style_issues.end());

    // Count issues by type and severity
    for (const auto& issue : result.issues) {
        result.issue_counts_by_type[issue.issue_type]++;
        result.issue_counts_by_severity[issue.severity]++;
    }

    result.overall_quality_score = calculate_code_quality_score(result.issues);
    result.recommendations = generate_recommendations(result);

    return result;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_memory_issues(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    auto null_pointer_issues = detect_null_pointer_dereferences(code, file_path);
    auto memory_leak_issues = detect_memory_leaks(code, file_path);
    auto buffer_overflow_issues = detect_buffer_overflows(code, file_path);

    issues.insert(issues.end(), null_pointer_issues.begin(), null_pointer_issues.end());
    issues.insert(issues.end(), memory_leak_issues.begin(), memory_leak_issues.end());
    issues.insert(issues.end(), buffer_overflow_issues.begin(), buffer_overflow_issues.end());

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_logic_errors(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    auto uninitialized_issues = detect_uninitialized_variables(code, file_path);
    auto infinite_loop_issues = detect_infinite_loops(code, file_path);

    issues.insert(issues.end(), uninitialized_issues.begin(), uninitialized_issues.end());
    issues.insert(issues.end(), infinite_loop_issues.begin(), infinite_loop_issues.end());

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_performance_issues(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: Inefficient string concatenation in loops
    std::regex inefficient_concat(R"(for\s*\([^)]*\)\s*\{[^}]*\+=[^}]*std::string[^}]*\})");
    std::smatch match;
    std::string::const_iterator search_start(code.cbegin());

    while (std::regex_search(search_start, code.cend(), match, inefficient_concat)) {
        CodeIssue issue{
            file_path,
            0, // Would need line number calculation
            "performance",
            "Inefficient string concatenation in loop",
            "medium",
            "Use std::stringstream or reserve string capacity",
            {}
        };
        issues.push_back(issue);
        search_start = match.suffix().first;
    }

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_security_vulnerabilities(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    auto race_condition_issues = detect_race_conditions(code, file_path);

    issues.insert(issues.end(), race_condition_issues.begin(), race_condition_issues.end());

    // Pattern: Use of dangerous functions
    std::vector<std::string> dangerous_functions = {"strcpy", "strcat", "sprintf", "gets"};
    for (const auto& func : dangerous_functions) {
        std::regex pattern("\\b" + func + "\\s*\\(");
        if (std::regex_search(code, pattern)) {
            CodeIssue issue{
                file_path,
                0,
                "security",
                "Use of dangerous function: " + func,
                "high",
                "Use safer alternatives like strcpy_s, strncat, or std::string",
                {}
            };
            issues.push_back(issue);
        }
    }

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_style_issues(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: Missing const correctness
    std::regex non_const_ref(R"((?:^|[^&])\s*\w+\s*&[^&])");
    if (std::regex_search(code, non_const_ref)) {
        CodeIssue issue{
            file_path,
            0,
            "style",
            "Potential missing const qualifier on reference parameter",
            "low",
            "Consider adding const qualifier for reference parameters",
            {}
        };
        issues.push_back(issue);
    }

    return issues;
}

// Specific bug detection implementations
std::vector<CodeIssue> CodeAnalysisSystem::detect_null_pointer_dereferences(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: Direct dereference without null check
    std::regex null_deref(R"(\w+\s*->\s*\w+|\\\*\s*\w+\s*\[)");
    std::smatch match;
    std::string::const_iterator search_start(code.cbegin());

    while (std::regex_search(search_start, code.cend(), match, null_deref)) {
        // Check if there's a null check nearby (simplified)
        std::string context = extract_code_context(read_file_lines(file_path), 0, 5);
        bool has_null_check = context.find("if (") != std::string::npos &&
                             (context.find("!= nullptr") != std::string::npos ||
                              context.find("!= NULL") != std::string::npos);

        if (!has_null_check) {
            CodeIssue issue{
                file_path,
                0,
                "memory",
                "Potential null pointer dereference",
                "high",
                "Add null check before dereferencing pointer",
                {context}
            };
            issues.push_back(issue);
        }
        search_start = match.suffix().first;
    }

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_memory_leaks(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: new without corresponding delete
    std::regex new_without_delete(R"(new\s+\w+)");
    std::smatch match;
    std::string::const_iterator search_start(code.cbegin());

    while (std::regex_search(search_start, code.cend(), match, new_without_delete)) {
        // Check if there's a corresponding delete (simplified check)
        std::string allocation = match.str();
        bool has_delete = code.find("delete") != std::string::npos;

        if (!has_delete) {
            CodeIssue issue{
                file_path,
                0,
                "memory",
                "Potential memory leak: new without delete",
                "high",
                "Ensure all new allocations have corresponding delete",
                {}
            };
            issues.push_back(issue);
        }
        search_start = match.suffix().first;
    }

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_buffer_overflows(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: Array access without bounds checking
    std::regex array_access(R"(\w+\s*\[\s*\w+\s*\])");
    std::smatch match;
    std::string::const_iterator search_start(code.cbegin());

    while (std::regex_search(search_start, code.cend(), match, array_access)) {
        CodeIssue issue{
            file_path,
            0,
            "memory",
            "Potential buffer overflow: unchecked array access",
            "high",
            "Add bounds checking before array access",
            {}
        };
        issues.push_back(issue);
        search_start = match.suffix().first;
    }

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_uninitialized_variables(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: Variable declaration without initialization
    std::regex uninitialized_var(R"(\bint\s+\w+\s*;|float\s+\w+\s*;|double\s+\w+\s*;)");
    std::smatch match;
    std::string::const_iterator search_start(code.cbegin());

    while (std::regex_search(search_start, code.cend(), match, uninitialized_var)) {
        std::string var_decl = match.str();
        // Skip if it's a function parameter or has initialization
        if (var_decl.find('=') == std::string::npos &&
            var_decl.find('(') == std::string::npos) {
            CodeIssue issue{
                file_path,
                0,
                "logic",
                "Uninitialized variable: " + var_decl,
                "medium",
                "Initialize variables at declaration",
                {}
            };
            issues.push_back(issue);
        }
        search_start = match.suffix().first;
    }

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_infinite_loops(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: while(true) or for(;;) without break
    std::regex infinite_loop(R"(while\s*\(\s*true\s*\)|for\s*\(\s*;;\s*\))");
    if (std::regex_search(code, infinite_loop)) {
        CodeIssue issue{
            file_path,
            0,
            "logic",
            "Potential infinite loop",
            "medium",
            "Ensure loop has proper termination condition",
            {}
        };
        issues.push_back(issue);
    }

    return issues;
}

std::vector<CodeIssue> CodeAnalysisSystem::detect_race_conditions(const std::string& code, const std::string& file_path) {
    std::vector<CodeIssue> issues;

    // Pattern: Shared data access without synchronization
    if (code.find("std::thread") != std::string::npos ||
        code.find("pthread") != std::string::npos) {
        if (code.find("std::mutex") == std::string::npos &&
            code.find("pthread_mutex") == std::string::npos) {
            CodeIssue issue{
                file_path,
                0,
                "security",
                "Potential race condition: multithreaded code without synchronization",
                "high",
                "Add proper synchronization primitives (mutex, atomic, etc.)",
                {}
            };
            issues.push_back(issue);
        }
    }

    return issues;
}

std::vector<CodeModification> CodeAnalysisSystem::generate_fixes(const std::vector<CodeIssue>& issues) {
    std::vector<CodeModification> modifications;

    for (const auto& issue : issues) {
        if (issue.issue_type == "memory") {
            modifications.push_back(create_memory_fix(issue));
        } else if (issue.issue_type == "logic") {
            modifications.push_back(create_logic_fix(issue));
        } else if (issue.issue_type == "performance") {
            modifications.push_back(create_performance_fix(issue));
        } else if (issue.issue_type == "security") {
            modifications.push_back(create_security_fix(issue));
        } else if (issue.issue_type == "style") {
            modifications.push_back(create_style_fix(issue));
        }
    }

    return modifications;
}

bool CodeAnalysisSystem::apply_modification(const CodeModification& modification) {
    // Read the file
    std::vector<std::string> lines = read_file_lines(modification.file_path);
    if (lines.empty()) return false;

    // Apply the modification
    if (modification.start_line >= 0 && modification.start_line < static_cast<int>(lines.size())) {
        if (modification.start_line == modification.end_line) {
            // Single line replacement
            lines[modification.start_line] = modification.modified_code;
        } else {
            // Multi-line replacement
            lines.erase(lines.begin() + modification.start_line,
                       lines.begin() + modification.end_line + 1);
            lines.insert(lines.begin() + modification.start_line, modification.modified_code);
        }

        // Write back to file
        std::ofstream file(modification.file_path);
        if (!file.is_open()) return false;

        for (const auto& line : lines) {
            file << line << '\n';
        }

        return true;
    }

    return false;
}

bool CodeAnalysisSystem::validate_modification(const CodeModification& modification) {
    // Basic validation: check if the modification makes sense
    if (modification.modified_code.empty()) return false;
    if (modification.confidence_score < 0.5f) return false;

    // Check for obvious syntax errors in the modified code
    if (modification.modified_code.find(";;") != std::string::npos) return false;
    if (modification.modified_code.find("{{") != std::string::npos) return false;
    if (modification.modified_code.find("}}") != std::string::npos) return false;

    return true;
}

bool CodeAnalysisSystem::test_compilation(const std::vector<CodeModification>& modifications) {
    // This would require integrating with a build system
    // For now, return true as placeholder
    std::cout << "Compilation test placeholder - would integrate with CMake/Make" << std::endl;
    return true;
}

bool CodeAnalysisSystem::run_unit_tests(const std::vector<CodeModification>& modifications) {
    // This would require integrating with a test framework
    // For now, return true as placeholder
    std::cout << "Unit test placeholder - would integrate with testing framework" << std::endl;
    return true;
}

// Private helper methods
void CodeAnalysisSystem::discover_source_files() {
    try {
        std::string project_path = fs::absolute(project_root_).string();
        
        // Handle Windows vs Unix path separators
        if (project_path.back() != '/' && project_path.back() != '\\') {
            project_path += fs::path::preferred_separator;
        }
        
        std::cout << "Scanning project path: " << project_path << std::endl;
        
        for (const auto& entry : fs::recursive_directory_iterator(project_path)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();
                
                // Check for source file extensions (case-insensitive)
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".cpp" || ext == ".hpp" || ext == ".cc" || 
                    ext == ".h" || ext == ".c" || ext == ".py" || 
                    ext == ".js" || ext == ".java" || ext == ".cs") {
                    source_files_.push_back(path);
                    std::cout << "Found source file: " << path << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error discovering source files: " << e.what() << std::endl;
        
        // Fallback: try common source directories
        std::vector<std::string> fallback_dirs = {"./src", "./include", "./lib"};
        for (const auto& dir : fallback_dirs) {
            if (fs::exists(dir) && fs::is_directory(dir)) {
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.is_regular_file()) {
                        std::string path = entry.path().string();
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".cpp" || ext == ".hpp" || ext == ".cc" || 
                            ext == ".h" || ext == ".c" || ext == ".py") {
                            source_files_.push_back(path);
                            std::cout << "Fallback found source: " << path << std::endl;
                        }
                    }
                }
            }
        }
    }
}

std::vector<std::string> CodeAnalysisSystem::read_file_lines(const std::string& file_path) {
    std::vector<std::string> lines;
    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}

std::string CodeAnalysisSystem::extract_code_context(const std::vector<std::string>& lines,
                                                   int line_number, int context_lines) {
    std::string context;
    int start = std::max(0, line_number - context_lines);
    int end = std::min(static_cast<int>(lines.size()) - 1, line_number + context_lines);

    for (int i = start; i <= end; ++i) {
        context += lines[i] + "\n";
    }

    return context;
}

void CodeAnalysisSystem::initialize_bug_patterns() {
    // Initialize regex patterns for common bugs
    bug_patterns_["null_pointer"] = std::regex(R"(\*\s*\w+\s*[\[\.])");
    bug_patterns_["memory_leak"] = std::regex(R"(new\s+\w+)");
    bug_patterns_["buffer_overflow"] = std::regex(R"(\w+\s*\[\s*\w+\s*\])");
}

float CodeAnalysisSystem::calculate_code_quality_score(const std::vector<CodeIssue>& issues) {
    if (issues.empty()) return 1.0f;

    float score = 1.0f;
    std::unordered_map<std::string, float> severity_weights = {
        {"low", 0.1f},
        {"medium", 0.3f},
        {"high", 0.6f},
        {"critical", 1.0f}
    };

    for (const auto& issue : issues) {
        score -= severity_weights[issue.severity] * 0.05f; // 5% penalty per issue
    }

    return std::max(0.0f, score);
}

std::vector<std::string> CodeAnalysisSystem::generate_recommendations(const CodeAnalysisResult& result) {
    std::vector<std::string> recommendations;

    auto critical_count = result.issue_counts_by_severity.find("critical");
    if (critical_count != result.issue_counts_by_severity.end() && critical_count->second > 0) {
        recommendations.push_back("Fix critical issues immediately - they may cause crashes or security vulnerabilities");
    }

    auto high_count = result.issue_counts_by_severity.find("high");
    if (high_count != result.issue_counts_by_severity.end() && high_count->second > 5) {
        recommendations.push_back("Address high-severity issues - they impact reliability and performance");
    }

    if (result.overall_quality_score < 0.7f) {
        recommendations.push_back("Consider comprehensive code refactoring to improve maintainability");
    }

    auto memory_count = result.issue_counts_by_type.find("memory");
    auto logic_count = result.issue_counts_by_type.find("logic");
    int mem_count = memory_count != result.issue_counts_by_type.end() ? memory_count->second : 0;
    int log_count = logic_count != result.issue_counts_by_type.end() ? logic_count->second : 0;

    if (mem_count > log_count) {
        recommendations.push_back("Focus on memory management improvements - memory issues are most prevalent");
    }

    return recommendations;
}

// Code modification creation methods
CodeModification CodeAnalysisSystem::create_memory_fix(const CodeIssue& issue) {
    CodeModification mod;
    mod.file_path = issue.file_path;
    mod.start_line = issue.line_number;
    mod.end_line = issue.line_number;
    mod.original_code = ""; // Would need to extract from file
    mod.reason = issue.description;
    mod.confidence_score = 0.7f;

    if (issue.description.find("null pointer") != std::string::npos) {
        mod.modified_code = "// TODO: Add null check before dereferencing";
    } else if (issue.description.find("memory leak") != std::string::npos) {
        mod.modified_code = "// TODO: Add corresponding delete for new allocation";
    }

    return mod;
}

CodeModification CodeAnalysisSystem::create_logic_fix(const CodeIssue& issue) {
    CodeModification mod;
    mod.file_path = issue.file_path;
    mod.start_line = issue.line_number;
    mod.end_line = issue.line_number;
    mod.original_code = "";
    mod.reason = issue.description;
    mod.confidence_score = 0.6f;

    if (issue.description.find("uninitialized") != std::string::npos) {
        mod.modified_code = "// TODO: Initialize variable at declaration";
    }

    return mod;
}

CodeModification CodeAnalysisSystem::create_performance_fix(const CodeIssue& issue) {
    CodeModification mod;
    mod.file_path = issue.file_path;
    mod.start_line = issue.line_number;
    mod.end_line = issue.line_number;
    mod.original_code = "";
    mod.reason = issue.description;
    mod.confidence_score = 0.8f;

    if (issue.description.find("string concatenation") != std::string::npos) {
        mod.modified_code = "// TODO: Use std::stringstream for efficient string building";
    }

    return mod;
}

CodeModification CodeAnalysisSystem::create_security_fix(const CodeIssue& issue) {
    CodeModification mod;
    mod.file_path = issue.file_path;
    mod.start_line = issue.line_number;
    mod.end_line = issue.line_number;
    mod.original_code = "";
    mod.reason = issue.description;
    mod.confidence_score = 0.9f;

    if (issue.description.find("dangerous function") != std::string::npos) {
        mod.modified_code = "// TODO: Replace with safe alternative (e.g., strcpy_s, std::string)";
    }

    return mod;
}

CodeModification CodeAnalysisSystem::create_style_fix(const CodeIssue& issue) {
    CodeModification mod;
    mod.file_path = issue.file_path;
    mod.start_line = issue.line_number;
    mod.end_line = issue.line_number;
    mod.original_code = "";
    mod.reason = issue.description;
    mod.confidence_score = 0.5f;

    if (issue.description.find("const qualifier") != std::string::npos) {
        mod.modified_code = "// TODO: Add const qualifier to reference parameter";
    }

    return mod;
}