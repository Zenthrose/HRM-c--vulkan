# Build Commands
- C++ Build (Linux/macOS): `mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`
- C++ Build (Windows): `mkdir build && cd build && cmake .. && cmake --build . --config Release`
- Python Install: `pip install -r requirements.txt`
- Test Suite (Linux/macOS): `./test_hrm_system.sh`
- Test Suite (Windows): `.\test_hrm_system.bat` (basic Windows compatibility added)

# Windows Compatibility Status
- ✅ CMake configuration works
- ✅ Vulkan SDK detection works
- ✅ Shader compilation works
- ✅ Basic Windows API integration added
- ✅ System monitoring implemented (memory/disk/CPU)
- ✅ GUI terminal functions use Windows Console API
- ✅ Dynamic library loading (LoadLibrary/FreeLibrary)
- ✅ Directory operations (FindFirstFile/FindNextFile)
- ✅ Process execution (_popen/_pclose)
- ✅ Cross-platform conditional compilation
- ✅ Type conversion warnings fixed (double→float, size_t→int)
- ✅ Security warnings fixed (localtime_s, _dupenv_s)
- ✅ Unreferenced variable warnings fixed
- ⚠️ Network monitoring simplified (basic implementation)
- ⚠️ Minor STL warnings remain (intentional conversions)
- ✅ Build completes successfully with minimal warnings

# Test Commands
- Run all tests (Linux/macOS): `./test_hrm_system.sh`
- Run all tests (Windows): `.\test_hrm_system.bat`
- Run single C++ test: `./src/hrm_system --test` (or `.\src\Release\hrm_system.exe --test` on Windows)
- Python evaluation: `python evaluate.py checkpoint=<path>`

# Code Style Guidelines

## C++ Style
- Use C++17 standard with `<iostream>`, `<vector>`, `<string>` etc.
- Include order: standard library, then local headers with quotes
- Naming: PascalCase for classes/types, camelCase for functions/variables
- Error handling: throw `std::runtime_error` with descriptive messages
- Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw pointers
- RAII pattern for resource management

## Python Style
- Type hints required for function parameters and return values
- Import order: standard library, third-party packages, local modules
- Naming: snake_case for functions/variables, PascalCase for classes
- Use f-strings for string formatting
- Exception handling with specific exception types
- Docstrings for public functions and classes

## General
- No comments unless explaining complex logic
- Consistent indentation (4 spaces for Python, tabs for C++)
- Line length: 100 characters maximum
- Use meaningful variable names over abbreviations</content>
<parameter name="filePath">C:\HRM-c--vulkan-main\AGENTS.md