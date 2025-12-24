# Tests CMakeLists.txt - GTest integration for HRM system

# Smoke test target (always built — doesn't require GTest)
add_executable(attention_cpu_grad_smoke attention_cpu_grad_smoke.cpp)
target_link_libraries(attention_cpu_grad_smoke PRIVATE core)
target_compile_definitions(attention_cpu_grad_smoke PRIVATE PORTABLE_BUILD)
target_include_directories(attention_cpu_grad_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Copy test data for the smoke test into the binary directory
add_custom_command(TARGET attention_cpu_grad_smoke PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:attention_cpu_grad_smoke>/tests/data
    COMMAND $<TARGET_FILE:generate_test_data> --output-dir $<TARGET_FILE_DIR:attention_cpu_grad_smoke>/tests/data
    COMMENT "Generating attention test data for attention_cpu_grad_smoke using generate_test_data"
)

add_dependencies(attention_cpu_grad_smoke generate_test_data)

# Find GTest package (optional)
find_package(GTest QUIET)
if(GTest_FOUND)
    include(GoogleTest)
    message(STATUS "GTest found - building tests")
else()
    message(WARNING "GTest not found. Tests will not be built. Install libgtest-dev to enable automated testing.")
    # Return to skip building tests
    return()
endif()

# Enable testing
enable_testing()

# Basic HRM system test
add_executable(hrm_system_test hrm_system_test.cpp)
target_link_libraries(hrm_system_test PRIVATE
    GTest::gtest_main
    core hrm vulkan self_mod system training utils
    Vulkan::Vulkan
)
target_compile_definitions(hrm_system_test PRIVATE PORTABLE_BUILD)

# Self-modification integration test
add_executable(self_mod_test integration/self_mod_test.cpp)
target_link_libraries(self_mod_test PRIVATE
    GTest::gtest_main
    core hrm vulkan self_mod system training utils
    Vulkan::Vulkan
)
target_compile_definitions(self_mod_test PRIVATE PORTABLE_BUILD)

# (already declared above to ensure it is present even when GTest is absent)

# Platform-specific linking
if(WIN32)
    target_link_libraries(hrm_system_test PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(hrm_system_test PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(hrm_system_test PRIVATE pthread dl)
endif()

# Copy test data and shaders for tests
add_custom_command(TARGET hrm_system_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_system_test>/tests/data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/test_data_key.txt $<TARGET_FILE_DIR:hrm_system_test>/tests/data/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/test_data_query.txt $<TARGET_FILE_DIR:hrm_system_test>/tests/data/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/test_data_value.txt $<TARGET_FILE_DIR:hrm_system_test>/tests/data/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/test_data_output.txt $<TARGET_FILE_DIR:hrm_system_test>/tests/data/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_system_test>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/attention.spv $<TARGET_FILE_DIR:hrm_system_test>/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/embedding.spv $<TARGET_FILE_DIR:hrm_system_test>/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/linear.spv $<TARGET_FILE_DIR:hrm_system_test>/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/rms_norm.spv $<TARGET_FILE_DIR:hrm_system_test>/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/swiglu.spv $<TARGET_FILE_DIR:hrm_system_test>/shaders/
)

# Register tests with CTest
gtest_discover_tests(hrm_system_test)
gtest_discover_tests(self_mod_test)