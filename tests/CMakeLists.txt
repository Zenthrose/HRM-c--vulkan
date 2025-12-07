# Tests CMakeLists.txt - GTest integration for HRM system

# Find GTest package (optional)
find_package(GTest)
if(GTest_FOUND)
    include(GoogleTest)
else()
    message(WARNING "GTest not found. Tests will not be built. Install GTest to enable automated testing.")
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
    GTest::gmock_main
    core hrm vulkan self_mod system training utils
    Vulkan::Vulkan
)
target_compile_definitions(self_mod_test PRIVATE PORTABLE_BUILD)

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