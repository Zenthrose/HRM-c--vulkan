add_subdirectory(core)
add_subdirectory(hrm)
add_subdirectory(vulkan)
add_subdirectory(self_mod)
add_subdirectory(system)
add_subdirectory(training)
add_subdirectory(utils)

# Main Vulkan test executable
add_executable(hrm_vulkan main/main.cpp)
target_link_libraries(hrm_vulkan PRIVATE utils core hrm vulkan self_mod system ${Vulkan_LIBRARIES})
target_compile_definitions(hrm_vulkan PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(hrm_vulkan PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(hrm_vulkan PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(hrm_vulkan PRIVATE pthread dl)
endif()

# HRM system executable with CLI and GUI
add_executable(hrm_system main/hrm_main.cpp main/hrm_cli.cpp main/hrm_gui.cpp)
target_link_libraries(hrm_system PRIVATE utils core hrm vulkan self_mod system training ${Vulkan_LIBRARIES})
target_compile_definitions(hrm_system PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(hrm_system PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(hrm_system PRIVATE stdc++fs)
        find_program(LIBGCC libgcc_s_seh-1.dll HINTS $ENV{MINGW_PATH}/bin C:/msys64/mingw64/bin)
        find_program(LIBSTDCXX libstdc++-6.dll HINTS $ENV{MINGW_PATH}/bin C:/msys64/mingw64/bin)
        find_program(LIBWINPTHREAD libwinpthread-1.dll HINTS $ENV{MINGW_PATH}/bin C:/msys64/mingw64/bin)
    endif()
elseif(UNIX)
    target_link_libraries(hrm_system PRIVATE pthread dl)
endif()

add_custom_command(TARGET hrm_vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_vulkan>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/attention.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/attention.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/embedding.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/embedding.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/linear.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/linear.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/rms_norm.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/swiglu.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/swiglu.spv

)

add_custom_command(TARGET hrm_system POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_system>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/attention.spv $<TARGET_FILE_DIR:hrm_system>/shaders/attention.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/embedding.spv $<TARGET_FILE_DIR:hrm_system>/shaders/embedding.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/linear.spv $<TARGET_FILE_DIR:hrm_system>/shaders/linear.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/linear_backward.spv $<TARGET_FILE_DIR:hrm_system>/shaders/linear_backward.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/adam_optimizer.spv $<TARGET_FILE_DIR:hrm_system>/shaders/adam_optimizer.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/cross_entropy_loss.spv $<TARGET_FILE_DIR:hrm_system>/shaders/cross_entropy_loss.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/gradient_accumulation.spv $<TARGET_FILE_DIR:hrm_system>/shaders/gradient_accumulation.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv $<TARGET_FILE_DIR:hrm_system>/shaders/rms_norm.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/swiglu.spv $<TARGET_FILE_DIR:hrm_system>/shaders/swiglu.spv

)

add_dependencies(hrm_vulkan compile_shaders)

# Evaluation tool
add_executable(evaluate evaluate.cpp)
target_link_libraries(evaluate PRIVATE utils core hrm vulkan self_mod system training Vulkan::Vulkan)
target_compile_definitions(evaluate PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(evaluate PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(evaluate PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(evaluate PRIVATE pthread dl)
endif()

# Test data generation tool
add_executable(generate_test_data generate_test_data.cpp)
target_link_libraries(generate_test_data PRIVATE core Vulkan::Vulkan)
target_compile_definitions(generate_test_data PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(generate_test_data PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(generate_test_data PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(generate_test_data PRIVATE pthread dl)
endif()

# ARC evaluation tool
add_executable(arc_eval arc_eval.cpp)
target_link_libraries(arc_eval PRIVATE utils core Vulkan::Vulkan)
target_compile_definitions(arc_eval PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(arc_eval PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(arc_eval PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(arc_eval PRIVATE pthread dl)
endif()

# Copy assets to build directory for portable GUI deployment
add_custom_command(TARGET hrm_system POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_system>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:hrm_system>/assets
)
