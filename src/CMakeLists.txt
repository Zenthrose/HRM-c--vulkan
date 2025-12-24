add_subdirectory(core)
add_subdirectory(hrm)
if(NOT NO_VULKAN)
add_subdirectory(vulkan)
endif()
add_subdirectory(self_mod)
add_subdirectory(system)
add_subdirectory(training)
add_subdirectory(utils)
add_subdirectory(physics)
add_subdirectory(cad)

# Main Vulkan test executable
if(NOT NO_VULKAN)
add_executable(nyx_vulkan main/main.cpp)
target_link_libraries(nyx_vulkan PRIVATE utils core hrm vulkan self_mod system Vulkan::Vulkan)
target_compile_definitions(nyx_vulkan PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(nyx_vulkan PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(nyx_vulkan PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(nyx_vulkan PRIVATE pthread dl)
endif()

add_custom_command(TARGET nyx_vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:nyx_vulkan>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/attention.spv $<TARGET_FILE_DIR:nyx_vulkan>/shaders/attention.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/embedding.spv $<TARGET_FILE_DIR:nyx_vulkan>/shaders/embedding.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/linear.spv $<TARGET_FILE_DIR:nyx_vulkan>/shaders/linear.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv $<TARGET_FILE_DIR:nyx_vulkan>/shaders/rms_norm.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/swiglu.spv $<TARGET_FILE_DIR:nyx_vulkan>/shaders/swiglu.spv

)

add_dependencies(nyx_vulkan compile_shaders)
endif()

# Nyx system executable with CLI and GUI
add_executable(nyx_system main/hrm_main.cpp main/hrm_cli.cpp main/hrm_gui.cpp)
if(NOT NO_VULKAN)
target_link_libraries(nyx_system PRIVATE utils core hrm vulkan self_mod system training physics cad Vulkan::Vulkan)
else()
target_link_libraries(nyx_system PRIVATE utils core hrm self_mod system training physics cad)
endif()
target_compile_definitions(nyx_system PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(nyx_system PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(nyx_system PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(nyx_system PRIVATE pthread dl)
endif()

# Qt GUI Application
if(NOT NO_GUI)
add_executable(nyx_gui main/nyx_gui_main.cpp)
target_link_libraries(nyx_gui PRIVATE ${QT_LIBRARIES} utils core hrm self_mod system training physics cad)
if(NOT NO_VULKAN)
    target_link_libraries(nyx_gui PRIVATE vulkan Vulkan::Vulkan)
endif()
target_compile_definitions(nyx_gui PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(nyx_gui PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(nyx_gui PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(nyx_gui PRIVATE pthread dl)
endif()
endif()

# Copy MinGW, dependency, and Vulkan DLLs for Windows portability
if(WIN32 AND MINGW)
add_custom_command(TARGET nyx_system POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libgcc_s_seh-1.dll $<TARGET_FILE_DIR:nyx_system>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libstdc++-6.dll $<TARGET_FILE_DIR:nyx_system>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libwinpthread-1.dll $<TARGET_FILE_DIR:nyx_system>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libspdlog-1.15.dll $<TARGET_FILE_DIR:nyx_system>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libfmt-12.dll $<TARGET_FILE_DIR:nyx_system>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/Windows/System32/vulkan-1.dll $<TARGET_FILE_DIR:nyx_system>
    )
endif()



# Evaluation tool
add_executable(evaluate evaluate.cpp)
if(NOT NO_VULKAN)
target_link_libraries(evaluate PRIVATE utils core hrm vulkan self_mod system training Vulkan::Vulkan)
else()
target_link_libraries(evaluate PRIVATE utils core hrm self_mod system training)
endif()
target_compile_definitions(evaluate PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(evaluate PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(evaluate PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(evaluate PRIVATE pthread dl)
endif()

# Copy MinGW, dependency, and Vulkan DLLs for Windows portability
if(WIN32 AND MINGW)
    add_custom_command(TARGET evaluate POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libgcc_s_seh-1.dll $<TARGET_FILE_DIR:evaluate>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libstdc++-6.dll $<TARGET_FILE_DIR:evaluate>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libwinpthread-1.dll $<TARGET_FILE_DIR:evaluate>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libspdlog-1.15.dll $<TARGET_FILE_DIR:evaluate>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libfmt-12.dll $<TARGET_FILE_DIR:evaluate>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/Windows/System32/vulkan-1.dll $<TARGET_FILE_DIR:evaluate>
    )
endif()

# Test data generation tool
add_executable(generate_test_data generate_test_data.cpp)
if(NOT NO_VULKAN)
target_link_libraries(generate_test_data PRIVATE core Vulkan::Vulkan)
else()
target_link_libraries(generate_test_data PRIVATE core)
endif()
target_compile_definitions(generate_test_data PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(generate_test_data PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(generate_test_data PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(generate_test_data PRIVATE pthread dl)
endif()

# Copy MinGW, dependency, and Vulkan DLLs for Windows portability
if(WIN32 AND MINGW)
    add_custom_command(TARGET generate_test_data POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libgcc_s_seh-1.dll $<TARGET_FILE_DIR:generate_test_data>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libstdc++-6.dll $<TARGET_FILE_DIR:generate_test_data>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libwinpthread-1.dll $<TARGET_FILE_DIR:generate_test_data>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libspdlog-1.15.dll $<TARGET_FILE_DIR:generate_test_data>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libfmt-12.dll $<TARGET_FILE_DIR:generate_test_data>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/Windows/System32/vulkan-1.dll $<TARGET_FILE_DIR:generate_test_data>
    )
endif()

# ARC evaluation tool
add_executable(arc_eval arc_eval.cpp)
if(NOT NO_VULKAN)
target_link_libraries(arc_eval PRIVATE utils core Vulkan::Vulkan)
else()
target_link_libraries(arc_eval PRIVATE utils core)
endif()
target_compile_definitions(arc_eval PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(arc_eval PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(arc_eval PRIVATE stdc++fs)
    endif()
elseif(UNIX)
    target_link_libraries(arc_eval PRIVATE pthread dl)
endif()

# Copy MinGW, dependency, and Vulkan DLLs for Windows portability
if(WIN32 AND MINGW)
    add_custom_command(TARGET arc_eval POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libgcc_s_seh-1.dll $<TARGET_FILE_DIR:arc_eval>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libstdc++-6.dll $<TARGET_FILE_DIR:arc_eval>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libwinpthread-1.dll $<TARGET_FILE_DIR:arc_eval>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libspdlog-1.15.dll $<TARGET_FILE_DIR:arc_eval>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/msys64/mingw64/bin/libfmt-12.dll $<TARGET_FILE_DIR:arc_eval>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different C:/Windows/System32/vulkan-1.dll $<TARGET_FILE_DIR:arc_eval>
    )
endif()

# Copy assets to build directory for portable GUI deployment
add_custom_command(TARGET nyx_system POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:nyx_system>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:nyx_system>/assets
)
