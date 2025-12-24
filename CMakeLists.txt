cmake_minimum_required(VERSION 3.10)
project(HRM_Vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")  # Commented out - causes header issues

# Enable CTest so `add_test` registrations in subdirectories are visible to `ctest`
include(CTest)
enable_testing()
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CPU feature detection
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Check for AVX support
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    add_definitions(-D__AVX__)
endif()

# Check for AVX2 support
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    add_definitions(-D__AVX2__)
endif()

# Check for NEON support (ARM)
check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
if(COMPILER_SUPPORTS_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    add_definitions(-D__ARM_NEON)
endif()

# Set output directory for executables to build root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release)

# Vulkan support (optional)
option(NO_VULKAN "Build without Vulkan support" OFF)
if(NOT NO_VULKAN)
    find_package(Vulkan REQUIRED)
    include_directories(${Vulkan_INCLUDE_DIRS})
else()
    message(STATUS "Building without Vulkan support")
    add_definitions(-DNO_VULKAN)
endif()

# Standard library includes (fix for header resolution issues)
include_directories(SYSTEM /usr/include/c++/13)
include_directories(SYSTEM /usr/include)

# Qt GUI support (optional) - DISABLED for now
option(NO_GUI "Build without Qt GUI support" ON)
if(NOT NO_GUI)
    # Try Qt6 first, then Qt5
    find_package(Qt6 QUIET COMPONENTS Core Widgets)
    if(Qt6_FOUND)
        set(QT_VERSION 6)
        set(QT_LIBRARIES Qt6::Core Qt6::Widgets)
        set(Qt${QT_VERSION}_INCLUDE_DIRS ${Qt6_INCLUDE_DIRS})
        set(Qt${QT_VERSION}_DEFINITIONS ${Qt6_DEFINITIONS})
    else()
        find_package(Qt5 QUIET COMPONENTS Core Widgets)
        if(Qt5_FOUND)
            set(QT_VERSION 5)
            set(QT_LIBRARIES Qt5::Core Qt5::Widgets)
            set(Qt${QT_VERSION}_INCLUDE_DIRS ${Qt5_INCLUDE_DIRS})
            set(Qt${QT_VERSION}_DEFINITIONS ${Qt5_DEFINITIONS})
        else()
            message(WARNING "Qt not found - building without GUI support")
            set(NO_GUI ON)
        endif()
    endif()

    if(NOT NO_GUI)
        include_directories(${Qt${QT_VERSION}_INCLUDE_DIRS})
        add_definitions(${Qt${QT_VERSION}_DEFINITIONS})
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTORCC ON)
        set(CMAKE_AUTOUIC ON)
        message(STATUS "Building with Qt${QT_VERSION} GUI support")
    endif()
else()
    message(STATUS "Building without Qt GUI support")
    add_definitions(-DNO_GUI)
endif()

# Find spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    # If not found, try pkg-config
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(SPDLOG spdlog)
        if(SPDLOG_FOUND)
            include_directories(${SPDLOG_INCLUDE_DIRS})
            link_directories(${SPDLOG_LIBRARY_DIRS})
            set(SPDLOG_LIBRARIES ${SPDLOG_LIBRARIES})
        endif()
    endif()
endif()

if(NOT NO_VULKAN)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

    # Compile shaders to SPIR-V
    find_program(GLSLC_COMPILER glslc)
    if(NOT GLSLC_COMPILER)
        message(FATAL_ERROR "glslc compiler not found! Please ensure Vulkan SDK is installed and glslc is in your PATH.")
    endif()
endif()

# Ensure C++ standard library is properly available
# Let compiler handle standard library includes automatically

add_subdirectory(src)
add_subdirectory(dataset)
add_subdirectory(tests)

if(NOT NO_VULKAN)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

    # Compile shaders to SPIR-V
    find_program(GLSLC_COMPILER glslc)
    if(NOT GLSLC_COMPILER)
        message(FATAL_ERROR "glslc compiler not found! Please ensure Vulkan SDK is installed and glslc is in your PATH.")
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/shaders/attention.spv
        COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/attention.spv ${CMAKE_SOURCE_DIR}/shaders/attention.comp
        DEPENDS ${CMAKE_SOURCE_DIR}/shaders/attention.comp
        COMMENT "Compiling attention.comp to SPIR-V"
        VERBATIM
    )

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/swiglu.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/swiglu.spv ${CMAKE_SOURCE_DIR}/shaders/swiglu.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/swiglu.comp
    COMMENT "Compiling swiglu.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv ${CMAKE_SOURCE_DIR}/shaders/rms_norm.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/rms_norm.comp
    COMMENT "Compiling rms_norm.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/linear.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/linear.spv ${CMAKE_SOURCE_DIR}/shaders/linear.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/linear.comp
    COMMENT "Compiling linear.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/embedding.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/embedding.spv ${CMAKE_SOURCE_DIR}/shaders/embedding.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/embedding.comp
    COMMENT "Compiling embedding.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/linear_backward.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/linear_backward.spv ${CMAKE_SOURCE_DIR}/shaders/linear_backward.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/linear_backward.comp
    COMMENT "Compiling linear_backward.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/adam_optimizer.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/adam_optimizer.spv ${CMAKE_SOURCE_DIR}/shaders/adam_optimizer.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/adam_optimizer.comp
    COMMENT "Compiling adam_optimizer.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/cross_entropy_loss.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/cross_entropy_loss.spv ${CMAKE_SOURCE_DIR}/shaders/cross_entropy_loss.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/cross_entropy_loss.comp
    COMMENT "Compiling cross_entropy_loss.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/gradient_accumulation.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/gradient_accumulation.spv ${CMAKE_SOURCE_DIR}/shaders/gradient_accumulation.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/gradient_accumulation.comp
    COMMENT "Compiling gradient_accumulation.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/rope.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/rope.spv ${CMAKE_SOURCE_DIR}/shaders/rope.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/rope.comp
    COMMENT "Compiling rope.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/quant_linear_int8.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/quant_linear_int8.spv ${CMAKE_SOURCE_DIR}/shaders/quant_linear_int8.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/quant_linear_int8.comp
    COMMENT "Compiling quant_linear_int8.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/quant_linear_int4.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/quant_linear_int4.spv ${CMAKE_SOURCE_DIR}/shaders/quant_linear_int4.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/quant_linear_int4.comp
    COMMENT "Compiling quant_linear_int4.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/quant_linear_fp16.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/quant_linear_fp16.spv ${CMAKE_SOURCE_DIR}/shaders/quant_linear_fp16.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/quant_linear_fp16.comp
    COMMENT "Compiling quant_linear_fp16.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/quant_linear_bf16.spv
    COMMAND ${GLSLC_COMPILER} --target-env=vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/quant_linear_bf16.spv ${CMAKE_SOURCE_DIR}/shaders/quant_linear_bf16.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/quant_linear_bf16.comp
    COMMENT "Compiling quant_linear_bf16.comp to SPIR-V"
    VERBATIM
)

    # Generate basic C++ headers for SPIR-V files (simplified)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/shaders/attention.h
        COMMAND ${CMAKE_COMMAND} -E echo "#pragma once" > ${CMAKE_BINARY_DIR}/shaders/attention.h &&
                ${CMAKE_COMMAND} -E echo "const unsigned char attention_spv[] = {0};" >> ${CMAKE_BINARY_DIR}/shaders/attention.h &&
                ${CMAKE_COMMAND} -E echo "const size_t attention_spv_size = 0;" >> ${CMAKE_BINARY_DIR}/shaders/attention.h
        DEPENDS ${CMAKE_BINARY_DIR}/shaders/attention.spv
        COMMENT "Generating attention.h header"
    )

    # Add more shader headers as needed...

    add_custom_target(compile_shaders ALL
        DEPENDS ${CMAKE_BINARY_DIR}/shaders/attention.spv ${CMAKE_BINARY_DIR}/shaders/swiglu.spv ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv ${CMAKE_BINARY_DIR}/shaders/linear.spv ${CMAKE_BINARY_DIR}/shaders/embedding.spv ${CMAKE_BINARY_DIR}/shaders/linear_backward.spv ${CMAKE_BINARY_DIR}/shaders/adam_optimizer.spv ${CMAKE_BINARY_DIR}/shaders/cross_entropy_loss.spv ${CMAKE_BINARY_DIR}/shaders/gradient_accumulation.spv ${CMAKE_BINARY_DIR}/shaders/rope.spv
        ${CMAKE_BINARY_DIR}/shaders/attention.h
    )
endif()

# Copy assets to build directory
file(GLOB ASSETS assets/*)
foreach(ASSET ${ASSETS})
    get_filename_component(ASSET_NAME ${ASSET} NAME)
    configure_file(${ASSET} ${CMAKE_BINARY_DIR}/assets/${ASSET_NAME} COPYONLY)
endforeach()

# Windows NSIS Installer Configuration
set(CPACK_PACKAGE_NAME "HRM")
set(CPACK_PACKAGE_VENDOR "Your Organization")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Hierarchical Reasoning Module")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "HRM")
set(CPACK_PACKAGE_EXECUTABLES "hrm_system" "HRM System")
set(CPACK_CREATE_DESKTOP_LINKS "hrm_system")

# NSIS Specific Settings
set(CPACK_GENERATOR "NSIS")
set(CPACK_NSIS_DISPLAY_NAME "HRM - Hierarchical Reasoning Module")
set(CPACK_NSIS_PACKAGE_NAME "HRM")
set(CPACK_NSIS_MODIFY_PATH ON)  # Allow PATH modification for Vulkan
set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/Zenthrose/HRM-c--vulkan")
set(CPACK_NSIS_CONTACT "your@email.com")

# License
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

# Installation Rules
install(TARGETS nyx_system evaluate generate_test_data arc_eval
    RUNTIME DESTINATION bin
    COMPONENT Runtime)

# Install DLLs
install(FILES
    ${CMAKE_BINARY_DIR}/release/libgcc_s_seh-1.dll
    ${CMAKE_BINARY_DIR}/release/libstdc++-6.dll
    ${CMAKE_BINARY_DIR}/release/libwinpthread-1.dll
    ${CMAKE_BINARY_DIR}/release/libspdlog-1.15.dll
    ${CMAKE_BINARY_DIR}/release/libfmt-12.dll
    ${CMAKE_BINARY_DIR}/release/vulkan-1.dll
    DESTINATION bin
    COMPONENT Runtime)

# Install assets and shaders
install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets/ DESTINATION assets COMPONENT Runtime)
install(DIRECTORY ${CMAKE_BINARY_DIR}/shaders/ DESTINATION shaders COMPONENT Runtime)

# Install documentation
install(FILES ${CMAKE_SOURCE_DIR}/README.md ${CMAKE_SOURCE_DIR}/LICENSE
    DESTINATION docs COMPONENT Runtime)

include(CPack)
