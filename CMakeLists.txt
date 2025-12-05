cmake_minimum_required(VERSION 3.10)
project(HRM_Vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# Set output directory for executables to build root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release)

find_package(Vulkan REQUIRED)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

add_subdirectory(src)

# Compile shaders to SPIR-V
find_program(GLSLC_COMPILER glslc)
if(NOT GLSLC_COMPILER)
    message(FATAL_ERROR "glslc compiler not found! Please ensure Vulkan SDK is installed and glslc is in your PATH.")
endif()

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/attention.spv
    COMMAND ${GLSLC_COMPILER} -o ${CMAKE_BINARY_DIR}/shaders/attention.spv ${CMAKE_SOURCE_DIR}/shaders/attention.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/attention.comp
    COMMENT "Compiling attention.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/swiglu.spv
    COMMAND ${GLSLC_COMPILER} -o ${CMAKE_BINARY_DIR}/shaders/swiglu.spv ${CMAKE_SOURCE_DIR}/shaders/swiglu.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/swiglu.comp
    COMMENT "Compiling swiglu.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv
    COMMAND ${GLSLC_COMPILER} -o ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv ${CMAKE_SOURCE_DIR}/shaders/rms_norm.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/rms_norm.comp
    COMMENT "Compiling rms_norm.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/linear.spv
    COMMAND ${GLSLC_COMPILER} -o ${CMAKE_BINARY_DIR}/shaders/linear.spv ${CMAKE_SOURCE_DIR}/shaders/linear.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/linear.comp
    COMMENT "Compiling linear.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/embedding.spv
    COMMAND ${GLSLC_COMPILER} -o ${CMAKE_BINARY_DIR}/shaders/embedding.spv ${CMAKE_SOURCE_DIR}/shaders/embedding.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/embedding.comp
    COMMENT "Compiling embedding.comp to SPIR-V"
    VERBATIM
)

add_custom_target(compile_shaders ALL
    DEPENDS ${CMAKE_BINARY_DIR}/shaders/attention.spv ${CMAKE_BINARY_DIR}/shaders/swiglu.spv ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv ${CMAKE_BINARY_DIR}/shaders/linear.spv ${CMAKE_BINARY_DIR}/shaders/embedding.spv
)

