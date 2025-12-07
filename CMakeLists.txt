cmake_minimum_required(VERSION 3.10)
project(HRM_Vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
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

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

add_subdirectory(src)
add_subdirectory(dataset)

# Compile shaders to SPIR-V
find_program(GLSLC_COMPILER glslc)
if(NOT GLSLC_COMPILER)
    message(FATAL_ERROR "glslc compiler not found! Please ensure Vulkan SDK is installed and glslc is in your PATH.")
endif()

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/attention.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/attention.spv ${CMAKE_SOURCE_DIR}/shaders/attention.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/attention.comp
    COMMENT "Compiling attention.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/swiglu.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/swiglu.spv ${CMAKE_SOURCE_DIR}/shaders/swiglu.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/swiglu.comp
    COMMENT "Compiling swiglu.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv ${CMAKE_SOURCE_DIR}/shaders/rms_norm.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/rms_norm.comp
    COMMENT "Compiling rms_norm.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/linear.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/linear.spv ${CMAKE_SOURCE_DIR}/shaders/linear.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/linear.comp
    COMMENT "Compiling linear.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/embedding.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/embedding.spv ${CMAKE_SOURCE_DIR}/shaders/embedding.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/embedding.comp
    COMMENT "Compiling embedding.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/linear_backward.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/linear_backward.spv ${CMAKE_SOURCE_DIR}/shaders/linear_backward.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/linear_backward.comp
    COMMENT "Compiling linear_backward.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/adam_optimizer.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/adam_optimizer.spv ${CMAKE_SOURCE_DIR}/shaders/adam_optimizer.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/adam_optimizer.comp
    COMMENT "Compiling adam_optimizer.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/cross_entropy_loss.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/cross_entropy_loss.spv ${CMAKE_SOURCE_DIR}/shaders/cross_entropy_loss.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/cross_entropy_loss.comp
    COMMENT "Compiling cross_entropy_loss.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/gradient_accumulation.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/gradient_accumulation.spv ${CMAKE_SOURCE_DIR}/shaders/gradient_accumulation.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/gradient_accumulation.comp
    COMMENT "Compiling gradient_accumulation.comp to SPIR-V"
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/rope.spv
    COMMAND ${GLSLC_COMPILER} --target-env vulkan1.3 -O -o ${CMAKE_BINARY_DIR}/shaders/rope.spv ${CMAKE_SOURCE_DIR}/shaders/rope.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/rope.comp
    COMMENT "Compiling rope.comp to SPIR-V"
    VERBATIM
)

# Generate C++ headers from SPIR-V files for embedding
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/attention.h
    COMMAND ${CMAKE_COMMAND} -E echo "#pragma once" > ${CMAKE_BINARY_DIR}/shaders/attention.h
    COMMAND ${CMAKE_COMMAND} -E echo "const unsigned char attention_spv[] = {" >> ${CMAKE_BINARY_DIR}/shaders/attention.h
    COMMAND ${CMAKE_COMMAND} -E echo "    // SPIR-V bytecode would be embedded here" >> ${CMAKE_BINARY_DIR}/shaders/attention.h
    COMMAND ${CMAKE_COMMAND} -E echo "};" >> ${CMAKE_BINARY_DIR}/shaders/attention.h
    COMMAND ${CMAKE_COMMAND} -E echo "const size_t attention_spv_size = sizeof(attention_spv);" >> ${CMAKE_BINARY_DIR}/shaders/attention.h
    DEPENDS ${CMAKE_BINARY_DIR}/shaders/attention.spv
    COMMENT "Generating attention.h header"
)

# Add more shader headers as needed...

add_custom_target(compile_shaders ALL
    DEPENDS ${CMAKE_BINARY_DIR}/shaders/attention.spv ${CMAKE_BINARY_DIR}/shaders/swiglu.spv ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv ${CMAKE_BINARY_DIR}/shaders/linear.spv ${CMAKE_BINARY_DIR}/shaders/embedding.spv ${CMAKE_BINARY_DIR}/shaders/linear_backward.spv ${CMAKE_BINARY_DIR}/shaders/adam_optimizer.spv ${CMAKE_BINARY_DIR}/shaders/cross_entropy_loss.spv ${CMAKE_BINARY_DIR}/shaders/gradient_accumulation.spv ${CMAKE_BINARY_DIR}/shaders/rope.spv
    ${CMAKE_BINARY_DIR}/shaders/attention.h
)

# Copy assets to build directory
file(GLOB ASSETS assets/*)
foreach(ASSET ${ASSETS})
    get_filename_component(ASSET_NAME ${ASSET} NAME)
    configure_file(${ASSET} ${CMAKE_BINARY_DIR}/assets/${ASSET_NAME} COPYONLY)
endforeach()

