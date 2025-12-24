add_library(vulkan STATIC
    adaptive_quantization.cpp
    adaptive_quantization.hpp
    flash_attention.cpp
    flash_attention.hpp
    mixed_precision.cpp
    mixed_precision.hpp
    vulkan_compatibility.cpp
    vulkan_compatibility.h
    vulkan_instance_manager.hpp
    vulkan_resource_manager.cpp
    vulkan_resource_manager.hpp
    vulkan_trainer.cpp
    vulkan_trainer.hpp
)
target_include_directories(vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../core)
target_compile_definitions(vulkan PRIVATE SHADER_DIR=\"${CMAKE_BINARY_DIR}/shaders\")

# CPU quantization unit test (small, standalone)
add_executable(quantization_cpu_test quantization_cpu_test.cpp)
target_link_libraries(quantization_cpu_test PRIVATE vulkan utils core)
add_test(NAME quantization_cpu_test COMMAND quantization_cpu_test)