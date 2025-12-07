add_library(vulkan STATIC
    flash_attention.cpp
    flash_attention.hpp
    mixed_precision.cpp
    mixed_precision.hpp
    vulkan_compatibility.cpp
    vulkan_compatibility.h
    vulkan_resource_manager.cpp
    vulkan_resource_manager.hpp
    vulkan_trainer.cpp
    vulkan_trainer.hpp
)
target_include_directories(vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../core)