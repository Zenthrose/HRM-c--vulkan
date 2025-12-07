add_library(core STATIC
    attention.cpp
    attention.hpp
    block.cpp
    block.hpp
    embedding.cpp
    embedding.hpp
    linear.cpp
    linear.hpp
    losses.cpp
    losses.hpp
    mlp.cpp
    mlp.hpp
    rms_norm.cpp
    rms_norm.hpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT NO_VULKAN)
target_link_libraries(core PRIVATE utils Vulkan::Vulkan)
else()
target_link_libraries(core PRIVATE utils)
endif()