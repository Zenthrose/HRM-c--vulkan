add_library(hrm STATIC
    hierarchical_reasoning_module.cpp
    hierarchical_reasoning_module.hpp
    hrm.cpp
    hrm.hpp
    hrm_inner.cpp
    hrm_inner.hpp
    meta_reasoning_layer.cpp
    meta_reasoning_layer.hpp
    resource_aware_hrm.cpp
    resource_aware_hrm.hpp
    idle_learning_manager.cpp
    idle_learning_manager.hpp
)
target_include_directories(hrm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../core)
target_link_libraries(hrm PRIVATE core)