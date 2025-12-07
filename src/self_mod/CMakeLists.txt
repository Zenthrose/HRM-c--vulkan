add_library(self_mod STATIC
    code_analysis_system.cpp
    code_analysis_system.hpp
    runtime_compilation_system.cpp
    runtime_compilation_system.hpp
    self_evolving_hrm.cpp
    self_evolving_hrm.hpp
    self_modifying_hrm.cpp
    self_modifying_hrm.hpp
    self_repair_system.cpp
    self_repair_system.hpp
)
target_include_directories(self_mod PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../hrm ../utils)
target_link_libraries(self_mod PRIVATE hrm utils)