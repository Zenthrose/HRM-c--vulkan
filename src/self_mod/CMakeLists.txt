add_library(self_mod STATIC
    code_analysis_system.cpp
    code_analysis_system.hpp
    meta_quantization_learning.cpp
    meta_quantization_learning.hpp
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

# Small smoke test for quantization strategy storage (no Vulkan required)
add_executable(quantization_smoke_test quantization_smoke_test.cpp)
target_include_directories(quantization_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../vulkan ../hrm ../utils)
target_link_libraries(quantization_smoke_test PRIVATE self_mod utils hrm vulkan)
add_dependencies(quantization_smoke_test self_mod)

# Register smoke test with CTest
enable_testing()
if(NOT TARGET run_quantization_smoke_test)
    add_test(NAME quantization_smoke_test COMMAND quantization_smoke_test)
endif()