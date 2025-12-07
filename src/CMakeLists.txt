# Main Vulkan test executable
add_executable(hrm_vulkan main/main.cpp core/attention.cpp core/attention.hpp core/mlp.cpp core/mlp.hpp core/rms_norm.cpp core/rms_norm.hpp core/block.cpp core/block.hpp core/linear.cpp core/linear.hpp core/embedding.cpp core/embedding.hpp hrm/hierarchical_reasoning_module.cpp hrm/hierarchical_reasoning_module.hpp hrm/hrm_inner.cpp hrm/hrm_inner.hpp hrm/hrm.cpp hrm/hrm.hpp core/losses.cpp core/losses.hpp utils/utf8_processor.cpp utils/utf8_processor.hpp hrm/meta_reasoning_layer.cpp hrm/meta_reasoning_layer.hpp self_mod/self_evolving_hrm.cpp self_mod/self_evolving_hrm.hpp self_mod/code_analysis_system.cpp self_mod/code_analysis_system.hpp self_mod/runtime_compilation_system.cpp self_mod/runtime_compilation_system.hpp self_mod/self_modifying_hrm.cpp self_mod/self_modifying_hrm.hpp system/resource_monitor.cpp system/resource_monitor.hpp system/task_manager.cpp system/task_manager.hpp hrm/resource_aware_hrm.cpp hrm/resource_aware_hrm.hpp system/hardware_abstraction_layer.cpp system/hardware_abstraction_layer.hpp system/idle_time_repair_scheduler.cpp system/idle_time_repair_scheduler.hpp system/system_integration_manager.cpp system/system_integration_manager.hpp system/sandbox_manager.cpp system/sandbox_manager.hpp vulkan/vulkan_trainer.cpp vulkan/vulkan_trainer.hpp)
target_link_libraries(hrm_vulkan PRIVATE Vulkan::Vulkan)
target_compile_definitions(hrm_vulkan PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(hrm_vulkan PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(hrm_vulkan PRIVATE stdc++fs)
    endif()
endif()

# HRM system executable with CLI and GUI
add_executable(hrm_system main/hrm_main.cpp main/hrm_cli.cpp main/hrm_gui.cpp system/memory_compaction_system.cpp system/cloud_storage_manager.cpp hrm/resource_aware_hrm.cpp system/resource_monitor.cpp system/task_manager.cpp self_mod/self_modifying_hrm.cpp self_mod/self_evolving_hrm.cpp self_mod/code_analysis_system.cpp self_mod/runtime_compilation_system.cpp system/hardware_abstraction_layer.cpp system/idle_time_repair_scheduler.cpp system/system_integration_manager.cpp system/hardware_profiler.cpp system/hardware_profiler.hpp system/sandbox_manager.cpp system/sandbox_manager.hpp hrm/hrm.cpp hrm/hrm_inner.cpp hrm/hierarchical_reasoning_module.cpp hrm/meta_reasoning_layer.cpp utils/utf8_processor.cpp core/losses.cpp core/attention.cpp core/mlp.cpp core/rms_norm.cpp core/block.cpp core/linear.cpp core/embedding.cpp training/character_language_trainer.cpp training/character_language_config.hpp training/character_text_dataset.cpp training/character_text_dataset.hpp training/character_language_loss.cpp training/character_language_loss.hpp training/character_language_evaluator.cpp training/character_language_evaluator.hpp training/advanced_training_optimizations.hpp vulkan/flash_attention.cpp vulkan/flash_attention.hpp vulkan/mixed_precision.cpp vulkan/mixed_precision.hpp vulkan/vulkan_trainer.cpp vulkan/vulkan_trainer.hpp)
target_link_libraries(hrm_system PRIVATE Vulkan::Vulkan)
target_compile_definitions(hrm_system PRIVATE PORTABLE_BUILD)
if(WIN32)
    target_link_libraries(hrm_system PRIVATE ws2_32 iphlpapi pdh)
    if(MINGW)
        target_link_libraries(hrm_system PRIVATE stdc++fs)
        find_program(LIBGCC libgcc_s_seh-1.dll HINTS $ENV{MINGW_PATH}/bin C:/msys64/mingw64/bin)
        find_program(LIBSTDCXX libstdc++-6.dll HINTS $ENV{MINGW_PATH}/bin C:/msys64/mingw64/bin)
        find_program(LIBWINPTHREAD libwinpthread-1.dll HINTS $ENV{MINGW_PATH}/bin C:/msys64/mingw64/bin)
    endif()
endif()

add_custom_command(TARGET hrm_vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_vulkan>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/attention.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/attention.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/embedding.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/embedding.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/linear.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/linear.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/rms_norm.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/shaders/swiglu.spv $<TARGET_FILE_DIR:hrm_vulkan>/shaders/swiglu.spv
    $<$<PLATFORM_ID:Windows>:$<$<BOOL:${MINGW}>:COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBGCC} $<TARGET_FILE_DIR:hrm_vulkan>/libgcc_s_seh-1.dll>>
    $<$<PLATFORM_ID:Windows>:$<$<BOOL:${MINGW}>:COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBSTDCXX} $<TARGET_FILE_DIR:hrm_vulkan>/libstdc++-6.dll>>
    $<$<PLATFORM_ID:Windows>:$<$<BOOL:${MINGW}>:COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBWINPTHREAD} $<TARGET_FILE_DIR:hrm_vulkan>/libwinpthread-1.dll>>
)

add_custom_command(TARGET hrm_system POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_system>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/attention.spv $<TARGET_FILE_DIR:hrm_system>/shaders/attention.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/embedding.spv $<TARGET_FILE_DIR:hrm_system>/shaders/embedding.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/linear.spv $<TARGET_FILE_DIR:hrm_system>/shaders/linear.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/linear_backward.spv $<TARGET_FILE_DIR:hrm_system>/shaders/linear_backward.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/adam_optimizer.spv $<TARGET_FILE_DIR:hrm_system>/shaders/adam_optimizer.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/cross_entropy_loss.spv $<TARGET_FILE_DIR:hrm_system>/shaders/cross_entropy_loss.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/gradient_accumulation.spv $<TARGET_FILE_DIR:hrm_system>/shaders/gradient_accumulation.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/rms_norm.spv $<TARGET_FILE_DIR:hrm_system>/shaders/rms_norm.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/swiglu.spv $<TARGET_FILE_DIR:hrm_system>/shaders/swiglu.spv
    $<$<PLATFORM_ID:Windows>:$<$<BOOL:${MINGW}>:COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBGCC} $<TARGET_FILE_DIR:hrm_system>/libgcc_s_seh-1.dll>>
    $<$<PLATFORM_ID:Windows>:$<$<BOOL:${MINGW}>:COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBSTDCXX} $<TARGET_FILE_DIR:hrm_system>/libstdc++-6.dll>>
    $<$<PLATFORM_ID:Windows>:$<$<BOOL:${MINGW}>:COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBWINPTHREAD} $<TARGET_FILE_DIR:hrm_system>/libwinpthread-1.dll>>
)

add_dependencies(hrm_vulkan compile_shaders)
