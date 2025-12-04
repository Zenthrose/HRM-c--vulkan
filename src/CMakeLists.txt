# Main Vulkan test executable
add_executable(hrm_vulkan main.cpp attention.cpp attention.hpp mlp.cpp mlp.hpp rms_norm.cpp rms_norm.hpp block.cpp block.hpp linear.cpp linear.hpp embedding.cpp embedding.hpp hierarchical_reasoning_module.cpp hierarchical_reasoning_module.hpp hrm_inner.cpp hrm_inner.hpp hrm.cpp hrm.hpp losses.cpp losses.hpp utf8_processor.cpp utf8_processor.hpp meta_reasoning_layer.cpp meta_reasoning_layer.hpp self_evolving_hrm.cpp self_evolving_hrm.hpp code_analysis_system.cpp code_analysis_system.hpp runtime_compilation_system.cpp runtime_compilation_system.hpp self_modifying_hrm.cpp self_modifying_hrm.hpp resource_monitor.cpp resource_monitor.hpp task_manager.cpp task_manager.hpp resource_aware_hrm.cpp resource_aware_hrm.hpp hardware_abstraction_layer.cpp hardware_abstraction_layer.hpp idle_time_repair_scheduler.cpp idle_time_repair_scheduler.hpp system_integration_manager.cpp system_integration_manager.hpp)
target_link_libraries(hrm_vulkan PRIVATE Vulkan::Vulkan)

# HRM system executable with CLI and GUI
add_executable(hrm_system hrm_main.cpp hrm_cli.cpp hrm_gui.cpp memory_compaction_system.cpp cloud_storage_manager.cpp resource_aware_hrm.cpp resource_monitor.cpp task_manager.cpp self_modifying_hrm.cpp self_evolving_hrm.cpp code_analysis_system.cpp runtime_compilation_system.cpp hardware_abstraction_layer.cpp idle_time_repair_scheduler.cpp system_integration_manager.cpp hrm.cpp hrm_inner.cpp hierarchical_reasoning_module.cpp meta_reasoning_layer.cpp utf8_processor.cpp losses.cpp attention.cpp mlp.cpp rms_norm.cpp block.cpp linear.cpp embedding.cpp character_language_trainer.cpp character_language_config.hpp character_text_dataset.cpp character_text_dataset.hpp character_language_loss.cpp character_language_loss.hpp character_language_evaluator.cpp character_language_evaluator.hpp advanced_training_optimizations.hpp flash_attention.cpp flash_attention.hpp mixed_precision.cpp mixed_precision.hpp)
target_link_libraries(hrm_system PRIVATE Vulkan::Vulkan)

add_custom_command(TARGET hrm_vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_vulkan>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/attention.spv
              $<TARGET_FILE_DIR:hrm_vulkan>/shaders/attention.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/swiglu.spv
              $<TARGET_FILE_DIR:hrm_vulkan>/shaders/swiglu.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv
              $<TARGET_FILE_DIR:hrm_vulkan>/shaders/rms_norm.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/linear.spv
              $<TARGET_FILE_DIR:hrm_vulkan>/shaders/linear.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/embedding.spv
              $<TARGET_FILE_DIR:hrm_vulkan>/shaders/embedding.spv
    COMMENT "Copying generated shaders to runtime directory"
)

add_custom_command(TARGET hrm_system POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:hrm_system>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/attention.spv
              $<TARGET_FILE_DIR:hrm_system>/shaders/attention.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/swiglu.spv
              $<TARGET_FILE_DIR:hrm_system>/shaders/swiglu.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/rms_norm.spv
              $<TARGET_FILE_DIR:hrm_system>/shaders/rms_norm.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/linear.spv
              $<TARGET_FILE_DIR:hrm_system>/shaders/linear.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_BINARY_DIR}/shaders/embedding.spv
              $<TARGET_FILE_DIR:hrm_system>/shaders/embedding.spv
    COMMENT "Copying generated shaders to HRM system runtime directory"
)

add_dependencies(hrm_vulkan compile_shaders)
