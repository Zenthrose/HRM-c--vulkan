set(SYSTEM_SOURCES
    cloud_storage_manager.cpp
    cloud_storage_manager.hpp
    hardware_abstraction_layer.cpp
    hardware_abstraction_layer.hpp
    hardware_profiler.cpp
    hardware_profiler.hpp
    hybrid_execution_engine.cpp
    hybrid_execution_engine.hpp
    idle_time_repair_scheduler.cpp
    idle_time_repair_scheduler.hpp
    memory_compaction_system.cpp
    memory_compaction_system.hpp
    resource_monitor.cpp
    resource_monitor.hpp
    sandbox_manager.cpp
    sandbox_manager.hpp
    system_integration_manager.cpp
    system_integration_manager.hpp
    task_manager.cpp
    task_manager.hpp
)

if(WIN32)
    list(APPEND SYSTEM_SOURCES
        windows_service_manager.cpp
        windows_service_manager.hpp
    )
endif()

add_library(system STATIC ${SYSTEM_SOURCES})
target_include_directories(system PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../self_mod)