add_library(utils STATIC
    character_cache.cpp
    character_cache.hpp
    config_manager.cpp
    config_manager.hpp
    cpu_compatibility.cpp
    cpu_compatibility.h
    http_client.cpp
    http_client.hpp
    logger.hpp
    utf8_processor.cpp
    utf8_processor.hpp
)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../core)

# Link spdlog if available
if(spdlog_FOUND)
    target_link_libraries(utils PUBLIC spdlog::spdlog)
elseif(SPDLOG_LIBRARIES)
    target_link_libraries(utils PUBLIC ${SPDLOG_LIBRARIES})
endif()