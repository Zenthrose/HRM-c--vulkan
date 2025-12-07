add_library(utils STATIC
    config_manager.cpp
    cpu_compatibility.cpp
    cpu_compatibility.h
    http_client.cpp
    http_client.hpp
    logger.hpp
    utf8_processor.cpp
    utf8_processor.hpp
)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../core)