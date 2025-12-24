# Physics module
add_library(physics
    mcmc_simulator.cpp
)

target_include_directories(physics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(physics PRIVATE
    utils
)