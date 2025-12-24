# CAD module
add_library(cad
    freecad_interface.cpp
)

target_include_directories(cad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cad PRIVATE
    utils
)