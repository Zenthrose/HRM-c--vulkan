#include <iostream>
#include <vector>
#include "src/cad/freecad_interface.hpp"

int main() {
    std::cout << "Testing CAD/FreeCAD Interface" << std::endl;

    NyxCAD::FreeCADInterface cad;

    // Test primitive creation
    std::cout << "Creating cube primitive..." << std::endl;
    auto cube_result = cad.create_primitive("cube", {10.0, 10.0, 10.0});  // 10x10x10 cube

    std::cout << "Cube creation result:" << std::endl;
    std::cout << "Success: " << (cube_result.success ? "Yes" : "No") << std::endl;
    std::cout << "Scripts generated: " << cube_result.generated_scripts.size() << std::endl;

    if (!cube_result.generated_scripts.empty()) {
        std::cout << "Sample script:" << std::endl;
        std::cout << cube_result.generated_scripts[0].substr(0, 200) << "..." << std::endl;
    }

    // Test extrusion
    std::cout << "Testing extrusion..." << std::endl;
    double direction[3] = {0, 0, 5};
    auto extrude_result = cad.extrude_shape("cube", 5.0, direction);

    std::cout << "Extrusion result:" << std::endl;
    std::cout << "Success: " << (extrude_result.success ? "Yes" : "No") << std::endl;
    std::cout << "Scripts generated: " << extrude_result.generated_scripts.size() << std::endl;

    // Test export
    std::cout << "Testing STL export..." << std::endl;
    auto export_result = cad.export_model("stl", "test_cube.stl");

    std::cout << "Export result:" << std::endl;
    std::cout << "Success: " << (export_result.success ? "Yes" : "No") << std::endl;
    std::cout << "Output file: " << export_result.output_file << std::endl;

    std::cout << "CAD tests completed!" << std::endl;
    return 0;
}