#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace NyxCAD {

// CAD Operation Types
enum class CADOperation {
    CREATE_PRIMITIVE,
    EXTRUDE,
    REVOLVE,
    BOOLEAN_UNION,
    BOOLEAN_INTERSECTION,
    BOOLEAN_DIFFERENCE,
    EXPORT_STL,
    EXPORT_STEP,
    ANALYZE_VOLUME,
    ANALYZE_MASS
};

// CAD Result Structure
struct CADResult {
    bool success;
    std::string output_file;
    std::string diagnostics;
    double volume;  // cubic mm
    double mass;    // grams
    std::vector<std::string> generated_scripts;
};

// FreeCAD Interface Class
class FreeCADInterface {
public:
    FreeCADInterface();
    ~FreeCADInterface() = default;

    // Core CAD operations
    CADResult create_primitive(const std::string& shape_type, const std::vector<double>& dimensions);
    CADResult extrude_shape(const std::string& base_shape, double height, double direction[3]);
    CADResult boolean_operation(CADOperation op, const std::string& shape1, const std::string& shape2);
    CADResult export_model(const std::string& format, const std::string& filename);

    // Analysis operations
    CADResult analyze_geometry(const std::string& shape_name);

    // Script generation
    std::string generate_freecad_script(const std::string& description);
    bool execute_script(const std::string& script_path);

    // Utility functions
    bool is_freecad_available() const;
    std::string get_freecad_version() const;
    std::vector<std::string> get_available_operations() const;

private:
    bool freecad_available_;
    std::string freecad_path_;
    std::string script_directory_;

    // Helper functions
    std::string create_temp_script(const std::string& content);
    std::string execute_freecad_command(const std::string& command);
    std::string parse_freecad_output(const std::string& output);
    bool validate_dimensions(const std::vector<double>& dims, const std::string& shape_type);
};

} // namespace NyxCAD