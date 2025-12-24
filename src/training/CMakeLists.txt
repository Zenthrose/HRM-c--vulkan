add_library(training STATIC
    advanced_training_optimizations.cpp
    advanced_training_optimizations.hpp
    character_language_config.hpp
    character_language_evaluator.cpp
    character_language_evaluator.hpp
    character_language_loss.cpp
    character_language_loss.hpp
    character_language_trainer.cpp
    character_language_trainer.hpp
    character_text_dataset.cpp
    character_text_dataset.hpp
    int4_training_engine.cpp
    int4_training_engine.hpp
)
target_include_directories(training PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../hrm ../system ../utils ../vulkan ../core)
target_link_libraries(training PRIVATE hrm utils)