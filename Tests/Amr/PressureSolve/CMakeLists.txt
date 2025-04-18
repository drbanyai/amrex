add_executable(PressureSolve
    main.cpp
    helpers.cpp
)

target_link_libraries(PressureSolve
    PRIVATE
    AMReX::amrex
)

target_include_directories(PressureSolve
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Only support 3D for now
target_compile_definitions(PressureSolve
    PRIVATE
    AMREX_SPACEDIM=3
) 