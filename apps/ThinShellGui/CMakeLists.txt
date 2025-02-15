################################################################################
# General Information
################################################################################

cmake_minimum_required(VERSION 3.1)
project(ThinShellGUi)

# Add your project files
# Create an executable
file(GLOB SRCFILES 
*.cpp
../../*.cpp ../../*.h ../../*.hpp
../../SecondFundamentalForm/*.cpp 
../../SecondFundamentalForm/*.h 
../../ThinShells/*.h
../../ThinShells/*.cpp
../../MeshLib/*.h
../../MeshLib/*.cpp
../../Optimization/*.h
../../Optimization/*.cpp
../../ExternalEnergies/*.h
../../ExternalEnergies/*.cpp
../../Collision/*.h
../../Collision/*.cpp
../../Common/*.h
../../Common/*.cpp
)
    
add_executable(
        ${PROJECT_NAME}_bin
        ${SRCFILES}
        )

set_target_properties(${PROJECT_NAME}_bin PROPERTIES CXX_STANDARD 17)
set_target_properties(${PROJECT_NAME}_bin PROPERTIES CXX_STANDARD_REQUIRED ON)

# Link settings
target_link_libraries( ${PROJECT_NAME}_bin
        polyscope 
        TBB::tbb 
        tbb_static 
        tbbmalloc_static
        igl::core 
        igl_restricted::triangle 
        igl_copyleft::comiso 
        ${SUITESPARSE_LIBRARIES}
        CLI11::CLI11)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_bin OpenMP::OpenMP_CXX)
endif()