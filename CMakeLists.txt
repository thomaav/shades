#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)
project (shades)

#
# CMake setup
#
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_VERBOSE_MAKEFILE 0) # 1 should be used for debugging
set (CMAKE_SUPPRESS_REGENERATION TRUE) # Suppresses ZERO_CHECK
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++11")
  if(NOT WIN32)
    set(GLAD_LIBRARIES dl)
  endif()
endif()

#
# GLFW options
#
option (GLFW_INSTALL OFF)
option (GLFW_BUILD_DOCS OFF)
option (GLFW_BUILD_EXAMPLES OFF)
option (GLFW_BUILD_TESTS OFF)
add_subdirectory (gloom/vendor/glfw)

#
# OpenAL options
#
add_subdirectory (gloom/vendor/openal-soft)

#
# freealut options
#
add_subdirectory (gloom/vendor/freealut)

#
# fftw3
#
include(ExternalProject)
ExternalProject_Add(
	project_fftw
	# GIT_REPOSITORY https://github.com/FFTW/fftw3
	URL http://www.fftw.org/fftw-3.3.8.tar.gz
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gloom/vendor/fftw
	CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/gloom/vendor/fftw/src/project_fftw/configure
      --prefix=${CMAKE_CURRENT_BINARY_DIR}/gloom/vendor/fftw/install
	INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/gloom/vendor/fftw/install
)

add_library(fftw STATIC IMPORTED)
set(lib_fftw_name ${CMAKE_STATIC_LIBRARY_PREFIX}fftw3${CMAKE_STATIC_LIBRARY_SUFFIX})
set_target_properties(fftw PROPERTIES IMPORTED_LOCATION
  ${CMAKE_CURRENT_BINARY_DIR}/gloom/vendor/fftw/install/lib/${lib_fftw_name})

#
# Set include paths
#
include_directories (gloom/src/
                     gloom/vendor/glad/include/
                     gloom/vendor/glfw/include/
                     gloom/vendor/glm/
                     gloom/vendor/stb/
                     gloom/vendor/opensoft-al/include/
                     gloom/vendor/freealut/include/
		     ${CMAKE_CURRENT_BINARY_DIR}/gloom/vendor/fftw/install/include/)

#
# Add files
#
file (GLOB         VENDORS_SOURCES gloom/vendor/glad/src/glad.c)
file (GLOB_RECURSE PROJECT_HEADERS gloom/src/*.hpp
                                   gloom/src/*.h)
file (GLOB_RECURSE PROJECT_SOURCES gloom/src/*.cpp
                                   gloom/src/*.cxx
                                   gloom/src/*.cc
                                   gloom/src/*.c)
file (GLOB_RECURSE PROJECT_SHADERS gloom/shaders/*.comp
                                   gloom/shaders/*.frag
                                   gloom/shaders/*.geom
                                   gloom/shaders/*.vert)
file (GLOB         PROJECT_CONFIGS CMakeLists.txt
                                   README.rst
                                  .gitignore
                                  .gitmodules)

#
# Organizing files
#
source_group ("headers" FILES ${PROJECT_HEADERS})
source_group ("shaders" FILES ${PROJECT_SHADERS})
source_group ("sources" FILES ${PROJECT_SOURCES})
source_group ("vendors" FILES ${VENDORS_SOURCES})

#
# Set executable and target link libraries
#
add_definitions (-DGLFW_INCLUDE_NONE
                 -DPROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")
add_executable (${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS}
                                ${PROJECT_SHADERS} ${PROJECT_CONFIGS}
                                ${VENDORS_SOURCES})
add_dependencies(shades project_fftw)
target_link_libraries (${PROJECT_NAME}
                       glfw
                       ${GLFW_LIBRARIES}
                       ${GLAD_LIBRARIES}
                       openal
                       alut
                       fftw)
set_target_properties (${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
