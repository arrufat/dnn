cmake_minimum_required(VERSION 3.11)
project("Neural Networks with dlib" LANGUAGES CXX)

# Use C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Enable ccache if it exists
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

# OpenMP support
find_package(OpenMP)

# Enable all optimization flags supported by the processor
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." ON)
if(${FORCE_COLORED_OUTPUT})
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options (-fdiagnostics-color=always)
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		add_compile_options (-fcolor-diagnostics)
	endif()
endif()

include(FetchContent)
# Fetch Dlib
set(DLIB_TAG master)
FetchContent_Declare(
	dlib
	GIT_REPOSITORY https://github.com/davisking/dlib.git
	GIT_TAG        ${DLIB_TAG}
)
message("-- Fetching Dlib ${DLIB_TAG}")
FetchContent_MakeAvailable(dlib)

# mnist example
add_executable(mnist_ex examples/mnist_ex.cpp)
target_include_directories(mnist_ex PRIVATE ${CMAKE_SOURCE_DIR}/dnn)
target_link_libraries(mnist_ex dlib::dlib OpenMP::OpenMP_CXX)
target_compile_options(mnist_ex PRIVATE -Wall -Wextra -pedantic)
install(TARGETS mnist_ex DESTINATION bin)
