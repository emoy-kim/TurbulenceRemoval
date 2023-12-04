cmake_minimum_required(VERSION 3.8)

project("TurbulenceRemoval")

include(cmake/check-compiler.cmake)

set(CMAKE_CXX_STANDARD 17)

set(
	SOURCE_FILES 
		main.cpp
		TurbulenceRemoval.cpp
)

configure_file(ProjectPath.h.in ${PROJECT_BINARY_DIR}/ProjectPath.h @ONLY)

if(MSVC)
   include(cmake/add-libraries-windows.cmake)
else()
   include(cmake/add-libraries-linux.cmake)
endif()

add_executable(TurbulenceRemoval ${SOURCE_FILES})

if(MSVC)
   include(cmake/target-link-libraries-windows.cmake)
else()
   include(cmake/target-link-libraries-linux.cmake)
endif()

target_include_directories(TurbulenceRemoval PUBLIC ${CMAKE_BINARY_DIR})