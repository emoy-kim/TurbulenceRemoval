include_directories("${CMAKE_SOURCE_DIR}/3rd_party/opencv/include")

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    link_directories("${CMAKE_SOURCE_DIR}/3rd_party/opencv/lib/windows/debug")
else()
    link_directories("${CMAKE_SOURCE_DIR}/3rd_party/opencv/lib/windows/release")
endif()