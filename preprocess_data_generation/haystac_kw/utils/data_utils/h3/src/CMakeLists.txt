cmake_minimum_required(VERSION 3.1)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

project(h3_convert)

find_package(h3 CONFIG REQUIRED)

set(EXECUTABLE_OUTPUT_PATH bin)

# Build each example
add_executable(h3_convert "h3_convert.c")
target_link_libraries(h3_convert PRIVATE h3::h3)
install(TARGETS h3_convert RUNTIME DESTINATION bin)
include(CPack)