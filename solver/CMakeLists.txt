SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# test
include_directories(${PROJECT_SOURCE_DIR}/gcge/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)

add_executable(test     ${SRCS})

target_link_libraries(test
    PRIVATE 
        GCGE
        ${MPI_C_LIBRARIES}  # Link MPI libraries for the test executable
)

# 添加自定义命令来拷贝 GCGE 库到 test 可执行文件所在的目录
add_custom_command(TARGET test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/shared/lib/ ${CMAKE_BINARY_DIR}
    COMMENT "Copying GCGE library to test output directory"
)
