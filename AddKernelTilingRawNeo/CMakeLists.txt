cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# ${KERNEL_FILES} are used to compile library, push files written by ascendc in ${KERNEL_FILES}.
# ref to cmake/npu.cmake ascendc_library, cmake/cpu.cmake add_library
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/multiply_add_custom.cpp)

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
#if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")      
#    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
#else()
#    set(ASCEND_PATH "/data/huaqin/Ascend/ascend-toolkit/latest")
#endif()
#set(INCLUDE_BASE_DIR_ACLNN "${ASCEND_PATH}/include")
#include_directories(
#    ${INCLUDE_BASE_DIR_ACLNN}
#    ${INCLUDE_BASE_DIR_ACLNN}/aclnn
#)

# 设置链接的库文件路径
#target_link_libraries(opapi_test PRIVATE
#                      ${ASCEND_PATH}/lib64/libascendcl.so
#                      ${ASCEND_PATH}/lib64/libnnopbase.so
#                      ${ASCEND_PATH}/lib64/libopapi.so)


add_executable(ascendc_kernels_bbit ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE}
)

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
