#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# C wrapper around the header-only C++ proxsuite (ProxQP) solver.
#
# NOTE: the proxsuite CMake project is deliberately NOT included via
# add_subdirectory: its jrl-cmakemodules hook the global install/uninstall
# targets (an install(CODE ...) triggers "uninstall" at install time, which
# collides with the acados uninstall target and removes installed files).
# Since proxsuite is header-only, only its include directory and the
# generated version header proxsuite/config.hpp are needed.

# CONFIG mode, since the FindEigen3.cmake module in cmake/ does not define the Eigen3::Eigen target
find_package(Eigen3 3.3 REQUIRED CONFIG)

# generate proxsuite/config.hpp with the version parsed from the submodule
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../proxsuite/package.xml _proxsuite_version_line REGEX "<version>")
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" PROXSUITE_VERSION "${_proxsuite_version_line}")
set(PROXSUITE_MAJOR_VERSION ${CMAKE_MATCH_1})
set(PROXSUITE_MINOR_VERSION ${CMAKE_MATCH_2})
set(PROXSUITE_PATCH_VERSION ${CMAKE_MATCH_3})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/proxsuite_config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/proxsuite/config.hpp @ONLY)

add_library(proxsuite_c SHARED src/proxsuite_c.cpp)

target_include_directories(proxsuite_c
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include/proxsuite_c/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../proxsuite/include
            ${CMAKE_CURRENT_BINARY_DIR}/include)

target_link_libraries(proxsuite_c PRIVATE Eigen3::Eigen)

# Optional SIMD kernels of ProxQP. They are written with x86 intrinsics and made portable
# through simde, which maps them to NEON on ARM. simde only emits real SIMD instructions if
# the compiler targets an ISA that has them, so the architecture flags below are part of the
# feature: with PROXSUITE_VECTORIZE but without them, simde emulates the wide operations and
# ProxQP ends up slower than its own scalar path.
if(ACADOS_PROXQP_VECTORIZE)
    find_path(SIMDE_INCLUDE_DIR simde/x86/avx2.h HINTS ${SIMDE_DIR} PATH_SUFFIXES include)
    if(NOT SIMDE_INCLUDE_DIR)
        message(FATAL_ERROR "ACADOS_PROXQP_VECTORIZE is ON but simde was not found. "
                            "Install it (e.g. libsimde-dev) or point SIMDE_DIR at a checkout.")
    endif()

    # architecture flags, overridable for targets that need something else
    if(NOT DEFINED ACADOS_PROXQP_ARCH_FLAGS)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            set(ACADOS_PROXQP_ARCH_FLAGS -mavx2 -mfma)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(ACADOS_PROXQP_ARCH_FLAGS "")  # NEON is part of the ARMv8-A baseline
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
            set(ACADOS_PROXQP_ARCH_FLAGS -mfpu=neon)
        else()
            message(FATAL_ERROR "ACADOS_PROXQP_VECTORIZE is ON but no architecture flags are "
                                "known for ${CMAKE_SYSTEM_PROCESSOR}. Set ACADOS_PROXQP_ARCH_FLAGS.")
        endif()
    endif()

    target_compile_definitions(proxsuite_c PRIVATE PROXSUITE_VECTORIZE)
    target_include_directories(proxsuite_c PRIVATE ${SIMDE_INCLUDE_DIR})
    target_compile_options(proxsuite_c PRIVATE ${ACADOS_PROXQP_ARCH_FLAGS})
    message(STATUS "ProxQP SIMD kernels enabled (${ACADOS_PROXQP_ARCH_FLAGS})")
endif()

# proxsuite requires C++17; enforced via the target property since the
# acados top-level CMakeLists sets -std=c++0x in the global CMAKE_CXX_FLAGS
set_target_properties(proxsuite_c PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

install(TARGETS proxsuite_c EXPORT proxsuite_cTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(EXPORT proxsuite_cTargets
    FILE proxsuite_cConfig.cmake
    DESTINATION lib/cmake/proxsuite_c)
install(FILES include/proxsuite_c.h DESTINATION include/proxsuite_c/include)
