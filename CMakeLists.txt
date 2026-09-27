# C engine (lob_engine shared library + bench_engine).
# Usually driven by `python scripts/build_engine.py`, which builds into build/.
cmake_minimum_required(VERSION 3.20)
project(lob_engine VERSION 1.0.0 LANGUAGES C)

option(LOBIMB_OPENMP "Parallelise batch kernels with OpenMP" ON)
option(LOBIMB_NATIVE "Tune for the build machine (-march=native)" ON)
option(LOBIMB_BUILD_LEGACY "Also build the legacy (v1) engines for benchmarks" ON)

# Benchmarks on a Debug build are meaningless.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
# build/bin is where the Python binding looks for the library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Shared library for ctypes. PREFIX "" gives lob_engine.dll, not
# liblob_engine.dll; hidden visibility exports only the LOB_API functions.
add_library(lob_engine SHARED engine/src/lob_engine.c)
target_include_directories(lob_engine PUBLIC engine/include)
set_target_properties(lob_engine PROPERTIES PREFIX "" C_VISIBILITY_PRESET hidden)

# -march=native binaries may not run on other CPUs (build with --no-native).
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lob_engine PRIVATE -Wall -Wextra -O3)
  if(LOBIMB_NATIVE)
    target_compile_options(lob_engine PRIVATE -march=native)
  endif()
elseif(MSVC)
  target_compile_options(lob_engine PRIVATE /W4 /O2 /fp:precise)
endif()

if(MINGW)
  # Link libgcc/libgomp/winpthread statically: Python 3.8+ does not search
  # PATH for dependent DLLs. OpenMP::OpenMP_C would pin libgomp.dll.a, hence
  # the plain -fopenmp.
  if(LOBIMB_OPENMP)
    target_compile_options(lob_engine PRIVATE -fopenmp)
    target_link_options(lob_engine PRIVATE -fopenmp)
  endif()
  target_link_options(lob_engine PRIVATE -static)
elseif(LOBIMB_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C)
  target_link_libraries(lob_engine PRIVATE OpenMP::OpenMP_C)
endif()

if(NOT WIN32)
  target_link_libraries(lob_engine PRIVATE m)
endif()

# Legacy (v1) engines, built only for the before/after benchmarks.
# Flags mirror the original builds (OBI at -O0, WOBI at -O2).
if(LOBIMB_BUILD_LEGACY)
  add_library(lob_legacy SHARED legacy/src_c/OBI/lob_engine.c)
  add_library(wobi_legacy SHARED legacy/src_c/WOBI/wobi_engine.c)
  target_compile_options(lob_legacy PRIVATE -O0)
  target_compile_options(wobi_legacy PRIVATE -O2)
  foreach(t lob_legacy wobi_legacy)
    # The legacy sources have no dllexport annotations.
    set_target_properties(${t} PROPERTIES PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MINGW)
      target_link_options(${t} PRIVATE -static)
    endif()
    if(NOT WIN32)
      target_link_libraries(${t} PRIVATE m)
    endif()
  endforeach()
endif()

# Pure-C benchmark. Links the DLL rather than the source, so it measures the
# same cross-module call as a real consumer.
add_executable(bench_engine engine/bench/bench_engine.c)
target_link_libraries(bench_engine PRIVATE lob_engine)
if(MINGW)
  target_link_options(bench_engine PRIVATE -static)
endif()
