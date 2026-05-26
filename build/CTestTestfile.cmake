# CMake generated Testfile for 
# Source directory: /mnt/c/trpo12/CBLAS_test/test
# Build directory: /mnt/c/trpo12/CBLAS_test/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(InterfaceTests "/mnt/c/trpo12/CBLAS_test/build/run_tests")
set_tests_properties(InterfaceTests PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/trpo12/CBLAS_test/test/CMakeLists.txt;14;add_test;/mnt/c/trpo12/CBLAS_test/test/CMakeLists.txt;0;")
add_test(PerformanceTests "/mnt/c/trpo12/CBLAS_test/build/benchmark_symm")
set_tests_properties(PerformanceTests PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/trpo12/CBLAS_test/test/CMakeLists.txt;15;add_test;/mnt/c/trpo12/CBLAS_test/test/CMakeLists.txt;0;")
