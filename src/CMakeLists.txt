add_executable(auto_stitcher alignment.cpp alignment.h featureMatch.cpp featureMatch.h graphPro.cpp graphPro.h topology.cpp topology.h util.cpp util.h main.cpp)
target_link_libraries(auto_stitcher tbb ${CMAKE_SOURCE_DIR}/../levmar-2.6/build/liblevmar.a ${LAPACK_LIBRARIES} ${LAPACK_LINKER_FLAGS} ${BLAS_LIBRARIES} ${OpenCV_LIBS})
