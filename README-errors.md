my version of the script I reached the following... before i went for a cup of coffee...

```
--   NVIDIA CUDA:                   YES (ver 12.6, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             87
--     NVIDIA PTX archs:            87
-- 
--   cuDNN:                         YES (ver 9.4.0)
-- 
--   Python 3:
--     Interpreter:                 /usr/local/bin/python3 (ver 3.10.12)
--     Libraries:                   /usr/lib/aarch64-linux-gnu/libpython3.10.so (ver 3.10.12)
--     Limited API:                 NO
--     numpy:                       /usr/local/lib/python3.10/dist-packages/numpy/core/include (ver 1.26.4)
--     install path:                /usr/lib/python3/dist-packages/cv2/python-3.10
-- 
--   Python (for build):            /usr/local/bin/python3
-- 
--   Java:                          
--     ant:                         NO
--     Java:                        NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
-- 
--   Install to:                    /usr
-- -----------------------------------------------------------------
-- 
-- Configuring done (58.8s)
-- Generating done (1.9s)
-- Build files have been written to: /root/opencv/build
[  0%] Built target opencv_highgui_plugins
[  0%] Generate opencv4.pc
[  0%] Built target opencv_dnn_plugins
[  0%] Building C object 3rdparty/openjpeg/openjp2/CMakeFiles/libopenjp2.dir/thread.c.o
CMake Deprecation Warning at /root/opencv/cmake/OpenCVGenPkgconfig.cmake:113 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.

```
encounter the following error:

```
[ 37%] Building NVCC (Device) object modules/dnn/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_grid_nms.cu.o
[ 37%] Building CXX object modules/features2d/CMakeFiles/opencv_features2d.dir/src/matchers.cpp.o
/root/opencv/modules/dnn/src/cuda/grid_nms.cu(98): warning #20054-D: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
          __attribute__((shared)) vector_type group_i_boxes[BLOCK_SIZE];
                                              ^
          detected during:
            instantiation of "void cv::dnn::cuda4dnn::kernels::raw::grid_nms<T,NORMALIZED_BBOX,BLOCK_SIZE>(cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::Span<int>, cv::dnn::cuda4dnn::csl::View<T>, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::index_type, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::size_type, float) [with T=__half, NORMALIZED_BBOX=true, BLOCK_SIZE=128]" at line 434
            instantiation of "void cv::dnn::cuda4dnn::kernels::grid_nms(const cv::dnn::cuda4dnn::csl::Stream &, cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorView<T>, int, __nv_bool, float) [with T=__half]" at line 464

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/root/opencv/modules/dnn/src/cuda/grid_nms.cu(98): warning #20054-D: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
          __attribute__((shared)) vector_type group_i_boxes[BLOCK_SIZE];
                                              ^
          detected during:
            instantiation of "void cv::dnn::cuda4dnn::kernels::raw::grid_nms<T,NORMALIZED_BBOX,BLOCK_SIZE>(cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::Span<int>, cv::dnn::cuda4dnn::csl::View<T>, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::index_type, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::size_type, float) [with T=__half, NORMALIZED_BBOX=false, BLOCK_SIZE=128]" at line 439
            instantiation of "void cv::dnn::cuda4dnn::kernels::grid_nms(const cv::dnn::cuda4dnn::csl::Stream &, cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorView<T>, int, __nv_bool, float) [with T=__half]" at line 464

/root/opencv/modules/dnn/src/cuda/grid_nms.cu(98): warning #20054-D: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
          __attribute__((shared)) vector_type group_i_boxes[BLOCK_SIZE];
                                              ^
          detected during:
            instantiation of "void cv::dnn::cuda4dnn::kernels::raw::grid_nms<T,NORMALIZED_BBOX,BLOCK_SIZE>(cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::Span<int>, cv::dnn::cuda4dnn::csl::View<T>, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::index_type, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::size_type, float) [with T=float, NORMALIZED_BBOX=true, BLOCK_SIZE=128]" at line 434
            instantiation of "void cv::dnn::cuda4dnn::kernels::grid_nms(const cv::dnn::cuda4dnn::csl::Stream &, cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorView<T>, int, __nv_bool, float) [with T=float]" at line 465

/root/opencv/modules/dnn/src/cuda/grid_nms.cu(98): warning #20054-D: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
          __attribute__((shared)) vector_type group_i_boxes[BLOCK_SIZE];
                                              ^
          detected during:
            instantiation of "void cv::dnn::cuda4dnn::kernels::raw::grid_nms<T,NORMALIZED_BBOX,BLOCK_SIZE>(cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::Span<int>, cv::dnn::cuda4dnn::csl::View<T>, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::index_type, cv::dnn::cuda4dnn::csl::device::size_type, cv::dnn::cuda4dnn::csl::device::size_type, float) [with T=float, NORMALIZED_BBOX=false, BLOCK_SIZE=128]" at line 439
            instantiation of "void cv::dnn::cuda4dnn::kernels::grid_nms(const cv::dnn::cuda4dnn::csl::Stream &, cv::dnn::cuda4dnn::csl::Span<unsigned int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorSpan<int>, cv::dnn::cuda4dnn::csl::TensorView<T>, int, __nv_bool, float) [with T=float]" at line 465

[ 37%] Building CXX object modules/features2d/CMakeFiles/opencv_features2d.dir/src/mser.cpp.o
[ 37%] Building NVCC (Device) object modules/dnn/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_max_unpooling.cu.o
[ 37%] Building NVCC (Device) object modules/cudafilters/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_column_filter.32fc1.cu.o
[ 37%] Building CXX object modules/features2d/CMakeFiles/opencv_features2d.dir/src/orb.cpp.o
[ 37%] Building CXX object modules/features2d/CMakeFiles/opencv_features2d.dir/src/sift.dispatch.cpp.o
```

encounter the following error:

```
[ 38%] Built target opencv_fuzzy
[ 38%] Building NVCC (Device) object modules/hfs/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_magnitude.cu.o
[ 38%] Building NVCC (Device) object modules/cudafilters/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_column_filter.32fc4.cu.o
[ 38%] Building NVCC (Device) object modules/hfs/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_gslic_seg_engine_gpu.cu.o
[ 38%] Building NVCC (Device) object modules/dnn/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_permute.cu.o
/root/opencv_contrib/modules/hfs/src/cuda/gslic_seg_engine_gpu.cu(190): warning #20054-D: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
      __attribute__((shared)) Float4_ color_shared[16*16];
                                      ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/root/opencv_contrib/modules/hfs/src/cuda/gslic_seg_engine_gpu.cu(191): warning #20054-D: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
      __attribute__((shared)) Float2_ xy_shared[16*16];
                                      ^

/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp: In instantiation of ‘void cv::hfs::orutils::MemoryBlock<T>::clear(unsigned char) [with T = cv::hfs::orutils::Vector4<unsigned char>]’:
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp:42:1:   required from ‘cv::hfs::orutils::MemoryBlock<T>::MemoryBlock(size_t) [with T = cv::hfs::orutils::Vector4<unsigned char>; size_t = long unsigned int]’
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_image.hpp:18:84:   required from ‘cv::hfs::orutils::Image<T>::Image(cv::hfs::orutils::Vector2<int>) [with T = cv::hfs::orutils::Vector4<unsigned char>]’
/root/opencv_contrib/modules/hfs/src/cuda/gslic_seg_engine_gpu.cu:43:115:   required from here
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp:47:7: warning: ‘void* memset(void*, int, size_t)’ writing to an object of non-trivial type ‘class cv::hfs::orutils::Vector4<unsigned char>’; use assignment instead [-Wclass-memaccess]
   47 |         memset(data_cpu, defaultValue, dataSize * sizeof(T));
      |       ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_vector.hpp:63:26: note: ‘class cv::hfs::orutils::Vector4<unsigned char>’ declared here
   63 | template <class T> class Vector4 : public Vector4_ < T >
      |                          ^~~~~~~
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp: In instantiation of ‘void cv::hfs::orutils::MemoryBlock<T>::clear(unsigned char) [with T = cv::hfs::orutils::Vector4<float>]’:
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp:42:1:   required from ‘cv::hfs::orutils::MemoryBlock<T>::MemoryBlock(size_t) [with T = cv::hfs::orutils::Vector4<float>; size_t = long unsigned int]’
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_image.hpp:18:84:   required from ‘cv::hfs::orutils::Image<T>::Image(cv::hfs::orutils::Vector2<int>) [with T = cv::hfs::orutils::Vector4<float>]’
/root/opencv_contrib/modules/hfs/src/cuda/gslic_seg_engine_gpu.cu:44:104:   required from here
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp:47:7: warning: ‘void* memset(void*, int, size_t)’ writing to an object of non-trivial type ‘class cv::hfs::orutils::Vector4<float>’; use assignment instead [-Wclass-memaccess]
   47 |         memset(data_cpu, defaultValue, dataSize * sizeof(T));
      |       ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_vector.hpp:63:26: note: ‘class cv::hfs::orutils::Vector4<float>’ declared here
   63 | template <class T> class Vector4 : public Vector4_ < T >
      |                          ^~~~~~~
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp: In instantiation of ‘void cv::hfs::orutils::MemoryBlock<T>::clear(unsigned char) [with T = cv::hfs::slic::gSpixelInfo]’:
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp:42:1:   required from ‘cv::hfs::orutils::MemoryBlock<T>::MemoryBlock(size_t) [with T = cv::hfs::slic::gSpixelInfo; size_t = long unsigned int]’
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_image.hpp:18:84:   required from ‘cv::hfs::orutils::Image<T>::Image(cv::hfs::orutils::Vector2<int>) [with T = cv::hfs::slic::gSpixelInfo]’
/root/opencv_contrib/modules/hfs/src/cuda/gslic_seg_engine_gpu.cu:54:80:   required from here
/root/opencv_contrib/modules/hfs/src/cuda/../or_utils/or_memory_block.hpp:47:7: warning: ‘void* memset(void*, int, size_t)’ writing to an object of non-trivial type ‘struct cv::hfs::slic::gSpixelInfo’; use assignment instead [-Wclass-memaccess]
   47 |         memset(data_cpu, defaultValue, dataSize * sizeof(T));
      |       ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/opencv_contrib/modules/hfs/src/cuda/../slic/slic.hpp:67:8: note: ‘struct cv::hfs::slic::gSpixelInfo’ declared here
   67 | struct gSpixelInfo
      |        ^~~~~~~~~~~
[ 38%] Building CXX object modules/hfs/CMakeFiles/opencv_hfs.dir/src/hfs.cpp.o
[ 38%] Building CXX object modules/hfs/CMakeFiles/opencv_hfs.dir/src/hfs_core.cpp.o
In file included from /root/opencv_contrib/modules/hfs/src/or_utils/or_image.hpp:9,
                 from /root/opencv_contrib/modules/hfs/src/or_utils/or_types.hpp:9,
                 from /root/opencv_contrib/modules/hfs/src/precomp.hpp:12,
                 from /root/opencv_contrib/modules/hfs/src/hfs_core.cpp:5:
/root/opencv_contrib/modules/hfs/src/or_utils/or_memory_block.hpp: In instantiation of ‘void cv::hfs::orutils::MemoryBlock<T>::clear(unsigned char) [with T = cv::hfs::orutils::Vector4<unsigned char>]’:
/root/opencv_contrib/modules/hfs/src/or_utils/or_memory_block.hpp:42:9:   required from ‘cv::hfs::orutils::MemoryBlock<T>::MemoryBlock(size_t) [with T = cv::hfs::orutils::Vector4<unsigned char>; size_t = long unsigned int]’
/root/opencv_contrib/modules/hfs/src/or_utils/or_image.hpp:19:49:   required from ‘cv::hfs::orutils::Image<T>::Image(cv::hfs::orutils::Vector2<int>) [with T = cv::hfs::orutils::Vector4<unsigned char>]’
/root/opencv_contrib/modules/hfs/src/hfs_core.cpp:66:58:   required from here
/root/opencv_contrib/modules/hfs/src/or_utils/or_memory_block.hpp:47:15: warning: ‘void* memset(void*, int, size_t)’ writing to an object of non-trivial type ‘class cv::hfs::orutils::Vector4<unsigned char>’; use assignment instead [-Wclass-memaccess]
   47 |         memset(data_cpu, defaultValue, dataSize * sizeof(T));
      |         ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /root/opencv_contrib/modules/hfs/src/or_utils/or_types.hpp:8,
                 from /root/opencv_contrib/modules/hfs/src/precomp.hpp:12,
                 from /root/opencv_contrib/modules/hfs/src/hfs_core.cpp:5:
/root/opencv_contrib/modules/hfs/src/or_utils/or_vector.hpp:63:26: note: ‘class cv::hfs::orutils::Vector4<unsigned char>’ declared here
   63 | template <class T> class Vector4 : public Vector4_ < T >
      |                          ^~~~~~~
[ 38%] Building NVCC (Device) object modules/dnn/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_prior_box.cu.o
[ 38%] Building CXX object modules/hfs/CMakeFiles/opencv_hfs.dir/src/magnitude/magnitude.cpp.o
[ 38%] Building CXX object modules/hfs/CMakeFiles/opencv_hfs.dir/src/merge/merge.cpp.o
[ 38%] Building CXX object modules/hfs/CMakeFiles/opencv_hfs.dir/src/slic/gslic_engine.cpp.o
[ 39%] Building CXX object modules/hfs/CMakeFiles/opencv_hfs.dir/src/slic/slic.cpp.o
[ 39%] Building CXX object modules/img_hash/CMakeFiles/opencv_img_hash.dir/src/average_hash.cpp.o
[ 40%] Building NVCC (Device) object modules/dnn/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_region.cu.o
[ 40%] Linking CXX shared library ../../lib/libopencv_hfs.so
[ 40%] Built target opencv_hfs
```
encountered the following error:

```
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/concat_layer.cpp.o
[ 46%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/upnp.cpp.o
[ 46%] Linking CXX shared library ../../lib/libopencv_videoio.so
/root/opencv_contrib/modules/cudafilters/src/cuda/wavelet_matrix_2d.cuh(451): warning #2912-D: constexpr if statements are a C++17 feature
                  if constexpr(CH_NUM > 1) {
                     ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

[ 46%] Built target opencv_videoio
[ 46%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/bundle.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/const_layer.cpp.o
[ 46%] Building CXX object modules/highgui/CMakeFiles/opencv_highgui.dir/src/backend.cpp.o
[ 46%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/degeneracy.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/convolution_layer.cpp.o
/root/opencv_contrib/modules/cudafilters/src/cuda/wavelet_matrix_float_supporter.cuh(81): warning #2912-D: constexpr if statements are a C++17 feature
      if constexpr (CH_NUM >= 2) {
         ^
          detected during:
            instantiation of "void cv::cuda::device::wavelet_matrix_median::WMMedianFloatSupporter::WMMedianFloatSupporter<ValT, CH_NUM, IdxT>::sort_and_set(IdxT *, IdxT) [with ValT=float, CH_NUM=1, IdxT=uint32_t]" at line 381 of /root/opencv_contrib/modules/cudafilters/src/cuda/median_filter.cu
            instantiation of "void cv::cuda::device::medianFiltering_wavelet_matrix_gpu<CH_NUM,T>(cv::cuda::PtrStepSz<T>, cv::cuda::PtrStepSz<T>, int, cudaStream_t) [with CH_NUM=1, T=unsigned char]" at line 400 of /root/opencv_contrib/modules/cudafilters/src/cuda/median_filter.cu
            instantiation of "void cv::cuda::device::medianFiltering_wavelet_matrix_gpu(cv::cuda::PtrStepSz<T>, cv::cuda::PtrStepSz<T>, int, int, cudaStream_t) [with T=unsigned char]" at line 410 of /root/opencv_contrib/modules/cudafilters/src/cuda/median_filter.cu

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

[ 46%] Building CXX object modules/highgui/CMakeFiles/opencv_highgui.dir/src/window.cpp.o
[ 46%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/dls_solver.cpp.o
[ 46%] Building CXX object modules/highgui/CMakeFiles/opencv_highgui.dir/src/roiSelector.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/correlation_layer.cpp.o
[ 46%] Building CXX object modules/highgui/CMakeFiles/opencv_highgui.dir/src/window_gtk.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cpu_kernels/conv_depthwise.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cpu_kernels/conv_winograd_f63.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cpu_kernels/conv_winograd_f63.dispatch.cpp.o
[ 46%] Linking CXX shared library ../../lib/libopencv_highgui.so
[ 46%] Built target opencv_highgui
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cpu_kernels/convolution.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cpu_kernels/fast_gemm.cpp.o
[ 46%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/essential_solver.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cpu_kernels/fast_norm.cpp.o
/root/opencv_contrib/modules/cudafilters/src/cuda/wavelet_matrix_2d.cuh: In member function ‘void cv::cuda::device::wavelet_matrix_median::WaveletMatrix2dCu5C<ValT, CH_NUM, MultiWaveletMatrixImpl, TH_NUM, WORD_SIZE>::construct(const ValT*, cudaStream_t, bool)’:
/root/opencv_contrib/modules/cudafilters/src/cuda/wavelet_matrix_2d.cuh:885:4: warning: ‘if constexpr’ only available with ‘-std=c++17’ or ‘-std=gnu++17’
  885 |         if constexpr (sizeof(ValT) >= 4) for (; h > 16; --h) {
      |    ^    ~~~~
/root/opencv_contrib/modules/cudafilters/src/cuda/wavelet_matrix_2d.cuh:903:4: warning: ‘if constexpr’ only available with ‘-std=c++17’ or ‘-std=gnu++17’
  903 |         if constexpr (sizeof(ValT) >= 4) if (h == 16 || (is_same<ValT, uint32_t>::value && h >= 0)) do {
      |    ^    ~~~~
/root/opencv_contrib/modules/cudafilters/src/cuda/wavelet_matrix_2d.cuh:922:4: warning: ‘if constexpr’ only available with ‘-std=c++17’ or ‘-std=gnu++17’
  922 |         if constexpr (sizeof(ValT) >= 2) for (; h > 8; --h) {
      |    ^    ~~~~
/root/opencv_contrib/modules/cudafilters/src/cuda/wavelet_matrix_2d.cuh:940:4: warning: ‘if constexpr’ only available with ‘-std=c++17’ or ‘-std=gnu++17’
  940 |         if constexpr (sizeof(ValT) >= 2) if (h == 8 || (is_same<ValT, uint32_t>::value && h >= 0)) do {
      |    ^    ~~~~
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cpu_kernels/softmax.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/crop_and_resize_layer.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/cumsum_layer.cpp.o
[ 46%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/estimator.cpp.o
[ 46%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/depth_space_ops_layer.cpp.o
[ 46%] Building NVCC (Device) object modules/cudafilters/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_row_filter.16sc1.cu.o
```
another error:

```
[ 47%] Building NVCC (Device) object modules/cudacodec/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_ColorSpace.cu.o
[ 47%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/ransac_solvers.cpp.o
[ 47%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/fully_connected_layer.cpp.o
/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu(163): warning #497-D: declaration of "Color" hides template parameter
          Color Color[2];
                ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu(217): warning #497-D: declaration of "Color" hides template parameter
          Color Color[2];
                ^

/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu(163): warning #497-D: declaration of "Color" hides template parameter
          Color Color[2];
                ^
          detected during instantiation of "void cv::cuda::device::Nv12ToColor24<COLOR24>(uint8_t *, int, uint8_t *, int, int, int, __nv_bool, cudaStream_t) [with COLOR24=cv::cuda::device::BGR24]" at line 691

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu(217): warning #497-D: declaration of "Color" hides template parameter
          Color Color[2];
                ^
          detected during instantiation of "void cv::cuda::device::YUV444ToColor24<COLOR24>(uint8_t *, int, uint8_t *, int, int, int, __nv_bool, cudaStream_t) [with COLOR24=cv::cuda::device::BGR24]" at line 727

/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:44: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                                            ^~~                      
/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu:52:6: warning: no previous declaration for ‘void cv::cuda::device::SetMatYuv2Rgb(int, bool)’ [-Wmissing-declarations]
   52 | void SetMatYuv2Rgb(int iMatrix, bool fullRange = false) {
      |      ^~~~~~~~~~~~~
/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu:371:6: warning: no previous declaration for ‘void cv::cuda::device::Y8ToGray8(uint8_t*, int, uint8_t*, int, int, int, bool, cudaStream_t)’ [-Wmissing-declarations]
  371 | void Y8ToGray8(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
      |      ^~~~~~~~~
/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu:379:6: warning: no previous declaration for ‘void cv::cuda::device::Y8ToGray16(uint8_t*, int, uint8_t*, int, int, int, bool, cudaStream_t)’ [-Wmissing-declarations]
  379 | void Y8ToGray16(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
      |      ^~~~~~~~~~
/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu:387:6: warning: no previous declaration for ‘void cv::cuda::device::Y16ToGray8(uint8_t*, int, uint8_t*, int, int, int, bool, cudaStream_t)’ [-Wmissing-declarations]
  387 | void Y16ToGray8(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
      |      ^~~~~~~~~~
/root/opencv_contrib/modules/cudacodec/src/cuda/ColorSpace.cu:395:6: warning: no previous declaration for ‘void cv::cuda::device::Y16ToGray16(uint8_t*, int, uint8_t*, int, int, int, bool, cudaStream_t)’ [-Wmissing-declarations]
  395 | void Y16ToGray16(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
      |      ^~~~~~~~~~~
[ 47%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/gather_elements_layer.cpp.o
[ 47%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/sampler.cpp.o
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/NvEncoder.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/NvEncoder.cpp:4:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/gather_layer.cpp.o
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/NvEncoderCuda.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/NvEncoderCuda.cpp:4:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/termination.cpp.o
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/cuvid_video_source.cpp.o
[ 47%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/gemm_layer.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/cuvid_video_source.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/ffmpeg_video_source.cpp.o
[ 47%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/src/usac/utils.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/ffmpeg_video_source.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/frame_queue.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/frame_queue.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/nvidia_surface_format_to_color_converter.cpp.o
[ 47%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/group_norm_layer.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/nvidia_surface_format_to_color_converter.cpp:5:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
/root/opencv_contrib/modules/cudacodec/src/nvidia_surface_format_to_color_converter.cpp:97:7: warning: base class ‘class cv::cudacodec::NVSurfaceToColorConverter’ has accessible non-virtual destructor [-Wnon-virtual-dtor]
   97 | class NVSurfaceToColorConverterImpl : public NVSurfaceToColorConverter {
      |       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/opencv_contrib/modules/cudacodec/src/nvidia_surface_format_to_color_converter.cpp:97:7: warning: ‘class NVSurfaceToColorConverterImpl’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/thread.cpp.o
[ 47%] Building CXX object modules/calib3d/CMakeFiles/opencv_calib3d.dir/opencl_kernels_calib3d.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/thread.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/instance_norm_layer.cpp.o
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/video_decoder.cpp.o
[ 47%] Linking CXX shared library ../../lib/libopencv_calib3d.so
[ 47%] Built target opencv_calib3d
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/video_parser.cpp.o
[ 47%] Building NVCC (Device) object modules/cudafilters/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_row_filter.16sc4.cu.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/video_decoder.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/video_reader.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/video_parser.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/video_source.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/video_reader.cpp:43:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 47%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/layer_norm.cpp.o
[ 48%] Building CXX object modules/cudacodec/CMakeFiles/opencv_cudacodec.dir/src/video_writer.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/video_source.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 48%] Processing OpenCL kernels (bioinspired)
[ 48%] Building CXX object modules/bioinspired/CMakeFiles/opencv_bioinspired.dir/src/basicretinafilter.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/src/precomp.hpp:54,
                 from /root/opencv_contrib/modules/cudacodec/src/video_writer.cpp:44:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 48%] Linking CXX shared library ../../lib/libopencv_cudacodec.so
[ 48%] Built target opencv_cudacodec
```
another error:

```
[ 56%] Processing OpenCL kernels (xfeatures2d)
[ 56%] Building NVCC (Device) object modules/xfeatures2d/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_surf.cu.o
[ 56%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/vkcom/src/op_naryEltwise.cpp.o
[ 56%] Building CXX object modules/structured_light/CMakeFiles/opencv_structured_light.dir/src/sinusoidalpattern.cpp.o
[ 56%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/vkcom/src/pipeline.cpp.o
[ 56%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/vkcom/src/tensor.cpp.o
[ 56%] Building CXX object modules/rgbd/CMakeFiles/opencv_rgbd.dir/src/pose_graph.cpp.o
[ 56%] Linking CXX shared library ../../lib/libopencv_structured_light.so
[ 56%] Built target opencv_structured_light
[ 56%] Building CXX object modules/rgbd/CMakeFiles/opencv_rgbd.dir/src/tsdf.cpp.o
[ 56%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/vkcom/vulkan/vk_functions.cpp.o
[ 56%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/src/vkcom/vulkan/vk_loader.cpp.o
[ 56%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/layers/cpu_kernels/conv_block.neon.cpp.o
/root/opencv_contrib/modules/xfeatures2d/src/cuda/surf.cu:733:30: warning: no previous declaration for ‘void cv::cuda::device::surf::calc_dx_dy(cv::cudev::TexturePtr<unsigned char>, const float*, const float*, const float*, const float*, float&, float&)’ [-Wmissing-declarations]
  733 |         __device__ void calc_dx_dy(cudev::TexturePtr<uchar> tex, const float* featureX, const float* featureY, const float* featureSize, const float* featureDir,
      |                              ^~~~~~~~~~
/root/opencv_contrib/modules/xfeatures2d/src/cuda/surf.cu: In function ‘void cv::cuda::device::surf::icvFindMaximaInLayer_gpu(const cv::cuda::PtrStepSz<unsigned int>&, const PtrStepf&, const PtrStepf&, int4*, unsigned int*, int, int, int, bool, int)’:
/root/opencv_contrib/modules/xfeatures2d/src/cuda/surf.cu:379:15: warning: ‘mask.cv::cuda::device::surf::Mask<false>::tex.cv::cudev::TexturePtr<unsigned int>::tex’ may be used uninitialized [-Wmaybe-uninitialized]
  379 |                 Mask<false> mask;
      |               ^ ~~
[ 56%] Building CXX object modules/xfeatures2d/CMakeFiles/opencv_xfeatures2d.dir/src/affine_feature2d.cpp.o
[ 56%] Building CXX object modules/rgbd/CMakeFiles/opencv_rgbd.dir/src/tsdf_functions.cpp.o
[ 56%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/layers/cpu_kernels/conv_winograd_f63.neon.cpp.o
[ 57%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/layers/cpu_kernels/fast_gemm_kernels.neon.cpp.o
[ 57%] Building CXX object modules/bioinspired/CMakeFiles/opencv_perf_bioinspired.dir/perf/opencl/perf_retina.ocl.cpp.o
[ 57%] Building CXX object modules/rgbd/CMakeFiles/opencv_rgbd.dir/src/utils.cpp.o
[ 57%] Building CXX object modules/xfeatures2d/CMakeFiles/opencv_xfeatures2d.dir/src/beblid.cpp.o
[ 57%] Building CXX object modules/dnn/CMakeFiles/opencv_dnn.dir/layers/cpu_kernels/conv_block.neon_fp16.cpp.o
[ 57%] Building CXX object modules/rgbd/CMakeFiles/opencv_rgbd.dir/src/volume.cpp.o
[ 57%] Linking CXX shared library ../../lib/libopencv_dnn.so
[ 57%] Building CXX object modules/bioinspired/CMakeFiles/opencv_perf_bioinspired.dir/perf/perf_main.cpp.o
[ 57%] Building CXX object modules/xfeatures2d/CMakeFiles/opencv_xfeatures2d.dir/src/boostdesc.cpp.o
[ 57%] Built target opencv_dnn
```


```
[ 65%] Built target opencv_text
[ 65%] Building CXX object modules/cudacodec/CMakeFiles/opencv_perf_cudacodec.dir/perf/perf_main.cpp.o
[ 65%] Linking CXX executable ../../bin/opencv_perf_calib3d
In file included from /root/opencv_contrib/modules/cudacodec/perf/perf_precomp.hpp:48,
                 from /root/opencv_contrib/modules/cudacodec/perf/perf_main.cpp:43:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 65%] Built target opencv_perf_calib3d
[ 65%] Building CXX object modules/cudacodec/CMakeFiles/opencv_perf_cudacodec.dir/perf/perf_video.cpp.o
[ 65%] Building CXX object modules/cudaimgproc/CMakeFiles/opencv_cudaimgproc.dir/src/connectedcomponents.cpp.o
[ 65%] Building CXX object modules/cudaimgproc/CMakeFiles/opencv_cudaimgproc.dir/src/corners.cpp.o
In file included from /root/opencv_contrib/modules/cudacodec/perf/perf_precomp.hpp:48,
                 from /root/opencv_contrib/modules/cudacodec/perf/perf_video.cpp:43:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 65%] Linking CXX executable ../../bin/opencv_perf_cudacodec
[ 65%] Built target opencv_perf_cudacodec
[ 65%] Building NVCC (Device) object modules/cudafeatures2d/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_orb.cu.o
```

another error:

```
[ 82%] Built target opencv_cudabgsegm
[ 82%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_NCVPyramid.cu.o
[ 82%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_NPP_staging.cu.o
[ 82%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/radon_transform.cpp.o
[ 82%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/ridgedetectionfilter.cpp.o
[ 82%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_bm.cu.o
[ 82%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/rolling_guidance_filter.cpp.o
[ 82%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/run_length_morphology.cpp.o
[ 82%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_bm_fast.cu.o
[ 82%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/scansegment.cpp.o
[ 83%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_calib3d.cu.o
[ 83%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_ccomponetns.cu.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/seeds.cpp.o
[ 83%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_fgd.cu.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/selectivesearchsegmentation.cpp.o
[ 83%] Building NVCC (Device) object modules/cudalegacy/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_gmg.cu.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/slic.cpp.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/sparse_match_interpolators.cpp.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/structured_edge_detection.cpp.o
[ 83%] Building CXX object modules/face/CMakeFiles/opencv_face.dir/src/bif.cpp.o
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/NCV.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/NCV.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/face/CMakeFiles/opencv_face.dir/src/eigen_faces.cpp.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/thinning.cpp.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/weighted_median_filter.cpp.o
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/bm.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/bm.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/face/CMakeFiles/opencv_face.dir/src/face_alignment.cpp.o
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/bm_fast.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/bm_fast.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/grunarg.cpp.o
[ 83%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/opencl_kernels_ximgproc.cpp.o
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/calib3d.cpp.o
[ 83%] Linking CXX shared library ../../lib/libopencv_ximgproc.so
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/calib3d.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Built target opencv_ximgproc
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/fgd.cpp.o
[ 83%] Building CXX object modules/face/CMakeFiles/opencv_face.dir/src/face_basic.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/fgd.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/gorigin.cpp.o
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/gmg.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/gmg.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/graphcuts.cpp.o
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/gmat.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/graphcuts.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/face/CMakeFiles/opencv_face.dir/src/facemark.cpp.o
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/garray.cpp.o
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/image_pyramid.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/image_pyramid.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/gopaque.cpp.o
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/interpolate_frames.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/interpolate_frames.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/cudalegacy/CMakeFiles/opencv_cudalegacy.dir/src/needle_map.cpp.o
[ 83%] Building CXX object modules/face/CMakeFiles/opencv_face.dir/src/facemarkAAM.cpp.o
In file included from /root/opencv_contrib/modules/cudalegacy/src/precomp.hpp:84,
                 from /root/opencv_contrib/modules/cudalegacy/src/needle_map.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/gscalar.cpp.o
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/gframe.cpp.o
[ 83%] Linking CXX shared library ../../lib/libopencv_cudalegacy.so
[ 83%] Built target opencv_cudalegacy
[ 83%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/api/gkernel.cpp.o
[ 83%] Processing OpenCL kernels (tracking)
```
another error

```
[ 89%] Built target opencv_perf_cudalegacy
[ 89%] Building NVCC (Device) object modules/cudaobjdetect/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_lbp.cu.o
[ 90%] Building CXX object modules/ximgproc/CMakeFiles/opencv_perf_ximgproc.dir/perf/perf_radon_transform.cpp.o
[ 90%] Building CXX object modules/ximgproc/CMakeFiles/opencv_perf_ximgproc.dir/perf/perf_ridge_detection_filter.cpp.o
[ 90%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/executor/gtbbexecutor.cpp.o
[ 90%] Building NVCC (Device) object modules/cudaobjdetect/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_hog.cu.o
[ 90%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/executor/gthreadedexecutor.cpp.o
[ 90%] Processing OpenCL kernels (optflow)
[ 90%] Building CXX object modules/optflow/CMakeFiles/opencv_optflow.dir/src/deepflow.cpp.o
[ 90%] Building CXX object modules/ximgproc/CMakeFiles/opencv_perf_ximgproc.dir/perf/perf_rolling_guidance_filter.cpp.o
[ 90%] Building CXX object modules/optflow/CMakeFiles/opencv_optflow.dir/src/interfaces.cpp.o
[ 90%] Building CXX object modules/ximgproc/CMakeFiles/opencv_perf_ximgproc.dir/perf/perf_run_length_morphology.cpp.o
[ 90%] Building CXX object modules/optflow/CMakeFiles/opencv_optflow.dir/src/motempl.cpp.o
[ 90%] Building CXX object modules/gapi/CMakeFiles/opencv_gapi.dir/src/executor/gstreamingexecutor.cpp.o
[ 90%] Building CXX object modules/cudaobjdetect/CMakeFiles/opencv_cudaobjdetect.dir/src/cascadeclassifier.cpp.o
In file included from /root/opencv_contrib/modules/cudaobjdetect/src/precomp.hpp:59,
                 from /root/opencv_contrib/modules/cudaobjdetect/src/cascadeclassifier.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 90%] Building CXX object modules/optflow/CMakeFiles/opencv_optflow.dir/src/pcaflow.cpp.o
[ 90%] Building CXX object modules/ximgproc/CMakeFiles/opencv_perf_ximgproc.dir/perf/perf_thining.cpp.o
[ 90%] Building CXX object modules/cudaobjdetect/CMakeFiles/opencv_cudaobjdetect.dir/src/hog.cpp.o
[ 90%] Building CXX object modules/optflow/CMakeFiles/opencv_optflow.dir/src/rlof/geo_interpolation.cpp.o
In file included from /root/opencv_contrib/modules/cudaobjdetect/src/precomp.hpp:59,
                 from /root/opencv_contrib/modules/cudaobjdetect/src/hog.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 90%] Building CXX object modules/ximgproc/CMakeFiles/opencv_perf_ximgproc.dir/perf/perf_weighted_median_filter.cpp.o
[ 90%] Linking CXX shared library ../../lib/libopencv_cudaobjdetect.so
[ 90%] Built target opencv_cudaobjdetect
[ 90%] Processing OpenCL kernels (stitching)
```
one more

```
[ 99%] Linking CXX executable ../../bin/opencv_perf_gapi
[ 99%] Built target opencv_perf_gapi
[ 99%] Building CXX object modules/cudaoptflow/CMakeFiles/opencv_cudaoptflow.dir/src/brox.cpp.o
[ 99%] Building CXX object modules/cudaoptflow/CMakeFiles/opencv_cudaoptflow.dir/src/farneback.cpp.o
[ 99%] Building CXX object modules/cudaoptflow/CMakeFiles/opencv_cudaoptflow.dir/src/nvidiaOpticalFlow.cpp.o
[ 99%] Building CXX object modules/cudaoptflow/CMakeFiles/opencv_cudaoptflow.dir/src/pyrlk.cpp.o
In file included from /root/opencv_contrib/modules/cudaoptflow/src/precomp.hpp:61,
                 from /root/opencv_contrib/modules/cudaoptflow/src/brox.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
In file included from /root/opencv_contrib/modules/cudaoptflow/src/precomp.hpp:61,
                 from /root/opencv_contrib/modules/cudaoptflow/src/pyrlk.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
In file included from /root/opencv_contrib/modules/cudaoptflow/src/precomp.hpp:61,
                 from /root/opencv_contrib/modules/cudaoptflow/src/nvidiaOpticalFlow.cpp:7:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
In file included from /root/opencv_contrib/modules/cudaoptflow/src/precomp.hpp:61,
                 from /root/opencv_contrib/modules/cudaoptflow/src/farneback.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 99%] Building CXX object modules/cudaoptflow/CMakeFiles/opencv_cudaoptflow.dir/src/tvl1flow.cpp.o
In file included from /root/opencv_contrib/modules/cudaoptflow/src/precomp.hpp:61,
                 from /root/opencv_contrib/modules/cudaoptflow/src/tvl1flow.cpp:43:
/root/opencv_contrib/modules/cudalegacy/include/opencv2/cudalegacy/private.hpp:94:8: warning: extra tokens at end of #endif directive [-Wendif-labels]
   94 | #endif HAVE_CUDA
      |        ^~~~~~~~~
[ 99%] Linking CXX shared library ../../lib/libopencv_cudaoptflow.so
[ 99%] Built target opencv_cudaoptflow
[ 99%] Building NVCC (Device) object modules/superres/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_btv_l1_gpu.cu.o
```
one more
```
[ 99%] Building CXX object modules/cudaoptflow/CMakeFiles/opencv_perf_cudaoptflow.dir/perf/perf_main.cpp.o
[ 99%] Building CXX object modules/cudaoptflow/CMakeFiles/opencv_perf_cudaoptflow.dir/perf/perf_optflow.cpp.o
[ 99%] Building CXX object modules/superres/CMakeFiles/opencv_superres.dir/src/btv_l1.cpp.o
[ 99%] Building CXX object modules/superres/CMakeFiles/opencv_superres.dir/src/btv_l1_cuda.cpp.o
[ 99%] Linking CXX executable ../../bin/opencv_perf_cudaoptflow
In file included from /root/opencv_contrib/modules/superres/src/precomp.hpp:82,
                 from /root/opencv_contrib/modules/superres/src/btv_l1_cuda.cpp:46:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /root/opencv_contrib/modules/superres/src/precomp.hpp:82,
                 from /root/opencv_contrib/modules/superres/src/btv_l1.cpp:46:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 99%] Built target opencv_perf_cudaoptflow
[ 99%] Building CXX object modules/superres/CMakeFiles/opencv_superres.dir/src/frame_source.cpp.o
[ 99%] Building CXX object modules/superres/CMakeFiles/opencv_superres.dir/src/input_array_utility.cpp.o
[ 99%] Building CXX object modules/superres/CMakeFiles/opencv_superres.dir/src/optical_flow.cpp.o
In file included from /root/opencv_contrib/modules/superres/src/precomp.hpp:82,
                 from /root/opencv_contrib/modules/superres/src/frame_source.cpp:42:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[ 99%] Building CXX object modules/superres/CMakeFiles/opencv_superres.dir/src/super_resolution.cpp.o
In file included from /root/opencv_contrib/modules/superres/src/precomp.hpp:82,
                 from /root/opencv_contrib/modules/superres/src/input_array_utility.cpp:43:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/deblurring.cpp.o
In file included from /root/opencv_contrib/modules/superres/src/precomp.hpp:82,
                 from /root/opencv_contrib/modules/superres/src/optical_flow.cpp:43:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[100%] Building CXX object modules/superres/CMakeFiles/opencv_superres.dir/opencl_kernels_superres.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/fast_marching.cpp.o
In file included from /root/opencv_contrib/modules/superres/src/precomp.hpp:82,
                 from /root/opencv_contrib/modules/superres/src/super_resolution.cpp:43:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/frame_source.cpp.o
[100%] Linking CXX shared library ../../lib/libopencv_superres.so
[100%] Built target opencv_superres
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/global_motion.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/inpainting.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/log.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/motion_stabilizing.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/optical_flow.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/outlier_rejection.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/stabilizer.cpp.o
[100%] Building CXX object modules/videostab/CMakeFiles/opencv_videostab.dir/src/wobble_suppression.cpp.o
[100%] Building CXX object modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_main.cpp.o
[100%] Building CXX object modules/superres/CMakeFiles/opencv_perf_superres.dir/perf/perf_superres.cpp.o
[100%] Linking CXX shared library ../../lib/libopencv_videostab.so
[100%] Built target opencv_videostab
[100%] Building CXX object modules/python3/CMakeFiles/opencv_python3.dir/__/src2/cv2_util.cpp.o
[100%] Building CXX object modules/python3/CMakeFiles/opencv_python3.dir/__/src2/cv2_numpy.cpp.o
[100%] Building CXX object modules/python3/CMakeFiles/opencv_python3.dir/__/src2/cv2.cpp.o
[100%] Building CXX object modules/python3/CMakeFiles/opencv_python3.dir/__/src2/cv2_convert.cpp.o
[100%] Building CXX object modules/python3/CMakeFiles/opencv_python3.dir/__/src2/cv2_highgui.cpp.o
In file included from /root/opencv/build/modules/python_bindings_generator/pyopencv_generated_include.h:90,
                 from /root/opencv/modules/python/src2/cv2.cpp:11:
/root/opencv_contrib/modules/cudacodec/include/opencv2/cudacodec.hpp:386:20: warning: ‘class cv::cudacodec::NVSurfaceToColorConverter’ has virtual functions and accessible non-virtual destructor [-Wnon-virtual-dtor]
  386 | class CV_EXPORTS_W NVSurfaceToColorConverter {
      |                    ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /root/opencv/build/modules/python_bindings_generator/pyopencv_generated_include.h:91,
                 from /root/opencv/modules/python/src2/cv2.cpp:11:
/root/opencv_contrib/modules/cudafeatures2d/include/opencv2/cudafeatures2d.hpp:121:26: warning: ‘virtual void cv::cuda::DescriptorMatcher::clear()’ can be marked override [-Wsuggest-override]
  121 |     CV_WRAP virtual void clear() = 0;
      |                          ^~~~~
/root/opencv_contrib/modules/cudafeatures2d/include/opencv2/cudafeatures2d.hpp:125:26: warning: ‘virtual bool cv::cuda::DescriptorMatcher::empty() const’ can be marked override [-Wsuggest-override]
  125 |     CV_WRAP virtual bool empty() const = 0;
      |                          ^~~~~
[100%] Linking CXX executable ../../bin/opencv_perf_superres
[100%] Built target opencv_perf_superres
```

and some more info about the errors:

```
[100%] Built target opencv_perf_superres
[100%] Linking CXX shared module ../../lib/python3/cv2.cpython-310-aarch64-linux-gnu.so
[100%] Built target opencv_python3
CMake Deprecation Warning at CMakeLists.txt:25 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Detected processor: aarch64
-- Looking for ccache - not found
-- WARNING: Option ENABLE_NEON='ON' is deprecated and should not be used anymore
--          Behaviour of this option is not backward compatible
--          Refer to 'CPU_BASELINE'/'CPU_DISPATCH' CMake options documentation
-- Could NOT find AVIF (missing: AVIF_LIBRARY AVIF_INCLUDE_DIR) 
Cleaning INTERNAL cached variable: WEBP_LIBRARY
Cleaning INTERNAL cached variable: WEBP_INCLUDE_DIR
-- Could NOT find OpenJPEG (minimal suitable version: 2.0, recommended version >= 2.3.1). OpenJPEG will be built from sources
-- OpenJPEG: VERSION = 2.5.0, BUILD = opencv-4.12.0-dev-openjp2-2.5.0
-- OpenJPEG libraries will be built from sources: libopenjp2 (version "2.5.0")
-- NVCUVID: Header not found, WITH_NVCUVID requires Nvidia decoding library header /usr/local/cuda;/usr/local/cuda/include/nvcuvid.h
-- NVCUVENC: Header not found, WITH_NVCUVENC requires Nvidia encoding library header /usr/local/cuda;/usr/local/cuda/include/nvEncodeAPI.h
-- CUDA detected: 12.6
-- CUDA: Using CUDA_ARCH_BIN=8.7
-- CUDA: NVCC target flags -gencode;arch=compute_87,code=sm_87;-D_FORCE_INLINES;-gencode;arch=compute_87,code=compute_87
-- LAPACK(Atlas): LAPACK_LIBRARIES: /usr/lib/aarch64-linux-gnu/liblapack.so;/usr/lib/aarch64-linux-gnu/libcblas.so;/usr/lib/aarch64-linux-gnu/libatlas.so
-- LAPACK(Atlas): Support is enabled.
-- Could NOT find Java (missing: Java_JAVA_EXECUTABLE Java_JAR_EXECUTABLE Java_JAVAC_EXECUTABLE Java_JAVAH_EXECUTABLE Java_JAVADOC_EXECUTABLE) 
-- Could NOT find JNI (missing: JAVA_INCLUDE_PATH JAVA_INCLUDE_PATH2 AWT JVM) 
-- VTK is not found. Please set -DVTK_DIR in CMake to VTK build directory, or to VTK install subdirectory with VTKConfig.cmake file
CMake Deprecation Warning at 3rdparty/carotene/hal/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


CMake Deprecation Warning at 3rdparty/carotene/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Checking for module 'libavresample'
--   No package 'libavresample' found
-- freetype2:   YES (ver 24.1.18)
-- harfbuzz:    YES (ver 2.7.4)
-- Julia not found. Not compiling Julia Bindings. 
-- Module opencv_ovis disabled because OGRE3D was not found
-- CERES support is disabled. Ceres Solver for reconstruction API is required.
-- Tesseract:   YES (ver 4.1.1)
-- Allocator metrics storage type: 'int'
-- Excluding from source files list: modules/imgproc/src/corner.avx.cpp
-- Excluding from source files list: modules/imgproc/src/imgwarp.avx2.cpp
-- Excluding from source files list: modules/imgproc/src/imgwarp.lasx.cpp
-- Excluding from source files list: modules/imgproc/src/imgwarp.sse4_1.cpp
-- Excluding from source files list: modules/imgproc/src/resize.avx2.cpp
-- Excluding from source files list: modules/imgproc/src/resize.lasx.cpp
-- Excluding from source files list: modules/imgproc/src/resize.sse4_1.cpp
-- Registering hook 'INIT_MODULE_SOURCES_opencv_dnn': /root/opencv/modules/dnn/cmake/hooks/INIT_MODULE_SOURCES_opencv_dnn.cmake
-- opencv_dnn: filter out ocl4dnn source code
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.avx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.avx2.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.avx512_skx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.rvv.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.lasx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/int8layers/layers_common.avx2.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/int8layers/layers_common.avx512_skx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/int8layers/layers_common.rvv.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/int8layers/layers_common.lasx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_block.avx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_block.avx2.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_depthwise.avx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_depthwise.avx2.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_depthwise.rvv.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_depthwise.lasx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/fast_gemm_kernels.avx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/fast_gemm_kernels.avx2.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/fast_gemm_kernels.lasx.cpp
-- Excluding from source files list: modules/features2d/src/fast.avx2.cpp
-- imgcodecs: OpenEXR codec is disabled in runtime. Details: https://github.com/opencv/opencv/issues/21326
CMake Warning at /root/opencv_contrib/modules/cudacodec/CMakeLists.txt:26 (message):
  cudacodec::VideoReader requires Nvidia Video Codec SDK.  Please resolve
  dependency or disable WITH_NVCUVID=OFF


CMake Warning at /root/opencv_contrib/modules/cudacodec/CMakeLists.txt:30 (message):
  cudacodec::VideoWriter requires Nvidia Video Codec SDK.  Please resolve
  dependency or disable WITH_NVCUVENC=OFF


-- highgui: using builtin backend: GTK3
-- Use autogenerated whitelist /root/opencv/build/modules/js_bindings_generator/whitelist.json
-- CERES support is disabled. Ceres Solver for reconstruction API is required.
-- Building with NVIDIA Optical Flow API 2.0
-- Found 'misc' Python modules from /root/opencv/modules/python/package/extra_modules
-- Found 'mat_wrapper;utils' Python modules from /root/opencv/modules/core/misc/python/package
-- Found 'gapi' Python modules from /root/opencv/modules/gapi/misc/python/package
-- Found 'misc' Python modules from /root/opencv/modules/python/package/extra_modules
-- Found 'mat_wrapper;utils' Python modules from /root/opencv/modules/core/misc/python/package
-- Found 'gapi' Python modules from /root/opencv/modules/gapi/misc/python/package
CMake Warning at cmake/OpenCVGenSetupVars.cmake:54 (message):
  CONFIGURATION IS NOT SUPPORTED: validate setupvars script in install
  directory
Call Stack (most recent call first):
  CMakeLists.txt:1182 (include)


-- 
-- General configuration for OpenCV 4.12.0-dev =====================================
--   Version control:               4b2a33a
-- 
--   Extra modules:
--     Location (extra):            /root/opencv_contrib/modules
--     Version control (extra):     2af5458
-- 
--   Platform:
--     Timestamp:                   2025-01-26T21:40:50Z
--     Host:                        Linux 5.15.148-tegra aarch64
--     CMake:                       3.31.4
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/gmake
--     Configuration:               RELEASE
--     Algorithm Hint:              ALGO_HINT_ACCURATE
-- 
--   CPU/HW features:
--     Baseline:                    NEON FP16 NEON_DOTPROD NEON_FP16
--       requested:                 DETECT
--       required:                  NEON
--     Dispatched code generation:  NEON_BF16
--       requested:                 NEON_FP16 NEON_BF16 NEON_DOTPROD
--       NEON_BF16 (0 files):       + NEON_BF16
-- 
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ standard:                11
--     C++ Compiler:                /usr/bin/c++  (ver 11.4.0)
--     C++ flags (Release):         -march=native -mtune=native   -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG
--     C++ flags (Debug):           -march=native -mtune=native   -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/cc
--     C flags (Release):           -march=native -mtune=native   -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -fvisibility=hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG
--     C flags (Debug):             -march=native -mtune=native   -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -fvisibility=hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):      -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
--     Linker flags (Debug):        -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
--     ccache:                      NO
--     Precompiled headers:         NO
--     Extra dependencies:          m pthread cudart_static dl rt nppc nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps cublas cudnn cufft -L/usr/local/cuda/lib64 -L/usr/lib/aarch64-linux-gnu
--     3rdparty dependencies:
-- 
--   OpenCV modules:
--     To be built:                 alphamat aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev datasets dnn dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy gapi hdf hfs highgui img_hash imgcodecs imgproc intensity_transform line_descriptor mcc ml objdetect optflow phase_unwrapping photo plot python3 quality rapid reg rgbd saliency sfm shape signal stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab wechat_qrcode xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    world
--     Disabled by dependency:      -
--     Unavailable:                 cannops cvv fastcv java julia matlab ovis python2 viz
--     Applications:                perf_tests apps
--     Documentation:               NO
--     Non-free algorithms:         YES
-- 
--   GUI:                           GTK3
--     GTK+:                        YES (ver 3.24.33)
--     VTK support:                 NO
-- 
--   Media I/O: 
--     ZLib:                        /usr/lib/aarch64-linux-gnu/libz.so (ver 1.2.11)
--     JPEG:                        /usr/lib/aarch64-linux-gnu/libjpeg.so (ver 80)
--     WEBP:                        build (ver decoder: 0x0209, encoder: 0x020f, demux: 0x0107)
--     AVIF:                        NO
--     PNG:                         /usr/lib/aarch64-linux-gnu/libpng.so (ver 1.6.37)
--     TIFF:                        build (ver 42 - 4.6.0)
--     JPEG 2000:                   build (ver 2.5.0)
--     OpenEXR:                     build (ver 2.3.0)
--     GIF:                         NO
--     HDR:                         YES
--     SUNRASTER:                   YES
--     PXM:                         YES
--     PFM:                         YES
-- 
--   Video I/O:
--     FFMPEG:                      YES
--       avcodec:                   YES (58.134.100)
--       avformat:                  YES (58.76.100)
--       avutil:                    YES (56.70.100)
--       swscale:                   YES (5.9.100)
--       avresample:                NO
--     GStreamer:                   YES (1.20.3)
--     v4l/v4l2:                    YES (linux/videodev2.h)
-- 
--   Parallel framework:            TBB (ver 2021.11 interface 12110)
-- 
--   Trace:                         YES (with Intel ITT(3.25.4))
-- 
--   Other third-party libraries:
--     Lapack:                      YES (/usr/lib/aarch64-linux-gnu/liblapack.so /usr/lib/aarch64-linux-gnu/libcblas.so /usr/lib/aarch64-linux-gnu/libatlas.so)
--     Eigen:                       YES (ver 3.4.0)
--     Custom HAL:                  YES (carotene (ver 0.0.1))
--     Protobuf:                    build (3.19.1)
--     Flatbuffers:                 builtin/3rdparty (23.5.9)
-- 
--   NVIDIA CUDA:                   YES (ver 12.6, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             87
--     NVIDIA PTX archs:            87
-- 
--   cuDNN:                         YES (ver 9.4.0)
-- 
--   Python 3:
--     Interpreter:                 /usr/local/bin/python3 (ver 3.10.12)
--     Libraries:                   /usr/lib/aarch64-linux-gnu/libpython3.10.so (ver 3.10.12)
--     Limited API:                 NO
--     numpy:                       /usr/local/lib/python3.10/dist-packages/numpy/core/include (ver 1.26.4)
--     install path:                /usr/lib/python3/dist-packages/cv2/python-3.10
-- 
--   Python (for build):            /usr/local/bin/python3
-- 
--   Java:                          
--     ant:                         NO
--     Java:                        NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
-- 
--   Install to:                    /usr
-- -----------------------------------------------------------------
-- 
-- Configuring done (8.4s)
-- Generating done (2.3s)
-- Build files have been written to: /root/opencv/build
```
