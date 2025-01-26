# Install-OpenCV-Jetson-Nano
![output image]( https://qengineering.eu/images/LogoOpenJetsonGitHub.webp )

## OpenCV installation script for a Jetson (Orin) Nano

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/>

This is the full setup of OpenCV with CUDA and cuDNN support for the Jetson Nano.<br/>
The script will detect if you are working on a regular Nano, or with the new Orin Nano.<br>
For more information see [Q-engineering - Install OpenCV Jetson Nano](https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html)

------------

## Installing OpenCV.
Your Nano's default memory (4 GB RAM + 2 GB swap) is not enough for a quick build.<br/>
In this case, the compilation will be done by 1 core, which will take a long time.<br/>
It would be best if you had more memory allocated to your Nano for the fast 4-core build.<br/>
```
# check your total memory (RAM + swap) for a fast build. You need at least a total of:
# OpenCV 4.10.0 -> 8.5 GB!
# OpenCV 4.9.0 -> 8.5 GB!
# OpenCV 4.8.0 -> 8.5 GB!
# OpenCV 4.7.0 -> 8.5 GB!
# OpenCV 4.6.0 -> 8.5 GB!
# OpenCV 4.5.5 -> 8.5 GB!
# OpenCV 4.5.4 -> 8.5 GB!
# OpenCV 4.5.3 -> 8.5 GB!
# OpenCV 4.5.2 -> 8.5 GB!
# OpenCV 4.5.1 -> 6.5 GB
# OpenCV 4.5.0 -> 6.5 GB
# If not, enlarge your swap space as explained in the guide, 
# or only 1 core is used for the compilation.
$ free -m

$ wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-10-0.sh  --change to this repo's file location instead.
$ sudo chmod 755 ./OpenCV-4-10-0.sh
$ ./OpenCV-4-10-0.sh
```
:point_right: Don't forget to reset your swap memory afterwards.

------------

If you want to beautify OpenCV with the Qt5 GUI, you need to 👈 [kairin note] - qt5-default has been dropped from debian due to being outdated.
- $ sudo apt-get install qt5-default
- Set the -D WITH_QT=**ON** \ (± line 62) in the script<br/>
 
before running the script on your Nano

------------

OpenCV will be installed in the `/usr` directory, all files will be copied to the following locations:<br/>

- `/usr/bin` - executable files<br/>
- `/usr/lib/aarch64-linux-gnu` - libraries (.so)<br/>
- `/usr/lib/aarch64-linux-gnu/cmake/opencv4` - cmake package<br/>
- `/usr/include/opencv4` - headers<br/>
- `/usr/share/opencv4` - other files (e.g. trained cascades in XML format)<br/>

------------
![image](https://github.com/user-attachments/assets/a30a7ed0-90ea-4cd0-b378-21f3e94a3ea0)
in the install file, the sm_5.3 is in reference to the compute capability. i took a long time to figure out what it is referring to.
referencing to this https://forums.developer.nvidia.com/t/where-can-i-find-these-numbers-in-cuda-arch-bin/283328 and reading up on this https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability where you can find the info in screenshot here https://developer.nvidia.com/cuda-gpus

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
