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
