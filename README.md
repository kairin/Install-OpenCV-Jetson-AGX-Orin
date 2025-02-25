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

view errors encountered:  [ERRORS](https://github.com/kairin/Install-OpenCV-Jetson-AGX-Orin/blob/main/README-errors.md)

```
-- Installing: /usr/bin/opencv_model_diagnostics
-- Set non-toolchain portion of runtime path of "/usr/bin/opencv_model_diagnostics" to "/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64"
Hit:1 http://ports.ubuntu.com/ubuntu-ports jammy InRelease
Get:2 http://ports.ubuntu.com/ubuntu-ports jammy-updates InRelease [128 kB]
Hit:3 https://apt.repos.intel.com/oneapi all InRelease
Get:4 http://ports.ubuntu.com/ubuntu-ports jammy-backports InRelease [127 kB]
Get:5 http://ports.ubuntu.com/ubuntu-ports jammy-security InRelease [129 kB]
Get:6 http://ports.ubuntu.com/ubuntu-ports jammy-updates/main arm64 Packages [2,521 kB]
Get:7 http://ports.ubuntu.com/ubuntu-ports jammy-updates/universe arm64 Packages [1,470 kB]
Fetched 4,376 kB in 3s (1,534 kB/s)                        
Reading package lists... Done
N: Skipping acquire of configured file 'main/binary-arm64/Packages' as repository 'https://apt.repos.intel.com/oneapi all InRelease' doesn't support architecture 'arm64'
Congratulations!
You've successfully installed OpenCV 4.11.0 on your Nano
root@kkkORIN:/opt# 
```

but it still doesn't solve the following error:

```
root@kkkORIN:/opt/comf# python3 main.py
[START] Security scan
[DONE] Security scan
## ComfyUI-Manager: installing dependencies done.
** ComfyUI startup time: 2025-01-27 06:44:07.526
** Platform: Linux
** Python version: 3.10.12 (main, Jan 17 2025, 14:35:34) [GCC 11.4.0]
** Python executable: /usr/local/bin/python3
** ComfyUI Path: /opt/comf
** ComfyUI Base Folder Path: /opt/comf
** User directory: /opt/comf/user
** ComfyUI-Manager config path: /opt/comf/user/default/ComfyUI-Manager/config.ini
** Log path: /opt/comf/user/comfyui.log
[ComfyUI-Manager] Failed to restore opencv
invalid literal for int() with base 10: '0+6b45caa'
```

and likewise an issue with albumentations suggest that solving opencv would likely help with reinstalling this problem software.

![image]