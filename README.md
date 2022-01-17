# NVIDIA Driver Installation

Problems appeared during the NVIDIA Driver installation. After installing the driver and rebooting the computer, the computer would get stuch on a black/logo screen. To solve this problem, a [sequence of commands were found to install the nvidia driver](https://forums.developer.nvidia.com/t/ubuntu-20-04-boots-to-black-screen/184390) without needing to reboot the system.

To do this run `sudo bash install_driver.sh`

Or manually:

```
sudo rm /etc/X11/xorg.conf
sudo apt install nvidia-prime
sudo prime-select nvidia
sudo apt install nvidia-driver-460
sudo modprobe nvidia
sudo nvidia-smi -pm 1
```

After every reboot we need to remove the nvidia driver by running `sudo apt-get remove --purge "nvidia*"` and after that execute the commands in install_driver.sh again.

Run `nvidia-smi` to check the installation of the driver

## CUDA Installation

Since the driver version installed was the 460 version we're are going to install CUDA toolkit version 11.2. We opted for the Runfile installation of CUDA.

### Download CUDA Toolkit
Go to the [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive) and download your wanted version (in our case 11.2.0). We downloaded the runfile (.run)


Additionally by clicking the link "Versioned Online Documentation" you will be directed to the specific CUDA version installation instructions.

### CUDA Runfile Installation

For the runfile installation we followed the instructions [here](https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-linux/index.html#runfile). 
We performed the pre-installation actions and disabled the nouveau drivers as instructed. 
After that run `sudo sh <runfile_dowloaded>.run`.
- Accept the EULA terms and uncheck the Driver installation box since we already have a driver. 
- Please verify that the option to create a symbolic link in /usr/local/cuda is activated. 
- After that follow the [post-installation instructions](https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-linux/index.html#post-installation-actions) about environment setup.

To check if the installation was sucessful run `nvcc --version`
# OpenGaze Installation Procedure

## Openface installation

To install Openface first clone the [repository](https://github.com/TadasBaltrusaitis/OpenFace) from Github through `git clone https://github.com/TadasBaltrusaitis/OpenFace.git`

To install OpenFace on our machine we needed to make some alterations to the script install.sh provided by Openface, these were:
1. Add in line 61 `export LD_LIBRARY_PATH=/home/vislab/FFMpeg/lib/:$LD_LIBRARY_PATH`
2. Add in line 62 `export PKG_CONFIG_PATH=/home/vislab/FFMpeg/lib/pkgconfig:$PKG_CONFIG_PATH`
3. Alter line 70 to `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_SHARED_LIBS=ON DBUILDTIFF=ON ..`

After making the alterations to install.sh just run `yes | bash install.sh`

After installing, dowload the models and test it by running
```
bash download_models.sh
cd build/
cp ../lib/local/LandmarkDetector/model/patch_experts/cen_patches_0.25_of.dat ./bin/model/patch_experts/
cp ../lib/local/LandmarkDetector/model/patch_experts/cen_patches_0.35_of.dat ./bin/model/patch_experts/
cp ../lib/local/LandmarkDetector/model/patch_experts/cen_patches_0.50_of.dat ./bin/model/patch_experts/
cp ../lib/local/LandmarkDetector/model/patch_experts/cen_patches_1.00_of.dat ./bin/model/patch_experts/
./bin/FaceLandmarkVid -f "../samples/changeLighting.wmv" -f "../samples/2015-10-15-15-14.avi"
```

## Caffe Installation

To install Caffe we use the cmake installation of caffe, which can be found on the [Caffe Website](http://caffe.berkeleyvision.org/installation.html).
We can clone the caffe and Opengaze repository with
 ```
 git clone https://github.com/BVLC/caffe.git
 git clone https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze.git
 ```

We copy the customized opengaze layers before building caffe
 ```
 cp -r opengaze/caffe-layers/include caffe/
 cp -r opengaze/caffe-layers/src caffe/
 ```

We install the caffe dependencies with
```
sudo apt-get install cmake libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libopencv-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libboost-python-dev

sudo apt-get install --no-install-recommends libboost-all-dev
```
Then for the cmake installation we run
```
cd caffe/
mkdir build
cd build
cmake .. -DUSE_CUDNN=0 -DBLAS=Open
```
Then alter the CMakeCache.txt PYTHON_xxxx variables in the following way (we need to use python 3.8 instead of 2.7 since libboost-python in ubuntu 20.04 is for python 3.8)
- PYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3.8
- PYTHON_INCLUDE_DIR:PATH=/usr/include/python3.8
- PYTHON_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libpython3.8.so

Maybe these alterations can be done by altering in caffe/CmakeLists.txt line 35 'python_version "2"' to 'python_version "3"' bypassing the alterations to CMakeCache.txt

Then compile and install
```
make all
make install
```

## OpenGaze Installation

For the OpenGaze installation we just followed the "Compile OpenGaze" instructions [here](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze/-/wikis/Unix-installation).
- CAFFE_INSTALL_DIR = "~/caffe/install"
- OPENFACE_ROOT_DIR = "~/OpenFace"
- OPENGAZE_DIR = "~/opengaze"

To compile and install we just do
```
cd opengaze
mkdir build
cd build
cmake ..
make
sudo make install
```

To test we just do
```
cd exe/
mkdir build
cd build/
cmake ..
make
./bin/GazeVisualization -d -t camera -i 0
./bin/GazeVisualization -d -t video -i <videolocation>
```

# Gaze360 Installation

For the Gaze360 method we need to install Pytorch with Cuda Toolkit 11.2, theerefore we use a cuda environment to perform this


For the Anaconda installation refer to the [Anaconda Installation instructions](https://docs.anaconda.com/anaconda/install/linux/). 
Afte installing Anaconda do:
```
conda create --name gaze_env python=3.7
conda install pytorch torchvision cudatoolkit=11.2 -c conda-forge -c pytorch
```  
Then clone the [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2.git) and [Gaze360](git clone https://github.com/erkil1452/gaze360) repositories:
```
git clone https://github.com/cleardusk/3DDFA_V2.git
git clone https://github.com/erkil1452/gaze360
wget http://gaze360.csail.mit.edu/files/gaze360_model.pth.tar
```
Then install all dependencies of 3DDFA_V2 listed in requirements.txt (except torch and torchvision installed with conda) with 
`pip install matplotlib numpy opencv-python imageio imageio-ffmpeg pyyaml tqdm argparse cython scikit-image scipy onnxruntime gradio`
