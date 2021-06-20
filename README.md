# Caffe with cuDNN8
The original version caffe can only work with cuDNN7.x, but since CUDA11.x, only cudnn8 is supported, so I made some changes in code to make caffe able to work with CUDA11.x + cuDNN8.x.

I have made code changes in fo cudnn8:

       cmake/Cuda.cmake
       src/caffe/layers/cudnn_ndconv_layer.cu
       src/caffe/layers/cudnn_conv_layer.cpp
       src/caffe/layers/cudnn_deconv_layer.cpp

This version code can directly work with cuDNN8.x, If you do want to use caffe with cudnn7.x(original caffe), you need to make these changes in the above files:

    1) Open the file “cmake/Cuda.cmake”.  replace "cudnn_version.h" with "cudnn.h" by commenting/uncommenting the lines where they are.
    2) In cudnn_ndconv_layer.cu, cudnn_conv_layer.cpp and cudnn_deconv_layer.cpp  change all "if CUDNN_VERSION_MIN(8, 0, 0)  // 0" to "#if 0 // CUDNN_VERSION_MIN(8, 0, 0)".


# How to use
#### Step 1 Install dependencies
	sudo apt-get install libprotobuf-dev libleveldb-dev  libblas-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
	sudo apt-get install --no-install-recommends libboost-all-dev
	sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
	sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
	sudo apt-get install git cmake build-essential
	sudo pip install graphviz	


#### Step 2 download caffe-cudnn8 source code
	git clone https://github.com/Jeremy-J-J/caffe-cudnn8.git
	
	
#### Step 3 Build settings
	Although I provide the Makefile.config and Makefile that can directly work, but i think you also need to know how it works.
	(1) Create a file named "Makefile.config"
		cp Makefile.config.example Makefile.config
	
	(2) Set the CUDA ARCH
		According to the actual to set.
		CUDA_ARCH := -gencode arch=compute_75,code=sm_75
		
	(3) Set use Opencv3
		OPENCV_VERSION := 3
		
	(4) Set CUDA path
		CUDA_DIR := /usr/local/cuda
		
	(5) Set Python Include
		PYTHON_INCLUDE := /root/anaconda3/include \
						  /root/anaconda3/include/python3.7m \
						  /root/anaconda3/lib/python3.7/site-packages/numpy/core/include
	
	(6) Set Python Library
		PYTHON_LIB := /root/anaconda3/lib
		
	(7) Set the other Include and Library
		INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial /home/0_env/opencv/include
		LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial /home/0_env/opencv/lib
 
                        

		LIBRARIES += glog gflags protobuf leveldb snappy \
					 lmdb boost_system hdf5_hl hdf5 m\
					 opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
	
	(8) Modify the Makefile
		# After LIBRARIES add opencv_imgcodecs
		LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
		
		
#### Step 4 Build
	make all -j32
	make test -j32
	make runtest -j32
	make pycaffe
	

#### Step 5 Configure environment
	vim ~/.bashrc
	export PYTHONPATH=$PYTHONPATH:/home/Jremy-J-J/caffe/python
	
	source ~/.bashrc




