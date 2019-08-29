#!/bin/bash

echo "--------------CUDA VARIABLES---------------"

#CUDA base path
#export PROJECT_CUDA_PATH="/usr/local/cuda/" #Global v9
export PROJECT_CUDA_PATH="/mnt/sda2/sofoklis/cuda_drivers/cuda-10.0" #My Cuda drivers V10
echo "Env Variable :PROJECT_CUDA_PATH"       
echo "Path         :$PROJECT_CUDA_PATH"

#CUDA bin path
export PROJECT_CUDA_BIN_PATH=$PROJECT_CUDA_PATH/bin
echo "Env Variable :PROJECT_CUDA_BIN_PATH"       
echo "Path         :$PROJECT_CUDA_BIN_PATH"

#CUDA include path
export PROJECT_CUDA_INCLUDE_PATH=$PROJECT_CUDA_PATH/include
echo "Env Variable :PROJECT_CUDA_INCLUDE_PATH"
echo "Path         :$PROJECT_CUDA_INCLUDE_PATH"

#CUDA lib path
export PROJECT_CUDA_LIB_PATH=$PROJECT_CUDA_PATH/lib64
echo "Env Variable :PROJECT_CUDA_LIB_PATH"
echo "Path         :$PROJECT_CUDA_LIB_PATH"

#CUDA nvcc path
export PROJECT_CUDA_NVCC_PATH=$PROJECT_CUDA_PATH/bin/nvcc
echo "Env Variable :PROJECT_CUDA_NVCC_PATH"
echo "Path         :$PROJECT_CUDA_NVCC_PATH"


#echo "--------------GCC VARIABLES---------------"

#Tried older version for a configuration problem (same results as default 7.3)
#GCC 5.4 location
#export PROJECT_GCC_BIN_PATH=/opt/gcc-5.4.0/bin
#echo "Env Variable :PROJECT_GCC_BIN_PATH"
#echo "Path         :$PROJECT_GCC_BIN_PATH"


echo "--------------EXTRA SPACE---------------"

#Location of project extension folder (driver, data etc)
export EXTRA_SPACE_PATH=/mnt/sda2/sofoklis/subquery_extra_space
echo "Env Variable :EXTRA_SPACE_PATH"
echo "Path         :$EXTRA_SPACE_PATH"


#Location of project extension folder on SSD (data etc)
export FAST_EXTRA_SPACE_PATH=/mnt/ssd/sofoklis/subquery_extra_space_ssd
echo "Env Variable :FAST_EXTRA_SPACE_PATH"
echo "Path         :$FAST_EXTRA_SPACE_PATH"



echo "--------------STD ENV VARIABLES---------------"

#Note: Also update standard env variables

#CUDA v10.0 and GCC 5.4.0
#export PATH=$PROJECT_GCC_BIN_PATH:$PROJECT_CUDA_BIN_PATH:${PATH}
#echo "Env Variable :PATH"
#echo "Path         :$PATH"

#CUDA v10.0 and GCC Default (7.3.0)
export PATH=$PROJECT_CUDA_BIN_PATH:${PATH}
echo "Env Variable :PATH"
echo "Path         :$PATH"

export LD_LIBRARY_PATH=$PROJECT_CUDA_LIB_PATH:$LD_LIBRARY_PATH
echo "Env Variable :LD_LIBRARY_PATH"
echo "Path         :$LD_LIBRARY_PATH"

