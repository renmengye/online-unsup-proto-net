# Script to install with conda.
#
# Author: Mengye Ren (mren@cs.toronto.edu)
#
# Modify the path below:
# ------------------------------------------------
#CUDA=/usr/local/cuda-11.2
CUDA=/pkgs/cuda-11.0
OPEN_MPI=/pkgs/openmpi-4.0.0
#NCCL=/home/mren/code/nccl/build
#NCCL=/pkgs/nccl_2.6.4-1+cuda10.1_x86_64
NCCL=/pkgs/nccl_2.8.3-1+cuda11.0_x86_64
# ------------------------------------------------

# export PATH=$PATH:$OPEN_MPI/bin
pip install -I numpy \
        tensorflow==2.4.0 \
        keras \
        h5py
HOROVOD_GPU_ALLREDUCE=NCCL \
        HOROVOD_GPU_BROADCAST=NCCL \
        HOROVOD_WITH_TENSORFLOW=1 \
        HOROVOD_CUDA_HOME=$CUDA \
        HOROVOD_CUDA_INCLUDE=$CUDA/include \
        HOROVOD_CUDA_LIB=$CUDA/lib64 \
        HOROVOD_NCCL_HOME=$NCCL \
        HOROVOD_NCCL_INCLUDE=$NCCL/include HOROVOD_NCCL_LIB=$NCCL/lib \
        pip install -I --no-cache-dir horovod==v0.22.1
pip install tqdm \
        opencv-python==3.4.2.17 \
        pandas \
        tensorflow_addons==0.13.0 \
        tensorflow_probability==0.11.0 \
        scikit-learn matplotlib

bash -c "sed -i 's/if JAX_MODE/if True/' /h/mren/condaenvs/tf2.5/lib/python3.7/site-packages/tensorflow_probability/python/bijectors/sigmoid.py"
# bash -c "sed -i 's/if JAX_MODE/if True/' /usr/local/lib/python3.7/dist-packages/tensorflow_probability/python/bijectors/sigmoid.py"
# source setup_environ.sh
# ln -s $OUTPUT_DIR ./results
# ln -s $DATA_DIR ./data

