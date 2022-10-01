# online-unsup-proto-net

Code for our paper
*Online Unsupervised Learning of Visual Representations and Categories* 
[[arxiv](https://arxiv.org/abs/2109.05675)]

## System Requirements

Our code is tested on Ubuntu 18.04 with GPU capability. We provide docker
files for reproducible environments. We recommend at least 20GB CPU memory and
11GB GPU memory. Our code is based on TensorFlow 2.

## Installation Using Docker (Recommended)

1. Install `protoc` from
   [here](http://google.github.io/proto-lens/installing-protoc.html).

2. Run `make` to build proto buffer configuration files.

3. Install `docker` and `nvidia-docker`.

4. Build the docker container using `./build_docker.sh`.

5. Modify the environment paths. You need to change `DATA_DIR` and `OURPUT_DIR`
   in `setup_environ.sh`. `DATA_DIR` is the main folder where datasets are
   placed and `OUTPUT_DIR` is the main folder where training models are saved.

## Installation Using Conda

1. Install `protoc` from
   [here](http://google.github.io/proto-lens/installing-protoc.html).

2. Run `make` to build proto buffer configuration files.

3. Modify the environment paths. You need to change `DATA_DIR` and `OURPUT_DIR`
   in `setup_environ.sh`. `DATA_DIR` is the main folder where datasets are
   placed and `OUTPUT_DIR` is the main folder where training models are saved.

4. Create a conda environment:
```
conda create -n oupn python=3.6
conda activate oupn
conda install pip
```

5. Install CUDA 10.1

6. Install OpenMPI 4.0.0

7. Install NCCL 2.6.4 for CUDA 10.1

8. Modify installation paths in `install.sh`

9. Run `install.sh` 

## Setup Datasets

1. To download the RoamingRooms dataset, please visit
   [here](https://github.com/renmengye/oc-fewshot-public) for
   more information.

2. To download the SAYCam dataset, please vsit
   [here](https://nyu.databrary.org/volume/564). You need to request permission
   from the data owners.

3. To download the Omniglot dataset, run `script/download_omniglot.sh`. This
   script will download the Omniglot dataset to `DATA_DIR`.


## RoamingRooms Experiments
```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.oc_fewshot \
  --config {MODEL_CONFIG_PROTOTXT} \
  --data configs/episodes/{EPISODE_CONFIG}.prototxt \
  --env configs/environ/roaming-rooms-docker.prototxt \
  --tag {TAG}
```

Run contrastive learning baselines (IID or non-IID)
```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.pretrain_contrastive \
--config=configs/models/roaming-rooms/simclr-ch4-b50.prototxt \
--env=configs/environ/roaming-rooms-contrastive.prototxt \
--tag={TAG} --sim [--non_iid] --max_len 50
```

Evaluation (instance unsupervised):
```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.oc_fewshot
--config=configs/models/roaming-rooms/opn-v2-siam-ch4.prototxt
--data=configs/episodes/roaming-rooms/roaming-rooms-100-siam-map.prototxt
--env=configs/environ/roaming-rooms-docker.prototxt
--tag={TAG} --max_classes=150 --eval --select_threshold
```

Evaluation (instance supervised):
```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.oc_fewshot
--config=configs/models/roaming-rooms/opn-v2-siam-ch4.prototxt
--data=configs/episodes/roaming-rooms/roaming-rooms-100.prototxt
--env=configs/environ/roaming-rooms-docker.prototxt
--tag={TAG} --max_classes=150 --eval
```

Evaluation (semantic):
```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.eval_contrastive \
--config configs/models/roaming-rooms/readout.prototxt \
--env configs/environ/roaming-rooms-docker.prototxt \
--tag {TAG} --nepoch 20 \
--eval_type readout --last --lr 0.001 --optimizer adam
```

## SAYCam Experiments
```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.oc_fewshot \
  --config {MODEL_CONFIG_PROTOTXT} \
  --data configs/episodes/{EPISODE_CONFIG}.prototxt \
  --env configs/environ/say-cam-2880s-docker.prototxt \
  --tag {TAG}
```

Evaluation (semantic):
```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.eval_contrastive \
 --config=configs/models/say-cam/readout.prototxt \
--env=configs/environ/say-cam-labeled-docker.prototxt \
--reload=[RELOAD_DIR] \
--eval \
--eval_type=readout_v2
```

## Pretrained Checkpoints

RoamingRooms: [link](https://drive.google.com/file/d/1Cr8fn6l6BqHBU6CkvOzlgYEwKiT-r4MH/view?usp=sharing)

SAYCam: [link](https://drive.google.com/file/d/1U8ToKGbBAWdHBeQ8RMPSMmdSMXgby7c3/view?usp=sharing)


## Citation

If you use our code, please consider cite the following:
* Mengye Ren, Tylor R. Scott, Michael L. Iuzzolino, Michael C. Mozer and Richard S. Zemel.
  Online Unsupervised Learning of Visual Representations and Categories. 2021.

```
@article{ren21oupn,
  author    = {Mengye Ren and
               Tyler R. Scott and
               Michael L. Iuzzolino and
               Michael C. Mozer and
               Richard S. Zemel},
  title     = {Online Unsupervised Learning of Visual Representations and Categories},
  journal   = {CoRR},
  volume    = {abs/2109.05675},
  year      = {2021}
}
```
