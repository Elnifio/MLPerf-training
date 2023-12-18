# RGAT Example on IGBH

---

- GraphLearn-Torch's RGAT [training script](https://github.com/alibaba/graphlearn-for-pytorch/tree/main/examples/igbh). 
- IGBH Datasets and [training script](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/tree/main). 

This repository is adapted from IGB's official training code and serves as an **example DGL implementation** for MLPerf GNN benchmark. 

### Environments

Base image: `nvcr.io/nvidia/dgl:23.07-py3`. The following packages are additionally required: 

- NVIDIA DLLogger ([repository](https://github.com/NVIDIA/dllogger))

Alternatively, the environment can be directly built with `docker build -f Dockerfile -t <tag-here> .`


### Dataset Preparation

Please prepare the dataset using GLT's pre-processing script, including FP16 conversion using [`compress_graph.py`](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/igbh/compress_graph.py) and train/eval node indices generation with [`split_seeds.py`](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/igbh/split_seeds.py). 

### Training

Currently only single-node training is supported. To run the single-node multi-GPU training script: 

```bash
COMMAND=(
python3 train_multi_hetero.py 
# Dataset configs
--path $DATASET_PATH --dataset_size full 
--num_classes 2983 
# Currently only FP16 codepath is tested.
# Please refer to pre-processing section for more details. 
--in_memory 1 --use_fp16 
--layout csc --use_uva 
# Configurable HPs
--batch_size 512 --learning_rate 0.001 
--epochs 1 
# Model configurations
# Notice that DGL fanout is reversed, 
# so 5,10,15 is equivalent to GLT's 15,10,5
--fan_out 5,10,15 --hidden_channels 512 --num_heads 4 
--add_timer --in_epoch_eval_times 4 
--gpu_devices $GPU_DEVICES
)

${COMMAND[@]}
```

### Logs

Logs will be exported under path `/results` as JSON formats. 
