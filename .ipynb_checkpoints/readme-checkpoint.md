![](figures/model.png)

<font size=4> We propose a single-cell foundation model, CellFM.  </font> <br><br>

# CellFM

The official implementation for "**CellFM**".

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Tutorial](#Tutorial)
* [Citation](#Citation)

## Datasets


We provide an easy access to the used datasets in the [synapse](https://www.synapse.org/#!Synapse:syn52559388/files/).



## Installation

To reproduce **CellFM**, we suggest first create a conda environment by:

~~~shell
conda create -n cellFM python=3.9
conda activate cellFM
~~~

and then install the required packages below:

- mindspore=2.2.10
- scanpy=1.10

## Usage

### data preprocessing

To run **CellFM**, we need to first preprocess the data in h5 or h5ad format. The preprocess pipeline for different downstream tasks can refer [process.ipynb](https://github.com/biomed-AI/CellFM/tutorials/process.ipynb). We recommond storing the processed datasets in [datasets](https://github.com/biomed-AI/CellFM/datasets/) directory.

### Train on new dataset

We provided a script [train.py](https://github.com/biomed-AI/CellFM/train.py) for finetuning or training on new datasets. For example, we can train on HumanPBMC dataset with single NPU device by execute:

~~~shell
# Train with single device
python train.py --data HumanPBMC --batch 4 --epoch 5 --load_pretrain [--fp16] [--lora LORA_RANK] [--workpath /DIR/TO/WORKSPACE]
~~~

- --data: dataset name. Note that the dataset should be located in /DIR/TO/WORKSPACE/datasets with h5 or h5ad format.
- --batch: batch size.
- --epoch: the number of training epoch.
- --load_pretrain: load the pretrained weight of **CellFM**.
- --fp16: unnecessary. Set training process under half-precision mode.
- --lora: unnecessary. Using LoRA algorithm to update the weights using LORA_RANK as the hidden dimension of lora block, default 0 i.e. not use LoRA.
- --workpath: unnecessary when train with single device. Set the **absolute directory** of work path, default the directory containing codes.

We also provide a script to apply parallelly training within one node. For the same example, the command below works the same as the command above except it will works on 8 devices while each device handle an input with batch size=4.

```shell
# Train parallelly in one node
bash 1node_train.sh train 4 5 HumanPBMC
```

### Tutorial

### Tutorial 1: Cell Annotation

See [CellAnnotation.ipynb](https://github.com/biomed-AI/CellFM/tutorials/CellAnnotation.ipynb).

### Tutorial 2: Gene Function Prediction

See [GeneFunction.ipynb](https://github.com/biomed-AI/CellFM/tutorials/GeneFunction.ipynb).

### Tutorial 3: Gene Pertubation

See [GenePertubation.ipynb](https://github.com/biomed-AI/CellFM/tutorials/GenePertubation.ipynb).


## Citation

If you find our codes useful, please consider citing our work:

~~~bibtex
@article{CellFM,
  title={CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells},
  author={Yuansong Zeng, Jiancong Xie, Zhuoyi Wei, Yun Su, Ningyuan Shangguan, Shuangyu Yang, Chengyang Zhang, Wenbing Li, Jinbo Zhang, Nan Fang, Hongyu Zhang, Huiying Zhao, Yutong Lu, Jue Fan, Weijiang Yu, and Yuedong Yang},
  journal={},
  year={2024},
}
~~~