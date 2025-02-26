# Ehancing CellOT with CellMF


### 1. First step
Download the processed [sciplex](https://www.synapse.org/Synapse:syn64414070) dataset in ``./datasets``.

### 2. Second step
Fine-tune CellFM for perturbation trametinib in sciplex dataset:
`````
CUDA_VISIBLE_DEVICES=0 python finetune_sciplex.py trametinib
`````
The fine-tuned CellFM checkpoint is saved in ``./checkpoint``

### 3. Third step
Run CellOT with fine-tuned CellFM:
`````
CUDA_VISIBLE_DEVICES=0 python train.py --outdir ./results/scrna-sciplex3/drug-trametinib/model-cellot-cellfm --config ./configs/tasks/sciplex3.yaml --config ./configs/models/cellot_cellfm.yaml --config.data.target trametinib --config.data.cellfm_emb 1
`````


