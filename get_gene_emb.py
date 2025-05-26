import os
import glob
import time
import math
import datetime
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import pickle as pk
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
from tqdm import tqdm,trange
from mindspore import nn,ops
from scipy.sparse import csr_matrix as csr
from mindspore.amp import FixedLossScaleManager,all_finite,DynamicLossScaleManager
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, Accuracy
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.common.initializer import initializer, XavierNormal
from config import Config
from model import *
# from metrics import *
# from loss_function import *
# from utils import Wrapper,WrapperWithLossScaleCell,AdamWeightDecay
# from utils import WarmCosineDecay,Adam,set_weight_decay
# from itertools import cycle
import sys
# sys.path.append("../")

def freeze_module(module,filter_tag=[None]):
    for param in module.trainable_params():
        x=False
        for tag in filter_tag:
            if tag and tag in param.name:
                x=True
                break
        param.requires_grad = x
        
class WeightedMSE(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sub=P.Sub()
        self.sq=P.Square()
        self.div=P.Div()
        self.cast=P.Cast()
        self.mean=P.ReduceMean(False)
        self.sum=P.ReduceSum(False)
        self.fill_m=P.Mul()
        self.loss_logger=ops.ScalarSummary()
    def construct(self,pred,target,weight=None,mask=None):
        pred = self.cast(pred, ms.float32)
        target = self.cast(target, ms.float32)
        loss=self.sq(self.sub(pred,target))
        if weight is not None:
            weight = self.cast(weight, ms.float32)
            loss=loss*weight
        if mask is not None:
            mask=self.cast(mask,ms.float32)
            loss=self.sum(self.fill_m(loss,mask))
            num=self.sum(mask)
            loss=self.div(loss,num)
        else:
            loss=self.mean(loss)
        self.loss_logger(f'w_MSE',loss)
        return loss

class Encoder(nn.Cell):
    def __init__(self,n_genes,used_gene,cfg,shard=None):
        super().__init__()
        # const
        self.depth=cfg.enc_nlayers
        self.n_genes=n_genes
        # tensor
        # self.gene_emb=ms.Parameter(
        #     initializer(XavierNormal(0.5),[n_genes+1+(-n_genes-1)%8,cfg.enc_dims])
        # )
        self.gene_emb=ms.Parameter(initializer('normal',[n_genes+1+(-n_genes-1)%8,cfg.enc_dims]))
        self.zero_emb=ms.Parameter(initializer('normal',[1,1,cfg.enc_dims]))
        self.pert_token=ms.Parameter(initializer('normal',[3,cfg.enc_dims]))
        self.pert_token[0]=0
        self.pert_token[1]=-self.pert_token[0]
        self.pert_token[2]=0
        self.used_gene=ms.Tensor(used_gene,ms.int32)
        # layer
        self.value_enc=ValueEncoder(cfg.enc_dims)
        self.encoder=nn.CellList([
            RetentionLayer(
                cfg.enc_dims,cfg.enc_num_heads,cfg.enc_nlayers,
                cfg.enc_dropout*i/cfg.enc_nlayers,cfg.lora,
                cfg.recompute
            )
            for i in range(cfg.enc_nlayers)
        ])
        self.value_dec=ValueDecoder(cfg.enc_dims,cfg.dropout)
        # operator
        self.less=P.Less()
        self.one=P.Ones()
        self.zero=P.Zeros()
        self.tile=P.Tile()
        self.sum=P.ReduceSum(True)
        self.softmax=P.Softmax(-1)
        self.gather1=P.Gather()
        self.gather2=P.Gather()
        self.maskmul=P.Mul()
        self.mul=P.Mul()
        self.add=P.Add()
        self.posa=P.Add()
        self.rsqrt=P.Rsqrt()
        self.detach=P.StopGradient()
    def construct(self,expr,if_pert,pert_id,zero_mask):
        b,l=if_pert.shape
        len_scale=self.detach(self.rsqrt(self.sum(zero_mask,-1)).reshape(b,1,1,1))
        gene_emb=self.gather1(self.gene_emb,self.used_gene,0).reshape(1,l,-1)
        pert_emb=self.gather2(self.pert_token,if_pert,0)
        pert_mask=self.less(if_pert,2).reshape(b,l,1).astype(pert_emb.dtype)
        init_emb,unmask=self.value_enc(expr)
        
        expr_emb = init_emb + gene_emb
        
        attn_mask=None
        for i in range(self.depth):
            expr_emb=self.encoder[i](
                expr_emb,
                v_pos=len_scale,
                seq_mask=attn_mask
            )
        return expr_emb#,init_emb



class SCrna():
    def __init__(self,adata,mode=0, pert_names=None):
        self.gene_info=pd.read_csv(f'./csv/new_gene_info.csv',index_col=0,header=0)
        self.geneset={j:i+1 for i,j in enumerate(self.gene_info.index)}
        self.geneset['ctrl']=0
        if pert_names is None:
            self.pert_name=[i for i in adata.obs.condition.unique()]
        else:
            self.pert_name = pert_names
        self.gene=np.array([self.geneset[i] if i in self.geneset else 0 for i in adata.var_names]).astype(np.int32)
        self.pos_map={self.gene[i]:i+1 for i in range(len(self.gene))}
        self.pos_map[0]=0
        self.de_idx={}
        mode=['train','val','test','all'][mode]
        self.mode=mode
        if mode == 'all':
            self.key = set(self.pert_name)
        else:
            self.key=adata[adata.obs.split==mode].obs.condition.unique()
        self.get_cellpair(adata)
        self.idx_map=[]
        for i in self.key:
            idx=0
            for j in self.cell_pair[i]:
                self.idx_map.append((i,idx))
                idx+=1
        print(len(self.idx_map),len(self.gene))
        # self.adata=adata
        self.iteratation=0
    def get_cellpair(self,adata):
        self.cell_pair={i:[] for i in self.pert_name}
        self.ctrl_data=adata[adata.obs.condition=='ctrl']
        ctrl=self.ctrl_data.X.A.astype(np.float32)
        np.random.seed(0)
        for i in self.pert_name:
            if i=='ctrl':
                continue
            u,v=i.split('+')
            pert=adata[adata.obs.condition==f"{u}+{v}"].X.A.astype(np.float32)
            u,v=self.geneset[self.old2new[u]],self.geneset[self.old2new[v]]
            if len(pert) == 0:
                for j in range(100):
                    sample=np.random.randint(0,len(ctrl),1)[0]
                    self.cell_pair[i].append((ctrl[sample],ctrl[sample],[u,v]))
            else:
                for j in range(len(pert)):
                    sample=np.random.randint(0,len(ctrl),1)[0]
                    self.cell_pair[i].append((ctrl[sample],pert[j],[u,v]))
        if 'ctrl' in self.cell_pair.keys():
            for cell in ctrl:
                self.cell_pair['ctrl'].append((cell,cell,[0,0]))
        self.pert_name=list(self.cell_pair.keys())
        self.pert_map={}
        for i in self.pert_name:
            if i=='ctrl':
                self.pert_map[(0,0)]='ctrl'
            else:
                u,v=i.split('+')
                self.pert_map[(self.geneset[self.old2new[u]],self.geneset[self.old2new[v]])]=i
    def rematch_cell(self):
        adata=self.adata
        ctrl=self.ctrl_data.X.A.astype(np.float32)
        idx=0
        self.idx_map.clear()
        for i in self.key:
            if i=='ctrl':
                continue
            u,v=i.split('+')
            pert=adata[adata.obs.condition==f"{u}+{v}"].X.A.astype(np.float32)
            idx=0
            self.cell_pair[i].clear()
            for j in range(len(pert)):
                sample=np.random.randint(0,len(ctrl),1)[0]
                self.cell_pair[i].append((ctrl[sample],pert[j],[self.geneset[u],self.geneset[v]]))
                self.idx_map.append((i,idx))
                idx+=1
    def __len__(self):
        return len(self.idx_map)
    def __getitem__(self,idx):
        pert,i=self.idx_map[idx]
        ctrl_data,pert_data,pert_id=self.cell_pair[pert][i]
        u,v=pert_id
        if_pert=np.zeros(len(self.gene)+1,np.int32)+2
        if_pert[[self.pos_map[u],self.pos_map[v]]]=1
        pert_id=np.array(pert_id,np.int32)
        self.iteratation+=1
        return ctrl_data,pert_data,if_pert[1:],pert_id

def build_dataset(
    data,batch,
    mask_rate=0.2,
    drop=True,
    shuffle=True,
    rank_size=None,
    rank_id=None,
):
    def mask(ctrl_data,pert_id):
        unmask=np.ones_like(ctrl_data)
        ctrl_data=np.stack([unmask,ctrl_data],1)
        mask_idx=np.zeros_like(unmask)
        if pert_id.sum()==0:
            l=len(ctrl_data)
            mask=np.random.choice(np.arange(l),int(l*mask_rate),replace=False)
            ctrl_data[mask]=0
            mask_idx[mask]=1
        else:
            rate=0
            mask_idx[:]=1
        return ctrl_data,pert_id,mask_idx
    dataset = ds.GeneratorDataset(
        data, 
        column_names=["ctrl_data",'pert_data','if_pert','pert_id'],
        shuffle=shuffle,
        num_shards=rank_size, 
        shard_id=rank_id
    )
    dataset = dataset.map(
        lambda x:(x,(x!=0).astype(np.float32)), input_columns=['ctrl_data'],
        output_columns=["ctrl_data",'zero_mask']
    )
    # dataset = dataset.map(
    #     mask, input_columns=['ctrl_data','pert_id'],
    #     output_columns=['ctrl_data','pert_id','mask_idx']
    # )
    
    dataset=dataset.project(
        columns=[
            'ctrl_data','pert_data','if_pert','pert_id','zero_mask'
        ]
    )
    dataset = dataset.batch(
        batch,
        num_parallel_workers=4, 
        drop_remainder=drop, 
    )
    return dataset



def load_model(gpu_id, adata):
    ms.set_context(
        device_target='GPU', 
        mode=ms.GRAPH_MODE,
        device_id=gpu_id,
        # enable_compile_cache= True,
    )
    cfg=Config()
    ms.set_seed(0)
    shard=None
    cfg.enc_dims=1536
    cfg.enc_nlayers=40
    cfg.enc_num_heads=48
    
    gene_info=pd.read_csv(f'./csv/gene_info.csv',index_col=0,header=0)
    geneset={j:i+1 for i,j in enumerate(gene_info.index)}
    # geneset['ctrl'] = 0
    gene=np.array([geneset[i] if i in geneset else 0 for i in adata.var_names]).astype(np.int32)
    encoder=Encoder(len(geneset),gene,cfg)
    para=ms.load_checkpoint('./checkpoint/base_weight.ckpt')
    
    err,_ =ms.load_param_into_net(encoder, para)
    return encoder


class small_dataset():
    def __init__(self,x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.x[idx], np.zeros(self.x.shape[1],np.int32), np.array([0])


def process(x):

    dataloader=build_dataset(
        small_dataset(x),
        len(x),
        mask_rate=0,
        shuffle=False,
        rank_size=None,
        rank_id=None,
        drop=False
    )
    for data in dataloader:
        ctrl_data,pert_data,if_pert,pert_id,zero_mask=data
        return ctrl_data,if_pert,pert_id,zero_mask


def get_embedding(x, encoder):
    # x: batch_size * num_gene (gene expression)
    encoder.set_train(False)
    expr,if_pert,pert_id,zero_mask = process(x)
    expr_emb=encoder(expr,if_pert,pert_id,zero_mask).asnumpy()
    return expr_emb


dataname = 'norman'
adata = sc.read(f'./dataset/norman-1000.h5ad')
adata_ctrl = adata[adata.obs['condition'] == 'ctrl']

device_id=0
encoder = load_model(device_id, adata)
encoder.set_train(False)
batch_size = 2

dataset = small_dataset(adata_ctrl.X.A.astype(np.float32))
dataloader=build_dataset(
    dataset,
    batch_size,
    mask_rate=0,
    shuffle=False,
    rank_size=None,
    rank_id=None,
    drop=False
)


new_emb = []

for data in tqdm(dataloader):
    ctrl_data,pert_data,if_pert,pert_id,zero_mask=data
    expr_emb = encoder(ctrl_data,if_pert,pert_id,zero_mask).asnumpy()
    new_emb.append(expr_emb)
new_emb = np.concatenate(new_emb, axis=0)
print(new_emb.shape)
np.save(f'./dataset/cellFM-norman.npy', new_emb)
