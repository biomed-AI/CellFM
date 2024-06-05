import os
import gc
import time
import math
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
import multiprocessing as mp
from tqdm import tqdm,trange
from mindspore import nn,ops
from functools import partial
from multiprocessing import Process,Pool
from scipy.sparse import csr_matrix as csm
class Prepare():
    def __init__(
        self,pad_len,pad=1,zero_len=None,
        mask_ratio=0.2,random=False,cut=None
    ):
        self.zero_len=zero_len
        self.n_genes=24078
        self.mask_ratio=mask_ratio
        self.pad_len=pad_len
        self.bern=partial(np.random.binomial,p=0.5)
        self.beta=partial(np.random.beta,a=2,b=2)
        self.bino=np.random.binomial
        self.pad=pad
        self.cut=min(pad_len,(cut or pad_len))
        self.random=random
        self.empty_gene=np.zeros(self.n_genes+1,np.float32)
    def normalize(self,data,read):
        data=np.log1p(data/read*1e4).astype(np.float32)
        return data,read
    def zero_idx(self,data):
        seq_len=len(data)
        one=(data!=0).astype(np.float32)
        zero=np.zeros(self.pad_len-seq_len,np.float32)
        zero_mask=np.concatenate([one,zero])
        return data,zero_mask
    def zero_mask(self,seq_len):
        zero_len=self.pad_len-seq_len
        unmasked=np.ones(zero_len,np.float32)
        pad=np.zeros(zero_len,np.float32)
        pad=np.stack([unmasked,pad],1)
        l=int(self.mask_ratio*min(seq_len,zero_len))
        mask=np.random.choice(np.arange(zero_len),l,replace=False)
        zero_mask=np.zeros(zero_len,np.float32)
        if not self.random:
            zero_mask[mask[:int(0.8*l)]]=1
        else:
            zero_mask[mask]=1
        pad[mask]=0
        return pad,zero_mask
    def mask(self,nzdata):
        seq_len=len(nzdata)
        l=int(self.mask_ratio*seq_len)
        mask=np.arange(seq_len)
        unmasked=np.ones_like(nzdata)
        nzdata=np.stack([unmasked,nzdata],1)
        if l>0:
            mask=np.random.permutation(seq_len)[:l]
            if not self.random:
                nzdata[mask[:int(0.8*l)]]=0
            else: 
                nzdata[mask]=0
        mask_gene=np.zeros(seq_len,np.float32)
        mask_gene[mask]=1
        return nzdata,mask_gene
    def pad_gene(self,data,z_data):
        return np.concatenate((data,z_data))
    def pad_zero(self,data):
        shape=(self.pad_len-data.shape[0],*data.shape[1:])
        pad=np.zeros(shape,data.dtype)
        data=np.concatenate((data,pad),0)
        return data
    def seperate(self,raw_data):
        nonz=raw_data.nonzero()[0]
        zero=np.where(raw_data==0)[0]
        return raw_data,nonz,zero
    def compress(self,data,idx):
        return data,data[idx],idx
    def sample(self,data,nonz,zero):
        cutted=np.array([])
        if len(nonz)>self.cut:
            w=np.log1p(data[nonz])
            w=w/w.sum()
            order=np.random.choice(np.arange(len(nonz)),len(nonz),replace=False,p=w)
            order=nonz[order]
            nonz=np.sort(order[:self.cut])
            cutted=np.sort(order[self.cut:])
        w=None
        l=self.zero_len or (self.pad_len-len(nonz))
        z_sample=np.random.choice(zero,l,replace=False,p=w)
        seq_len=len(nonz)
        return data,nonz,cutted,z_sample,seq_len
    def attn_mask(self,seq_len):
        mask_row=np.zeros(self.pad_len+self.pad)
        mask_row[:seq_len+self.pad]=1
        return mask_row.astype(np.float32)
    
class SCrna():
    def __init__(self,path,data,filt_len=(200,2048),prep=False):
        min_genes,max_genes=filt_len
        suffix=path.split('.')[-1]
        if suffix=='h5ad':
            adata=sc.read_h5ad(f"{path}/{data}")
        else:
            adata=sc.read_10x_h5(f"{path}/{data}")
        self.gene_info=pd.read_csv(f'./gene_info.csv',index_col=0,header=0)
        self.geneset={j:i+1 for i,j in enumerate(self.gene_info.index)}
        gene=np.intersect1d(adata.var_names,self.gene_info.index)
        if len(gene)<min_genes:
            raise Exception('common genes not enough')
        data=adata.X.astype(np.float32)
        T=adata.X.sum(1)
        data=csm(np.round(data/np.maximum(1,T/1e5,dtype=np.float32)))
        data.eliminate_zeros()
        adata.X=data
        
        self.adata=adata[:,gene]
        if prep:
            sc.pp.filter_genes(self.adata,min_cells=1)
            sc.pp.filter_cells(self.adata,min_genes=min_genes)
            sc.pp.filter_cells(self.adata,max_genes=max_genes)
        if len(self.adata)==0:
            raise Exception('samples are filtered')
        print(self.adata.shape)
        self.gene=np.array([self.geneset[i] for i in self.adata.var_names]).astype(np.int32)
        self.T=np.array(self.adata.X.sum(1)).reshape(-1)
        self.data=self.adata.X.astype(np.int32)
    def __len__(self):
        return len(self.adata)
    def __getitem__(self,idx):
        data=np.array(self.data[idx].todense()).reshape(-1)
        T=self.T[idx]
        return data,self.gene,T

def build_dataset(
    data,prep,batch,
    pad_zero=True,
    drop=True,
    label=False,
    shuffle=False,
    rank_size=None,
    rank_id=None,
):
    dataset = ds.GeneratorDataset(
        data, 
        column_names=['data','gene','T']+(['label'] if label else []),
        shuffle=shuffle,
        num_shards=rank_size, 
        shard_id=rank_id
    )
    dataset = dataset.map(
        prep.seperate, input_columns=['data'],
        output_columns=['data', 'nonz','zero']
    )
    dataset = dataset.map(
        prep.sample, input_columns=['data','nonz','zero'],
        output_columns=['data','nonz','cuted','z_sample','seq_len']
    )
    dataset = dataset.map(
        prep.compress, input_columns=['data','nonz'],
        output_columns=['raw_data','raw_nzdata', 'nonz']
    )
    dataset = dataset.map(
        prep.compress, input_columns=['gene','nonz'],
        output_columns=['gene','nonz_gene', 'nonz']
    )
    dataset = dataset.map(
        prep.normalize, input_columns=['raw_nzdata','T'],
    )
    dataset = dataset.map(
        prep.attn_mask, input_columns=['seq_len'],
        output_columns=['zero_idx']
    )
    dataset = dataset.map(
       lambda x:(x,x.copy()), input_columns=['raw_nzdata'],
        output_columns=['raw_nzdata', 'nzdata']
    )
    dataset = dataset.map(
        prep.mask, input_columns=['nzdata'],
        output_columns=['masked_nzdata', 'mask_gene']
    )
    dataset = dataset.map(prep.pad_zero, input_columns=['raw_nzdata'])
    dataset = dataset.map(prep.pad_zero, input_columns=['masked_nzdata'])
    dataset = dataset.map(prep.pad_zero, input_columns=['nonz_gene'])
    dataset = dataset.map(prep.pad_zero, input_columns=['mask_gene'])
    dataset=dataset.project(
        columns=[
            'raw_nzdata','masked_nzdata','nonz_gene','mask_gene','zero_idx'
        ]+(['label'] if label else [])
    )
    dataset = dataset.batch(
        batch,
        num_parallel_workers=4, 
        drop_remainder=drop, 
    )
    return dataset