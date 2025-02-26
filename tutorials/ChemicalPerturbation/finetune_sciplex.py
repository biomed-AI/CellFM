
import os
import sys
sys.path.append('../..')
import glob
import time
import math
import datetime
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
from tqdm import tqdm,trange
from mindspore import nn,ops
from scipy.sparse import csr_matrix as csm
from mindspore.ops import operations as P
from mindspore.amp import FixedLossScaleManager,all_finite,DynamicLossScaleManager
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.common.initializer import initializer, XavierNormal

from config import *
from metrics import annote_metric, mse_metric
from utils import Wrapper
from data_process import Prepare
from model import *

class CellwiseDecoder2(nn.Cell):
    def __init__(self,in_dims,emb_dims=None,dropout=0.,zero=False,shard=None):
        super().__init__()
        emb_dims=emb_dims or in_dims
        self.act=P.Sigmoid()
        self.sigmoid=P.Sigmoid()
        self.add=P.Add()
        self.tile=P.Tile()
        self.cat=P.Concat(-1)
        # self.map=nn.Dense(in_dims, emb_dims,has_bias=False)

        self.map=nn.SequentialCell(
            nn.Dense(emb_dims,512,has_bias=False),
            nn.Dropout(p=0.15),
            nn.SiLU(),
            nn.Dense(512,798,has_bias=False),
        )

        self.bmm=P.BatchMatMul(transpose_b=False)
        self.mm=P.MatMul(transpose_b=True)
        self.relu=P.ReLU()
        self.zero=zero
        if zero:
            self.zero_logit = nn.Dense(emb_dims, emb_dims)
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.map.matmul.shard(((dp,1),(1,1)))
        self.act.shard(((dp,1,1),))
        self.sigmoid.shard(((dp,1,1),))
        self.bmm.shard(((dp,1,1),(dp,1,1)))
        if self.zero:
            self.zero_logit.matmul.shard(((dp,1),(1,1)))
            self.zero_logit.bias_add.shard(((dp,1),(1,)))
    def construct(self,cell_emb,gene_emb):
        # b=cell_emb.shape[0]
        # query=self.act(self.map(gene_emb))
        # key=cell_emb.reshape(b,-1,1)
        # pred=self.bmm(query,key).reshape(b,-1)
        # if not self.zero:
        #     return pred
        # else:
        #     zero_query=self.zero_logit(gene_emb)
        #     zero_prob=self.sigmoid(self.bmm(zero_query,key)).reshape(b,-1)
        #     return pred,zero_prob

        pred = self.map(cell_emb)
        return pred

class backbone(nn.Cell):
    def __init__(self,n_genes,cfg,shard=None,**kwargs):
        super().__init__()
        # const
        self.depth=cfg.enc_nlayers
        self.if_cls=cfg.label
        self.n_genes=n_genes
        self.add_zero=cfg.add_zero and not cfg.pad_zero
        self.pad_zero=cfg.pad_zero
        # tensor
        self.gene_emb=ms.Parameter(
            initializer(XavierNormal(0.5),[n_genes+1+(-n_genes-1)%8,cfg.enc_dims])
        )
        self.ST_emb=ms.Parameter(
            initializer(XavierNormal(0.5),[1,2,cfg.enc_dims])
        )
        self.cls_token=ms.Parameter(initializer(XavierNormal(0.5),[1,1,cfg.enc_dims]))
        self.zero_emb=ms.Parameter(initializer('zeros',[1,1,cfg.enc_dims]))
        # self.platform_emb=ms.Parameter(initializer('zeros',[cfg.platforms+1,cfg.enc_dims]))
        self.gene_emb[0,:]=0
        # layer
        self.value_enc=ValueEncoder(cfg.enc_dims)
        self.ST_enc=FFN(1,cfg.enc_dims)
        self.encoder=nn.CellList([
            RetentionLayer(
                cfg.enc_dims,cfg.enc_num_heads,cfg.enc_nlayers,
                cfg.enc_dropout*i/cfg.enc_nlayers, cfg.lora,
                cfg.recompute
            )
            for i in range(cfg.enc_nlayers)
        ])
        # self.value_dec=ValueDecoder(cfg.enc_dims,cfg.dropout,zero=self.add_zero)
        self.cellwise_dec=CellwiseDecoder2(50,zero=self.add_zero)
        self.proj2=nn.SequentialCell(
            nn.Dense(1536,512),
            nn.SiLU(),
            nn.Dense(512,50),
            # nn.LeakyReLU(),
            # nn.Dense(cfg.enc_dims,cfg.enc_dims),
        )
        if cfg.label:
            cls_weight=kwargs.get('cls_weight',np.ones(cfg.num_cls))
            self.weight=ms.Tensor(cls_weight,ms.float32)
            self.cluster_emb=ms.Parameter(
                initializer(XavierNormal(0.5),[cfg.num_cls,cfg.enc_dims])
            )
            self.query=RetentionLayer(
                cfg.enc_dims,cfg.enc_num_heads,0.5,
                0,0,False
            )
            self.classifier=nn.Dense(cfg.enc_dims,1,has_bias=False)
            self.proj2=nn.SequentialCell(
                nn.Dense(1536,512),
                nn.SiLU(),
                nn.Dense(512,50),
                # nn.LeakyReLU(),
                # nn.Dense(cfg.enc_dims,cfg.enc_dims),
            )
            if shard is not None:
                dp,mp=shard
                self.classifier.matmul.shard(((dp,1),(1,1)))
                self.proj2[0].matmul.shard(((dp,1),(1,1)))
                self.proj2[0].bias_add.shard(((dp,1),(1,1)))
                self.proj2[1].select_op.shard(((dp,1),(dp,1)))
                self.proj2[2].matmul.shard(((dp,1),(1,1)))
                self.proj2[2].bias_add.shard(((dp,1),(1,1)))
                # self.proj[3].select_op.shard(((dp,1),(dp,1)))
                # self.proj[4].matmul.shard(((dp,1),(1,1)))
                # self.proj[4].bias_add.shard(((dp,1),(1,1)))
        # operator
        self.mm=P.MatMul(transpose_b=True)
        self.norm=SRMSNorm(cfg.enc_dims)
        self.one=P.Ones()
        self.zero=P.Zeros()
        self.tile=P.Tile()
        self.gather=P.Gather()
        self.gather2=P.Gather()
        self.maskmul=P.Mul()
        self.mul=P.Mul()
        self.add=P.Add()
        self.mean=P.ReduceMean()
        self.posa=P.Add()
        self.rsqrt=P.Rsqrt()
        self.cat1=P.Concat(1)
        self.cat2=P.Concat(1)
        self.slice=P.Slice()
        self.slc=P.Slice()
        self.sum=P.ReduceSum(True)
        self.detach=P.StopGradient()
        # self.split=P.SplitV([1,2,-1],1,3)
        self.logsoftmax=P.LogSoftmax(-1)
        # loss
        self.reconstruct1=MaskedMSE(tag='_ge')
        self.reconstruct2=MaskedMSE(tag='_ce')
        self.nll_loss=ops.NLLLoss()
        # self.sim=Similarity(cfg.sim)
        # shard
        self.logger=ops.ScalarSummary()
        # st=None
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
            st=(dp,1,1)
    def set_shard(self,dp,mp):
        self.mm.shard(((dp,1),(1,1)))
        self.tile.shard(((1,1,1),))
        self.rsqrt.shard(((dp,1),))
        self.gather.shard(((mp,1),(dp,1)))
        self.expand.shard(((dp,1),))
        self.maskmul.shard(((dp,1,1),(dp,1,1)))
        self.sub.shard(((dp,1),(dp,1)))
        self.add.shard(((1,1,1),(dp,1,1)))
        self.posa.shard(((dp,1,1),(dp,1,1)))
        self.cat2.shard(((dp,1,1),(dp,1,1)))
        self.cat1.shard(((dp,1,1),(dp,1,1),(dp,1,1)))
        self.slice.shard(((dp,1,1),))
        self.slc.shard(((dp,1),))
        # self.split.shard(((dp,1,1),))
    def encode(self,expr,gene,zero_idx):
        b,l=gene.shape
        gene_emb=self.gather(self.gene_emb,gene,0)
        expr_emb,unmask=self.value_enc(expr)
        len_scale=self.detach(self.rsqrt(self.sum(zero_idx,-1)-3).reshape(b,1,1,1))
        
        expr_emb=self.posa(gene_emb,expr_emb)
        cls_token=self.tile(self.cls_token,(b,1,1))
        expr_emb=self.cat1((cls_token,expr_emb))
        mask_pos=self.cat2((self.one((b,1,1),unmask.dtype),unmask)).reshape(b,1,-1,1)
        for i in range(self.depth//2):
            expr_emb=self.encoder[i](
                expr_emb,
                v_pos=len_scale,
                attn_mask=mask_pos
            )
        mask_pos=None
        for i in range(self.depth//2,self.depth):
            expr_emb=self.encoder[i](
                expr_emb,
                v_pos=len_scale,
                attn_mask=mask_pos
            )
        return expr_emb,gene_emb
    def forward(self,expr,gene,zero_idx):
        b,l=gene.shape
        emb,gene_emb=self.encode(expr,gene,zero_idx)
        cls_token,expr_emb=emb[:,0],emb[:,1:]
        cls_token=cls_token.reshape(b,-1)
        cls_token=self.proj2(cls_token)
        return expr_emb,gene_emb,cls_token
    def construct(
        self,raw_nzdata,
        nonz_gene,zero_idx,*args
    ):
        expr_emb,gene_emb,cls_token=self.forward(
            raw_nzdata,nonz_gene,zero_idx
        )
        # gw_pred=self.value_dec(expr_emb)
        cw_pred=self.cellwise_dec(cls_token,gene_emb)
        label = raw_nzdata
        if self.training:
            loss=0
            # loss1=self.reconstruct1(gw_pred,label,mask)
            loss2=self.reconstruct2(cw_pred,label)
            # loss=loss+loss1+loss2
            loss = loss2
            return loss
        else:
            loss2=self.reconstruct2(cw_pred,label)
            return loss2, cw_pred, label



def freeze_module(module,filter_tag=[None]):
    for param in module.trainable_params():
        x=False
        for tag in filter_tag:
            if tag and tag in param.name:
                x=True
                break
        param.requires_grad = x


class SCrna():
    def __init__(self, adata,prep=False):
        self.gene_info=pd.read_csv(gene_info_path,index_col=0,header=0)
        self.geneset={j:i+1 for i,j in enumerate(self.gene_info.index)}
        gene=np.intersect1d(adata.var_names,self.gene_info.index)
        data=adata.X.astype(np.float32)
        adata.X=data

        self.adata=adata[:,gene]
        if len(self.adata)==0:
            raise Exception('samples are filtered')
        print(self.adata.shape)
        self.gene=np.array([self.geneset[i] for i in self.adata.var['gene_short_name']]).astype(np.int32)
        self.data=self.adata.X.astype(np.float32)
    def __len__(self):
        return len(self.adata)
    def __getitem__(self,idx):
        data=np.array(self.data[idx].todense()).reshape(-1)
        zero_index=np.ones_like(data)
        return data,self.gene,zero_index

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
        column_names=['data','gene','zero_idx'],
        shuffle=shuffle,
        num_shards=rank_size, 
        shard_id=rank_id
    )

    # dataset = dataset.map(
    #     prep.mask, input_columns=['data'],
    #     output_columns=['data', 'mask_gene']
    # )

    dataset=dataset.project(
        columns=[
            'data','gene','zero_idx'
        ]
    )
    dataset = dataset.batch(
        batch,
        num_parallel_workers=1, 
        drop_remainder=drop, 
    )
    return dataset


drug = sys.argv[1]
device_id = 0
adata_path = "./datasets/sciplex3.h5ad" 
check_point_path = "../../checkpoint/CellFM_80M_weight.ckpt"
gene_info_path = '../../csv/expand_gene_info.csv'

save_checkpoint_path = "./checkpoint"
checkpoint_name = f"sciplex_{drug}"


ms.set_context(
    device_target='GPU', 
    mode=ms.GRAPH_MODE,
    device_id=device_id,
)
ms.set_seed(0)

cfg=Config_80M()

adata = sc.read(adata_path)
adata = adata[adata.obs['drug'].isin([drug])]


train_adata = adata[adata.obs['split'] == 'train']
valid_adata = adata[adata.obs['split'] == 'test']
print(valid_adata.X.A.mean(0)[:10])

train_adata.var.set_index('gene_short_name', inplace=True)
valid_adata.var.set_index('gene_short_name', inplace=True)


trainset=SCrna(train_adata)
testset=SCrna(valid_adata)

prep=Prepare(trainset.data.shape[1],pad=3,mask_ratio=0.4,random=False)
train_loader=build_dataset(trainset, prep, 32, drop=True, shuffle=True, pad_zero=cfg.pad_zero)
test_loader=build_dataset(testset, prep, 1, drop=False, shuffle=False, pad_zero=cfg.pad_zero)


train=True

if train:
    para=ms.load_checkpoint(check_point_path)

    model = backbone(len(trainset.geneset),cfg)
    ms.load_param_into_net(model, para)

    optimizer=nn.Adam(model.trainable_params(),learning_rate=1e-5,weight_decay=0)
    update_cell=nn.DynamicLossScaleUpdateCell(1,2,1000)
    wrapper=Wrapper(model.to_float(ms.float16),optimizer)
    trainer=Model(
        wrapper,
        eval_network=model,
        amp_level='O0',
        metrics={
            'L2':mse_metric(798),
        },
        eval_indexes=[0,1,2]
    )

    loss_cb = LossMonitor(20)
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=len(train_loader),
        # save_checkpoint_steps=10000,
        keep_checkpoint_max=3,
        integrated_save=False,
        async_save=False
    )
    ckpt_cb = ModelCheckpoint(
        prefix=checkpoint_name, 
        directory=save_checkpoint_path, 
        config=ckpt_config
    )
    cbs=[loss_cb,ckpt_cb]

    # trainer.train(3,train_loader,callbacks=cbs)
    trainer.fit(100, train_loader, test_loader, callbacks=cbs)

    X, Y = [], []
    for data in tqdm(test_loader):
        raw_nzdata,nonz_gene,zero_idx = data
        expr_emb,gene_emb,cls_token=model.forward(
            raw_nzdata,nonz_gene,zero_idx
        )
        cw_pred=model.cellwise_dec(cls_token,gene_emb)

        X.append(raw_nzdata.asnumpy())
        Y.append(cw_pred.asnumpy())

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)


    mse = np.linalg.norm(X.mean(0) - Y.mean(0))
    pcc = np.corrcoef(Y.mean(0), X.mean(0))[0, 1]
    print(pcc, mse)



else:
    from tqdm import tqdm
    check_point_path = ''
    para=ms.load_checkpoint(check_point_path)
    model = backbone(len(trainset.geneset),cfg)
    ms.load_param_into_net(model, para)

    X, Y = [], []
    cls_emb = []
    for data in tqdm(test_loader):
        raw_nzdata,nonz_gene,zero_idx = data
        expr_emb,gene_emb,cls_token=model.forward(
            raw_nzdata,nonz_gene,zero_idx
        )
        cw_pred=model.cellwise_dec(cls_token,gene_emb)

        X.append(raw_nzdata.asnumpy())
        Y.append(cw_pred.asnumpy())

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    print(X.mean(0)[:10])
    # print(Y.mean(0))


    mse = np.linalg.norm(X.mean(0) - Y.mean(0))
    pcc = np.corrcoef(Y.mean(0), X.mean(0))[0, 1]
    print(pcc, mse)

































