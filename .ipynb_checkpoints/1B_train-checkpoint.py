import os
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
from mindspore.amp import FixedLossScaleManager,all_finite,DynamicLossScaleManager
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.communication import init, get_rank, get_group_size
from config import Config
from utils import Wrapper,WrapperWithLossScaleCell,load_dist_model
from utils import WarmCosineDecay,Adam,AdamWeightDecay,set_weight_decay
from model import CellFM
from data_process import Prepare,SC100M,build_dataset
# os.environ['MINDSPORE_DUMP_CONFIG']='/share-nfs/w50035851/code/msver/dump.json'
def freeze_module(module,filter_tag=[None]):
    for param in module.trainable_params():
        x=False
        for tag in filter_tag:
            if tag and tag in param.name:
                x=True
                break
        param.requires_grad = x
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist',action='store_true')
    parser.add_argument('--long',action='store_true')
    parser.add_argument('--npu',type=int,default=0)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--data',type=str,default='normal')
    parser.add_argument('--data0',type=int,default=0)
    parser.add_argument('--data1',type=int,default=100)
    parser.add_argument('--datapath',type=str, default='/share-nfs/w50035851/datasets/sc100M/')
    parser.add_argument('--savepath',type=str, default='/share-nfs/w50035851/code/msver/checkpoint/sc100M_32_e2')
    args = parser.parse_args()
    ms.set_context(
        device_target='Ascend', 
        mode=ms.GRAPH_MODE,
        device_id=args.npu,
        # enable_compile_cache=True,
    )
    cfg=Config()
    rank_id = None
    rank_size = None
    if args.long:
        cfg.nonz_len=4096
        cfg.filt_len=2049
    if os.getenv('MS_ROLE')!='MS_SCHED':
        sc100m=SC100M(
            args.data0,args.data1,False,data=args.data,
            filt_len=(cfg.filt_len,cfg.nonz_len),
            root=args.datapath,shuffle=55
        )
        with open(f'/share-nfs/w50035851/code/msver/log/fin{os.getenv("MS_NODE_ID")}.txt','w+') as f:
            f.write(f'{os.getenv("MS_NODE_ID")} fin\n')
    while args.dist:
        time.sleep(5)
        fins=glob.glob('/share-nfs/w50035851/code/msver/log/fin*.txt')
        if len(fins)==int(os.getenv('MS_WORKER_NUM')):
            break
    if args.dist:
        ms.set_auto_parallel_context(
            # parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, 
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, 
            # enable_parallel_optimizer=True,
            parameter_broadcast=True,
            gradients_mean=True,
            # enable_alltoall=True,
            comm_fusion={"allreduce": {"mode": "auto", "config": None}},
        )
        ms.set_seed(0)
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
    cfg.enc_dims=1536
    cfg.enc_nlayers=40
    cfg.enc_num_heads=48
    # cfg.lora=64
    cfg.add_zero=False
    cfg.pad_zero=True
    cut=None #if cfg.pad_zero else int(0.75*cfg.nonz_len)
    prep=Prepare(
        cfg.nonz_len,pad=3,
        mask_ratio=0.2,dw=True,zero_len=None,
        cut=cut,random=False
    )
    dataset=build_dataset(
        sc100m,
        prep,
        args.batch,
        drop=True,
        pad_zero=cfg.pad_zero,
        rank_size=rank_size,
        rank_id=rank_id,
    )
    with open('/share-nfs/w50035851/code/msver/ckpt.txt','r') as f:
        latest=f.readlines()[0].strip()
        if 'classify' in latest:
            latest=latest[:-5]+'_combined.ckpt'
    tag=f"cancer_{args.data0}_{args.data1}"
    model=CellFM(len(sc100m.geneset),cfg)
    if cfg.lora>0:
        cfg.recompute=False
        freeze_module(model,['lora'])
    params=set_weight_decay(model.trainable_params())
    optimizer=nn.Adam(params,1e-7,eps=1e-8, beta1=0.9, beta2=0.95)
    # optimizer=nn.Adam(model.trainable_params(),1e-7,weight_decay=1e-7)
    update_cell=nn.DynamicLossScaleUpdateCell(1,2,1000)
    wrapper=WrapperWithLossScaleCell(model.to_float(ms.float16),optimizer,update_cell)
    print(f'load from {latest}')
    para=ms.load_checkpoint(latest)
    # for i in list(para.keys()):
    #     # if 'value_enc.mask' in i and 'value_enc.mask_emb' not in i:
    #     #     j=i.replace('mask','mask_emb')
    #     #     para[j]=para.pop(i)
    ms.load_param_into_net(wrapper, para)
    trainer=Model(
        wrapper,
        amp_level='O0',
    )
    loss_cb = LossMonitor(20)
    summary_cb = ms.SummaryCollector(
        summary_dir=f"/share-nfs/w50035851/analyse/{tag}", 
        collect_specified_data={"collect_metric": True},
        collect_freq=1, 
        keep_default_action=False, 
    )
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=5000,
        keep_checkpoint_max=2,
        integrated_save=False,
        async_save=False
    )
    ckpt_cb = ModelCheckpoint(
        prefix=tag, 
        directory=f"{args.savepath}/{rank_id or 0}/", 
        config=ckpt_config
    )
    cbs=[loss_cb]
    if rank_id==0 or rank_id is None:
        cbs.append(summary_cb)
        cbs.append(ckpt_cb)
    now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Begin training {len(dataset)} steps at {now}')
    trainer.train(args.epoch,dataset,callbacks=cbs)
    if rank_id==0 or rank_id is None:
        path=ckpt_cb.latest_ckpt_file_name
        with open('/share-nfs/w50035851/code/msver/ckpt.txt','w') as f:
            f.write(path)