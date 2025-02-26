import os
import scipy as sp
import numpy as np
import pandas as pd
import mindspore as ms
from sklearn.metrics.pairwise import rbf_kernel
from mindspore import nn,ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.communication import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context
class BinaryACC(ms.train.Metric):
    def __init__(self,threshold=0.5):
        super().__init__()
        self.threshold=threshold
        self.clear()
    def clear(self):
        self.tp=0
        self.ttl=0
    def update(self, pred, label):
        pred=pred.asnumpy().reshape(-1)
        label=label.asnumpy().reshape(-1)
        pred=(pred>self.threshold).astype(label.dtype)
        self.tp+=(pred*label).sum()
        self.ttl+=len(pred)
    def eval(self):
        return self.tp/self.ttl
class annote_metric(ms.train.Metric):
    def __init__(self,num_class,key=None):
        super().__init__()
        self.num_class=num_class
        self.eye=np.eye(num_class)
        self.key=key or 'accuracy'
        self.clear()
    def clear(self):
        self.ttl=0
        self.conf_mat=np.zeros((self.num_class,self.num_class))
    def update(self, pred, label):
        pred=pred.asnumpy().argmax(-1).reshape(-1)
        label=label.asnumpy().reshape(-1)
        self.ttl+=len(label)
        for i,j in zip(pred,label):
            self.conf_mat[i,j]+=1
    def eval(self):
        tp=self.conf_mat[self.eye.astype(np.bool_)]
        num=self.conf_mat.sum(0)
        err=self.conf_mat*(1-self.eye)
        fp=err.sum(1)
        fn=err.sum(0)
        acc=tp.sum()/self.ttl
        recall=np.nanmean(tp/np.maximum(tp+fn,0.1))
        precision=np.nanmean(tp/np.maximum(tp+fp,0.1))
        f1=2*tp/(2*tp+fp+fn)
        m_f1=np.nanmean(f1)
        w_f1=np.nansum(f1*(num/self.ttl))
        res={
            'accuracy':acc,'macro f1':m_f1,'weighted f1':w_f1,
            'macro recall':recall,'macro precision':precision,
        }
        return res[self.key]
class F1(ms.train.Metric):
    def __init__(self,num_class,mode='macro'):
        super().__init__()
        assert mode in ['macro','weighted']
        self.mode=mode
        self.num_class=num_class
        self.eye=np.eye(num_class)
        self.clear()
    def clear(self):
        self.tp=np.zeros(self.num_class)
        self.pred_pos=np.zeros(self.num_class)
        self.gt_pos=np.zeros(self.num_class)
        
    def update(self, pred, label):
        pred=pred.asnumpy()
        label=label.asnumpy().reshape(-1)
        pred=pred.argmax(-1).reshape(-1)
        label=self.eye[label]
        pred=self.eye[pred]
        self.tp+=(label*pred).sum(0)
        self.pred_pos+=pred.sum(0)
        self.gt_pos+=label.sum(0)
    def eval(self):
        f1=2*self.tp/(self.pred_pos+self.gt_pos)
        if self.mode=='macro':
            return np.nanmean(f1)
        elif self.mode=='weighted':
            f1=f1*(self.gt_pos/self.gt_pos.sum())
            return np.nansum(f1)

class perturb_metric(ms.train.Metric):
    def __init__(self,ctrl,de_idx,pert_map,key=None):
        super().__init__()
        self.ctrl=ctrl.mean(0)
        self.de_idx=de_idx
        self.pert_map=pert_map
        self.key=key or 'de_PCC2'
        self.clear()
    def clear(self):
        self.x={}
        self.y={}
        self.z={}
    def update(self,source,pred,target,pert_id):
        source=source.asnumpy()
        pred=pred.asnumpy()
        target=target.asnumpy()
        pert_id=pert_id.asnumpy()
        for i in range(len(pert_id)):
            u,v=pert_id[i]
            pert=self.pert_map[(u,v)]
            self.x[pert]=self.x.get(pert,[])+[pred[i]]
            self.y[pert]=self.y.get(pert,[])+[target[i]]
            self.z[pert]=self.z.get(pert,[])+[source[i]]
    def eval(self):
        res={}
        mse=[]
        de_mse=[]
        pcc1=[]
        pcc2=[]
        pcc3=[]
        de_pcc1=[]
        de_pcc2=[]
        de_pcc3=[]
        r2=[]
        de_r2=[]
        for i in self.x:
            if i=='ctrl':
                continue
            pred=np.stack(self.x[i],0).mean(0)
            target=np.stack(self.y[i],0).mean(0)
            source=np.stack(self.z[i],0).mean(0)
            de_idx=self.de_idx[i]
            de_pred=pred[de_idx]
            de_target=target[de_idx]
            de_source=source[de_idx]
            de_ctrl=self.ctrl[de_idx]
            L2=(pred-target)**2
            mse.append(L2.mean())
            de_mse.append(L2[de_idx].mean())
            pcc1.append(sp.stats.pearsonr(pred,target)[0])
            pcc2.append(sp.stats.pearsonr(pred-source,target-source)[0])
            pcc3.append(sp.stats.pearsonr(pred-self.ctrl,target-self.ctrl)[0])
            de_pcc1.append(sp.stats.pearsonr(de_pred,de_target)[0])
            de_pcc2.append(sp.stats.pearsonr(de_pred-de_source,de_target-de_source)[0])
            de_pcc3.append(sp.stats.pearsonr(de_pred-de_ctrl,de_target-de_ctrl)[0])
            r2.append(1-L2.sum()/((target-target.mean())**2).sum())
            de_r2.append(1-L2[de_idx].sum()/((de_target-de_target.mean())**2).sum())
        res['MSE']=np.array(mse).mean()
        res['PCC1']=np.array(pcc1).mean()
        res['PCC2']=np.array(pcc2).mean()
        res['PCC3']=np.array(pcc3).mean()
        res['R2']=np.array(r2).mean()
        res['de_MSE']=np.array(de_mse).mean()
        res['de_PCC1']=np.array(de_pcc1).mean()
        res['de_PCC2']=np.array(de_pcc2).mean()
        res['de_PCC3']=np.array(de_pcc3).mean()
        res['de_R2']=np.array(de_r2).mean()
        return res[self.key]
    
class eval_batch(ms.train.Metric):
    def __init__(self):
        super().__init__()
        self.loss = 0
        self.clear()
    def clear(self):
        self.loss = 0
    def update(self, mse):
        self.loss += mse
    def eval(self):
        print('val_loss: ',self.loss)
        return self.loss

class mse_metric(ms.train.Metric):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.clear()
    def clear(self):
        self.X = np.zeros(self.seq_len)
        self.Y = np.zeros(self.seq_len)
        self.l = 0
    def update(self, pred, label):
        pred=pred.asnumpy().reshape(-1)
        label=label.asnumpy().reshape(-1)
        self.X += pred
        self.Y += label
        self.l += 1
    def eval(self):
        self.X = self.X / self.l
        self.Y = self.Y / self.l
        l2 = np.linalg.norm(self.X - self.Y)
        print('L2:', l2)
        return l2