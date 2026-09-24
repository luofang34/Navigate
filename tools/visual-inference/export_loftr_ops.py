"""Portable operations for the fixed 640 by 480 LoFTR-DS export.

Adapted from LoFTR (Copyright SenseTime) and Kornia (Copyright 2018 Kornia Team).
Apache-2.0. MatMul replaces Einsum. Fixed-shape tensor operations replace
control flow in coarse selection and fine-window extraction. Weights do not change.
"""
import torch
import torch.nn.functional as F

def attention(self,queries,keys,values,q_mask=None,kv_mask=None):
 q,k=F.elu(queries)+1,F.elu(keys)+1
 if q_mask is not None:q=q*q_mask[:,:,None,None]
 if kv_mask is not None:k=k*kv_mask[:,:,None,None];values=values*kv_mask[:,:,None,None]
 length=values.shape[1]
 kv=k.permute(0,2,3,1)@(values/length).permute(0,2,1,3)
 z=1/((q*k.sum(1)[:,None]).sum(3)+self.eps)
 return ((q.permute(0,2,1,3)@kv).permute(0,2,1,3)*z[:,:,:,None]*length).contiguous()

def coarse(self,a,b,data,mask_c0=None,mask_c1=None):
 sim=(a/16)@(b/16).transpose(1,2)/self.temperature
 conf=torch.softmax(sim,1)*torch.softmax(sim,2)
 v0,j=conf.max(2);v1,i=conf.max(1)
 rows=torch.arange(a.shape[1],device=a.device)[None]
 valid=(i.gather(1,j)==rows)&(v0>self.thr)
 width,height=80,60
 for ids in [rows,j]:valid=valid&(ids%width>=2)&(ids%width<width-2)&(ids//width>=2)&(ids//width<height-2)
 idx=torch.where(valid[0])[0];jd=j[0,idx]
 data.update({'b_ids':torch.zeros_like(idx),'i_ids':idx,'j_ids':jd,'mconf':v0[0,idx],
  'mkpts0_c':torch.stack((idx%width,idx//width),dim=1).float()*8,
  'mkpts1_c':torch.stack((jd%width,jd//width),dim=1).float()*8})

def fine(self,a,b,data):
 sim=(a[:,12:13,:]@b.transpose(1,2)).squeeze(1)/128**.5
 heat=torch.softmax(sim,dim=1)
 grid=torch.stack(torch.meshgrid(torch.linspace(-1,1,5),torch.linspace(-1,1,5),indexing='ij'),dim=2).reshape(25,2).flip(1).to(a)
 data['mkpts0_f']=data['mkpts0_c'];data['mkpts1_f']=data['mkpts1_c']+(heat@grid)*4

def fine_preprocess(self,f0,f1,c0,c1,data):
 data['W']=5;i,j=data['i_ids'],data['j_ids']
 a=F.unfold(f0,kernel_size=5,stride=4,padding=2).reshape(1,128,25,-1).permute(0,3,2,1)[0,i]
 b=F.unfold(f1,kernel_size=5,stride=4,padding=2).reshape(1,128,25,-1).permute(0,3,2,1)[0,j]
 a=self.merge_feat(torch.cat((a,self.down_proj(c0[0,i]).unsqueeze(1).expand(-1,25,-1)),dim=2))
 b=self.merge_feat(torch.cat((b,self.down_proj(c1[0,j]).unsqueeze(1).expand(-1,25,-1)),dim=2))
 return a,b

