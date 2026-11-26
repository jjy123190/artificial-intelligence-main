"""Parameter-free local reconstruction head on frozen ResNet12 spatial maps.

Equation-level FRN-inspired ridge reconstruction, not a trained FRN reproduction.
No external model, query-label input, across-query adaptation or query balance
prior. Raw descriptors preserve local information discarded by global pooling.
"""
from __future__ import annotations
import math
import numpy as np


def validate_inputs(support,labels,query):
 s=np.asarray(support,dtype=np.float64);q=np.asarray(query,dtype=np.float64);y=np.asarray(labels)
 if s.ndim!=3 or q.ndim!=3 or min(s.shape)<1 or min(q.shape)<1 or s.shape[2]!=q.shape[2]:raise ValueError('Expected image x local position x channel tensors')
 if y.shape!=(len(s),) or not np.issubdtype(y.dtype,np.integer) or not np.isfinite(s).all() or not np.isfinite(q).all():raise ValueError('Invalid descriptor labels/features')
 classes=np.unique(y)
 if len(classes)<2 or not np.array_equal(classes,np.arange(len(classes))):raise ValueError('At least two contiguous support classes required')
 return s,y,q,len(classes)


def reconstruct_predict(support,support_labels,query,*,ridge=1.,normalization='channel_scale'):
 s,y,q,ways=validate_inputs(support,support_labels,query)
 if type(ridge) not in (int,float) or not math.isfinite(ridge) or not 0<ridge<=100:raise ValueError('Invalid ridge')
 if normalization=='channel_scale':s=s/np.sqrt(s.shape[2]);q=q/np.sqrt(q.shape[2])
 elif normalization=='local_l2':
  s=s/np.maximum(np.linalg.norm(s,axis=2,keepdims=True),1e-12);q=q/np.maximum(np.linalg.norm(q,axis=2,keepdims=True),1e-12)
 else:raise ValueError('Unknown local normalization')
 flat=q.reshape(-1,q.shape[2]);scores=[]
 for cls in range(ways):
  descriptors=s[y==cls].reshape(-1,s.shape[2]);n,d=descriptors.shape;regularization=(n/d)*ridge+1e-6
  if n<d:
   gram=descriptors@descriptors.T
   projection=descriptors.T@np.linalg.solve(gram+regularization*np.eye(n),descriptors)
  else:
   gram=descriptors.T@descriptors
   projection=np.linalg.solve(gram+regularization*np.eye(d),gram)
  residual=flat@projection-flat
  scores.append(-np.mean(np.sum(residual.reshape(q.shape)**2,axis=2),axis=1))
 logits=np.stack(scores,axis=1)
 return {'predictions':logits.argmax(1),'scores':logits,'transductive':False,'balanced_queries':False,'probability_calibration_claimed':False}
