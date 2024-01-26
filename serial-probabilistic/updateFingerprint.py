from scipy.sparse import csr_matrix, spmatrix
import networkx as nx
import scipy as sp
from scipy import sparse
from collections import defaultdict
from scipy.sparse import csr_matrix, spmatrix
import numpy as np
from sklearn.metrics import *
from scipy import spatial
from scipy.sparse import csr_matrix
import scipy.sparse
import scipy as sp
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


### old original function 
def updateFingerprint0(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''

    count_total=countfp+countvec
#     print(count_total)
    
    propfp=countfp/count_total
#     print(propfp)
    
    propvec=countvec/count_total
#     print(propfp)

    if isinstance(vec, spmatrix):
        new_fp = fp * propfp + (vec.A.astype(np.float) * propvec)
        return(new_fp)
    else:
        new_fp = fp * propfp + (vec*propvec)
        return(new_fp)

# the one we're using currently
def updateFingerprint1(in_merge, fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''

    count_total=np.sum(countfp)+np.sum(countvec)
    if in_merge:
        propfp=np.sum(countfp)/count_total
        propvec=np.sum(countvec)/count_total
    else:
        propfp=0.999
        propvec=0.1
    
    if isinstance(vec, spmatrix):
        new_fp = fp * propfp + (vec.A.astype(np.float) * propvec)
    else:
        new_fp = fp * propfp + (vec*propvec)
        
    new_fp=np.clip(new_fp, 0, 1.0) 
    
    
    return(new_fp)


def updateFingerprint2(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''
    if countvec==1:
        count_total=min(countfp+countvec,1000)
        propfp=min(countfp,999)/count_total
    else:
        count_total=countfp+countvec;
        propfp=countfp/count_total
        
    propvec=countvec/count_total
    
    if isinstance(vec, spmatrix):
        new_fp = fp * propfp + (vec.A.astype(np.float) * propvec)
    else:
        new_fp = fp * propfp + (vec*propvec)
        
    # clip the values so that the fp does not vanish    
    idc=np.nonzero(new_fp>=0.1)
    new_fp[idc]=np.clip(new_fp[idc], 0.6, 1.0) 
    
    idc=np.nonzero(new_fp)
    new_fp[idc]=np.clip(new_fp[idc], 0.05, 1.0)
        
    return(new_fp)

def updateFingerprint3(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''
#-thC 0.25 -thM 0.15
    count_total=countfp+countvec
    print(count_total)
    
    propfp=max(0.99,countfp/count_total)
    propvec=max(0.4,countvec/count_total)
    
    if isinstance(vec, spmatrix):
        new_fp = fp * propfp + (vec.A.astype(np.float) * propvec)
    else:
        new_fp = fp * propfp + (vec*propvec)
        
    # clip the values so that the fp does not vanish    
    idc=np.nonzero(new_fp>=0.45) #1
    new_fp[idc]=np.clip(new_fp[idc], 0.6, 1.0) 
    idc=np.nonzero(new_fp)
    new_fp[idc]=np.clip(new_fp[idc], 0.4, 1.0)
        
    return(new_fp)

# just
def updateFingerprint4(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''

    count_total=countfp+countvec;
    propfp=max(0.99,countfp/count_total)
    propvec=max(0.01,countvec/count_total)
    
    if isinstance(vec, spmatrix):
        new_fp = fp * propfp + (vec.A.astype(np.float) * propvec)
    else:
        new_fp = fp * propfp + (vec*propvec)
        
        
    idc=np.nonzero(new_fp>=0.3) #1
    new_fp[idc]=np.clip(new_fp[idc], 0.7, 1.0)
    
    idc=np.nonzero(new_fp)
    new_fp[idc]=np.clip(new_fp[idc], 0.1, 1.0)
    
    #new_fp=np.clip(new_fp, 0, 1.0) 
    
    return(new_fp)

# ok
def updateFingerprint5(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''

    count_total=countfp+countvec;
    propfp=max(0.99,countfp/count_total)
    propvec=max(0.01,countvec/count_total)
    
    if isinstance(vec, spmatrix):
        new_fp = fp * propfp + (vec.A.astype(np.float) * propvec)
    else:
        new_fp = fp * propfp + (vec*propvec)
        
    # clip the values so that the fp does not vanish    
    idc=np.nonzero(new_fp>=0.5) #1
    new_fp[idc]=np.clip(new_fp[idc], 0.6, 1.0)
    
    idc=np.nonzero(new_fp)
    new_fp[idc]=np.clip(new_fp[idc], 0.01, 1.0)
        
    return(new_fp)


##### similarity functions
    
def dotSimilarity(fp, vec):
    ''' gets similarity between a fingerprint and a row vector
        the number of non-zero components they share 
        divided by the total number of non-zero components of the vector '''
    return vec.dot(fp).max() / vec.sum()

def jaccardSimilarity(fp, vec):
    ''' gets similarity between a fingerprint and a row vector
        the number of non-zero components they share 
        divided by the total number of non-zero components of the vector '''
    return vec.dot(fp).max() / np.add(vec,fp).sum()

def cosineSimilarity(fp,vec):
    '''' gets similarity between a fingerprint and a row vector'''
    if scipy.sparse.isspmatrix_csr(vec):
        return 1 - spatial.distance.cosine(fp, vec.A.astype(np.float))
    else:
        return 1 - spatial.distance.cosine(fp, vec)

def NMISimilarity(fp,vec):
    '''' gets similarity between a fingerprint and a row vector using NMI'''
    if scipy.sparse.isspmatrix_csr(vec):       
        return normalized_mutual_info_score(fp,vec.A.astype(np.float).flatten())
    else:
        return normalized_mutual_info_score(fp, vec)