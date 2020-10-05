import networkx as nx
import scipy as sp
import copy
import pickle
from scipy import sparse
from collections import defaultdict
from scipy.sparse import csr_matrix, spmatrix
import numpy as np
from sklearn.metrics import *
from scipy import spatial
from scipy.sparse import csr_matrix
import scipy.sparse
import scipy as sp
import os
import time

from reader import *
from mpi4py import MPI

import warnings
warnings.filterwarnings('ignore')

# different updateFingerprint versions ================================================
def updateFingerprint2(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''
    if countvec==1:
        count_total=min(countfp+countvec,1000);
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
    
def updateFingerprint(in_merge, fp, vec, countfp, countvec=1):
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

def updateFingerprint_just(fp, vec, countfp, countvec=1):
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


def updateFingerprint_ok(fp, vec, countfp, countvec=1):
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


def updateFingerprint3(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''
    #-thC 0.25 -thM 0.15
    count_total=countfp+countvec;
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


def updateFingerprint_old(fp, vec, countfp, countvec=1):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''

    count_total=countfp+countvec;
    propfp=countfp/count_total
        
    propvec=countvec/count_total
    
    if isinstance(vec, spmatrix):
        new_fp = fp * propfp + (vec.A.astype(np.float) * propvec)
        return(new_fp)
    else:
        new_fp = fp * propfp + (vec*propvec)
        return(new_fp)

# =================================================================================
    
mapping = {'fp':0, 
           'rank':1, 
           'id':2, 
           'size':3, 
           'fmap':4}

def get_field(fingerprints_meta, field):
    ''' returns requested field from fingerprints_meta'''
    i = mapping[field]
    return [fingerprints_meta[j][i] for j in range(len(fingerprints_meta))]

def get_idx(r, fid, ranks, ids):
    idxsr = [i for i,r_ in enumerate(ranks) if r_ == r]
    idxsi = [i for i,j in enumerate(ids) if j == fid]
    return list(set(idxsr) & set(idxsi))

def findClusters(nodes, csr_matrixg, fps, save_path, similarity='jaccard', threshold=0.05, threshold_merge=0.04, broadcast_stride = 100):
    ''' 
    input
        nodes: list of nodes
        csr_matrix: sparse adjacency matrix
        save_path: path to save log file
        similarity: similarity metric to use: dostim, jaccard, euclidean
        threshold: threshold for asigning nodes to clusters or creating new clusters
        threshold_merge: ????
        broadcast_stride: broadcast every broadcast_stride steps
        
    returns:
        fingerprints_meta: list of fingerpints and their metadata [fp, rank, id, size, fmap]
        '''
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
 
    _MIN_LINKS = 3   # minimum number of links to be added as a fingerprint
    _LOG_INTERVAL = 100 
    _CHECKPOINT_INTERVAL = 10000 
    _FLUSH_INTERVAL = 2002
    _PRINT_TEMP = False
    FLAG_MERGE = False
    OUTLIER_ACTION='remove'
    c_time=0
    cont=0
    c_den=0
    
    fingerprints_meta = []
    outliers_meta = []
    fingerprints_meta_before = []

    start_timer=time.time()
    
    # [fingerprint, rank, id, size, fmap]
    '''fingerprints meta '''
    for ri,node in enumerate(nodes):
        print('processing node', node)
       
        row = csr_matrixg[ri]
        row_array=row.A[0].astype(np.float)
        
        idxs=np.nonzero(row_array)[0]
        sum_nodes = row_array[idxs].sum()
        if sum_nodes==0: continue
        # book keeping
        cont+=1
        c_den+=np.count_nonzero(row_array)
         
        # initialize fingerprints
        if len(fingerprints_meta) == 0:
            fingerprints_meta.append([row_array, rank, 0, 1, [node]])
            print('initialze fingerprints with node', node, 'fmap', get_field(fingerprints_meta, 'fmap'))
            continue

        fps = get_field(fingerprints_meta, 'fp') # returns a list of fingerprints like our old fps list  
        array_fps = np.transpose(np.array(fps))
    
        # compute similarity ====================================================    
        start = time.time() #time the similarity metric
        
        # get best scoring fingerprint using dotSimilarity
        if similarity == 'dotsim':        
            result=row_array[idxs].dot(np.transpose(np.array(get_field(fingerprints_meta, 'fp')))[idxs,:])  / sum_nodes
            score=np.amax(result)
            fi=np.where(result == np.amax(result))[0][-1]
            # print('fps', array_fps[idxs,:])
            print('check similarity node', node, 'and fp', fi , 'score', score)

        # get best scoring fingerprint using jaccard similarity    
        elif similarity == 'jaccard':
            sums=np.sum(array_fps[idxs,:],axis=0)+sum_nodes # array_fps is transposed
            result=np.divide(row_array[idxs].dot(array_fps[idxs,:]),sums)
            score=np.amax(result)
            fi=np.where(result == np.amax(result))[0][-1]

        # get best scoring fingerprint using jaccard similarity
        elif similarity == 'euclidean':
            # 1/1+d(v1,v2) i.e. inverse of Euclidean distance = similarity score
            # this uses the whole vector, not only idxs
            result = np.divide(1, 1 + np.sqrt(np.sum(np.square(np.subtract(row,array_fps.transpose()))))) 
            score=np.amax(result)
            fi=np.where(result == np.amax(result))[0][-1]
          
        end=time.time() # end of timer for the similarity metric
        c_time+=end-start
        
        # keep track of status and perform bookkepping==========================
        if node % _LOG_INTERVAL == 0: # print and save log every 100 nodes
            _MIN_LINKS=min(3,c_den/cont)
            print("node {}/{}, size of fps {}, avg density {:.2f}, avg time: {:.6f}, time since {:.6f}  \n".format(node, len(nodes), len(fingerprints_meta), c_den/cont, c_time/cont, end-start_timer))
            with open(save_path+'log.txt', 'a+') as f:
                f.write("{},{},{:.3f},{:.6f},{:.6f}\n".format(node, get_field(fingerprints_meta, 'size'), c_den/cont, c_time/cont, end-start_timer))
            start_timer = time.time()
            c_time=0
            cont=0
            c_den=0
              
        if node % _CHECKPOINT_INTERVAL == 0:  #save progress every 10000 nodes
            FLAG_MERGE = True
            OUTLIER_ACTION='add'  # add back the outliers and merge
            if _PRINT_TEMP:
                with open(save_path+'temp_'+str(node)+'.pkl', 'wb') as f:  # python 3: open(..., 'wb')
                    pickle.dump(fingerprints_meta, f, protocol=-1)
            
        if node % _FLUSH_INTERVAL == 0:  # remove outliers and merge every 2002 steps
            FLAG_MERGE = True
            OUTLIER_ACTION='remove'
        
        
        # update fingerprint ==================================================
        if score > threshold: 

            # map node to fingerprint
            fingerprints_meta[fi][mapping['fmap']].append(node) 

            # increase cluster size by one
            fingerprints_meta[fi][mapping['size']] = fingerprints_meta[fi][mapping['size']] + 1
            
            # update fingerprint with row weights - REVISE this in other versions, it wasn;t being updated            
            # it is being updated fp[:]=new_fp performs an inplace replacement; 
            # changing the value of fp to the value new_fp, the new vector will be broadcasted and copied as needed
            fingerprints_meta[fi][mapping['fp']][:] = updateFingerprint(False, fps[fi], row_array, fingerprints_meta[fi][mapping['size']])

            print('score higher than threshold, fp', fi, 'is now size', fingerprints_meta[fi][mapping['size']], 'with nodes', fingerprints_meta[fi][mapping['fmap']], '\n')
        
        elif sum_nodes > _MIN_LINKS: # if the node has enough edges (i.e. is not an outlier)

            # create a new fp with cluster size=1, i.e. one node
            fingerprints_meta.append([row_array, rank, ri, 1, [node]])  

            # print('score lower than threshold, create new fingerprint with node', node, 'fmap:', get_field(fingerprints_meta, 'fmap'), '\n')
            print('score lower than threshold, create new fingerprint with node', node, '\n')

            '''  IT'S PROBLEMATIC to do this before the broadcast_merge, will could have redundant fps (many versions of the same fp)
            merging with local ones. I'm doing a merge after the broadcast_merge and that will happen every broadcast_stride '''
            # if FLAG_MERGE and len(fingerprints_meta) > 100: # 300 
            #     print('should call mergeSequentialFingerprints')
                                
            # #     #fps_temp, fmap_temp = mergeSequentialFingerprints(fps, fmap, OUTLIER_ACTION, save_path, similarity, threshold*.85) #.75
            # #     fps_temp, fmap_temp = mergeSequentialFingerprints(fps, fmap, OUTLIER_ACTION, save_path, similarity,threshold_merge)
            #     fingerprints_meta_temp = mergeSequentialFingerprints(fingerprints_meta, OUTLIER_ACTION, save_path, similarity, threshold_merge)            
            #     fingerprints_meta.clear()
            #     fingerprints_meta = fingerprints_meta_temp.copy()
            #     FLAG_MERGE=False
             
            # fps = get_field(fingerprints_meta, 'fp')
            # array_fps = np.transpose(np.array(fps))    
            
        else: # node has very few links to be added as a fingerprint

            # initialize outliers cluster and assign node to it
            if len(outliers_meta) == 0:
                outliers_meta.append([[], rank, 0, 1, [node]]) # note this cluster doesn't have a fingerprint it's just a place holder for outlier nodes
                print('initialze outliers cluster with node', node, 'fmap', get_field(outliers_meta, 'fmap'), '\n')
            else:
                outliers_meta[0][mapping['fmap']].append(node) 
                outliers_meta[0][mapping['size']] = outliers_meta[0][mapping['size']] + 1
                print('node has', int(sum_nodes), 'edges, not enough to be added as a fingerprint, assigned to outliers cluster')
                print('size of outliers:', outliers_meta[0][mapping['size']], '\n')
            
        #-------------- MPI ------------#
        if (ri + 1) % broadcast_stride == 0 and size > 1:
            # do merging here 
            fingerprints_meta_new = []
            for i in range(size):
                if i == rank:
                    data = {
                        'fps' : get_field(fingerprints_meta, 'fp'),
                        'size' : get_field(fingerprints_meta, 'size'),
                        'id': get_field(fingerprints_meta, 'id'),
                        'ranks': get_field(fingerprints_meta, 'rank')                
                        }
                else:
                    data = None
                data = comm.bcast(data, root = i)
               
                if i != rank:
                    for j in range(len(data['size'])):
                        fingerprints_meta_new.append([data['fps'][j], data['ranks'][j], data['id'][j], data['size'][j], []])

            # add new fingerprints_meta to fps
            fingerprints_meta += fingerprints_meta_new

            fingerprints_meta_merged = []
            ids = get_field(fingerprints_meta, 'id')
            ids_before = get_field(fingerprints_meta_before, 'id')
            for r in range(size):
                for fid in set(ids):
                    idx = get_idx(r, fid, get_field(fingerprints_meta, 'rank'), ids)
                    idx_before = get_idx(r, fid, get_field(fingerprints_meta_before, 'rank'), ids_before)
                    if len(idx) != 0 and len(idx_before) != 0:
                        fp = broadcast_merge(list(np.array(fingerprints_meta_before)[idx_before]), list(np.array(fingerprints_meta)[idx]))
                        fingerprints_meta_merged.append(fp)
                    elif len(idx_before) == 0 and len(idx) != 0:
                        fp = list(np.array(fingerprints_meta)[idx])[0]
                        fingerprints_meta_merged.append(fp)
                    elif len(idx) == 0 and len(idx_before) == 0:
                        continue

            fingerprints_meta = fingerprints_meta_merged
            fingerprints_meta_before = copy.deepcopy(fingerprints_meta)
            #-------------- MPI ------------#
            if len(fingerprints_meta) > 1: # 300 
            # if FLAG_MERGE and len(fingerprints_meta) > 1: # 300 
                print('LOCAL SIMILARITY MERGE')
                                
                fingerprints_meta_temp = mergeSequentialFingerprints(fingerprints_meta, OUTLIER_ACTION, save_path, similarity, threshold_merge)   #threshold*.85, .75         
                fingerprints_meta.clear()
                fingerprints_meta = fingerprints_meta_temp.copy()
                FLAG_MERGE=False
             
            fps = get_field(fingerprints_meta, 'fp')
            array_fps = np.transpose(np.array(fps))   

    #-------------- MPI ------------#
    fingerprints_meta_new = []
    for i in range(size):
        if i == rank:
            data = {
                'fps' : get_field(fingerprints_meta, 'fp'),
                'size' : get_field(fingerprints_meta, 'size'),
                'id': get_field(fingerprints_meta, 'id'),
                'ranks': get_field(fingerprints_meta, 'rank')                
                }
        else:
            data = None
        data = comm.bcast(data, root = i)
       
        if i != rank:
            for j in range(len(data['size'])):
                fingerprints_meta_new.append([data['fps'][j], data['ranks'][j], data['id'][j], data['size'][j], []])

    # add new fingerprints_meta to fps
    fingerprints_meta += fingerprints_meta_new

    fingerprints_meta_merged = []
    ids = get_field(fingerprints_meta, 'id')
    ids_before = get_field(fingerprints_meta_before, 'id')
    for r in range(size):
        for fid in set(ids):
            idx = get_idx(r, fid, get_field(fingerprints_meta, 'rank'), ids)
            idx_before = get_idx(r, fid, get_field(fingerprints_meta_before, 'rank'), ids_before)
            if len(idx) != 0 and len(idx_before) != 0:
                fp = broadcast_merge(list(np.array(fingerprints_meta_before)[idx_before]), list(np.array(fingerprints_meta)[idx]))
                fingerprints_meta_merged.append(fp)
            elif len(idx_before) == 0 and len(idx) != 0:
                fp = list(np.array(fingerprints_meta)[idx])[0]
                fingerprints_meta_merged.append(fp)
            elif len(idx) == 0 and len(idx_before) == 0:
                continue

    fingerprints_meta = fingerprints_meta_merged
    fingerprints_meta_before = copy.deepcopy(fingerprints_meta)
    #-------------- MPI ------------#

    return fingerprints_meta, outliers_meta # returning reconciled fingerprints and outliers # [[fp, rank, id, size, fmap], [fp, rank, id, size, fmap]...] 


# mapping = {'fp':0, 'rank':1, 'id':2, 'size':3, 'fmap':4}
def broadcast_merge(fp, fingerprints_meta):
    fp_rank = fp[0][1]
    fp_id =fp[0][2]
    fp_size = fp[0][3]
    fp_map = fp[0][4]
    fp = np.array(fp[0][0])

    sizes = get_field(fingerprints_meta, 'size')
    fps = get_field(fingerprints_meta, 'fp')
    fmaps = get_field(fingerprints_meta, 'fmap')

    size_diff = [s-fp_size for s in sizes]

    fps_weighted = [(sizes[i]*(np.array(fps[i])))-(fp*fp_size) for i in range(len(fps))] 
    fps_sum = (fp_size*fp).copy()
    for fp_tmp in fps_weighted:
        fps_sum += fp_tmp

    size = sum([size_diff[i] for i in range(len(fps))]) + fp_size

    new_fmaps = []
    for fm in fmaps:
        new_fmaps += fm
    new_fmaps = list(set(new_fmaps + fp_map))

    return [fps_sum/size, fp_rank, fp_id, size, new_fmaps]

# need to adapt this to meta_fingerprints structure 
def mergeSequentialFingerprints(fingerprints_meta_temp, outliers, save_path, similarity='dotsim', threshold=0.3):
    '''     finds similar clusters and merges them  '''

    print("MERGE FINGERPRINTS {}".format(outliers))
    _MIN_DENSITY = 2 # minimum number of elements in cluster to not be flushed

    fingerprints_meta = []
    merged_fingerprints_meta = []
   
    if outliers == 'remove' or outliers== 'add':
        try:
            with open(save_path+'outliers.pkl', 'rb') as f: 
                out_fingerprints_meta = pickle.load(f)
        except:
            out_fingerprints_meta = []
                
    nmore=0
    if outliers == 'add':      
        for i, fp in enumerate(out_fingerprints_meta):
            fingerprints_meta.append(out_fingerprints_meta[i])
            nmore+=1
        out_fingerprints_meta.clear() #to empty the file
        
        if nmore>0:
            print("# Adding back {} fingerprints".format(nmore))
            os.remove(save_path+'outliers.pkl')
            
    # sort fingerprints from less to more density
    idxFP = np.argsort([fp[mapping['size']] for fp in fingerprints_meta_temp])
 
    # sort fingerprints
    for i in idxFP:
        fingerprints_meta.append(fingerprints_meta_temp[i])
    
    fps = get_field(fingerprints_meta, 'fp')
    row_array=np.array(fps)
    
    idxs_all=row_array.any(axis=0) #0
    array_fps=np.transpose(row_array)
    
    processed = []
    
    threshold_temp=threshold
    #threshold=threshold_temp/10000.0
    max_len=np.max([ fingerprint_meta[mapping['size']] for fingerprint_meta in fingerprints_meta]) # size of the biggest cluster 
    second_half=int(len(fps)//2)

    for ai, afp in enumerate(fps):
        
        if ai<second_half:
            threshold=threshold_temp*(fingerprints_meta[ai][mapping['size']]/max_len)
        else:
            threshold=threshold_temp
        
        # skip fingerprints that have already been merged
        if ai in processed: continue

        # dot similarity
        if similarity == 'dotsim':
            idxs=np.nonzero(row_array[ai,:])[0]
            sum_sonzeros=row_array[ai,idxs].sum()
            result=row_array[ai,idxs].dot(array_fps[idxs,:])  / sum_sonzeros

        elif similarity == 'jaccard':
            sum_sonzeros=row_array[ai,idxs_all].sum()
            sums=np.sum(array_fps[idxs_all,:],axis=0)+sum_sonzeros # array_fps is transposed
            result=np.divide(row_array[ai,idxs_all].dot(array_fps[idxs_all,:]),sums)

        elif similarity == 'euclidean':
            # 1/1+d(v1,v2) ie inverse of Euclidean distance = similarity score
            # using all the vector, not only idxs
            result = np.divide(1, 1 + np.sqrt(np.sum(np.square(np.subtract(row_array[ai,idxs_all],array_fps[idxs_all,:].transpose()))))) 
            score=np.amax(result)
            fi=np.where(result == np.amax(result))[0][-1]

        result[ai]=0.0 #make 0 it's own product
        bi=ai+1
        if bi<len(result):
            score=result[ai+1]
        else:
            continue        

        while (bi in processed or score<threshold) and bi<len(result)-1:
            bi+=1
            score=result[bi]
            
        if score<threshold or bi in processed:
            continue     
        else:        
            # merge fingerprints in proportion of densities
            fingerprints_meta[bi][mapping['fp']][:] = updateFingerprint(True, afp, fps[bi], fingerprints_meta[ai][mapping['size']], fingerprints_meta[bi][mapping['size']])
            fingerprints_meta[bi][mapping['fmap']] = list(set(fingerprints_meta[bi][mapping['fmap']] + fingerprints_meta[ai][mapping['fmap']]))

            print('ids',fingerprints_meta[ai][mapping['id']],fingerprints_meta[bi][mapping['id']])
            print('ranks',fingerprints_meta[ai][mapping['rank']],fingerprints_meta[bi][mapping['rank']])
            # avoid inconsistencies across ranks by keeping min id and rank of merged fingerprints
            min_fp_id = min(fingerprints_meta[ai][mapping['id']],fingerprints_meta[bi][mapping['id']])
            min_fp_rank = min(fingerprints_meta[ai][mapping['rank']],fingerprints_meta[bi][mapping['rank']])
            fingerprints_meta[bi][mapping['id']] = min_fp_id
            fingerprints_meta[bi][mapping['rank']] = min_fp_rank
            print('kept id and rank', fingerprints_meta[bi][mapping['id']], fingerprints_meta[bi][mapping['rank']])

            # mark as processed
            processed += [ai]
            
    print("# Fingerprints merged: {}".format(len(processed)))

    fps = get_field(fingerprints_meta, 'fp')
    row_array=np.array(fps)

    # add fingerprints that were not merged
    allnodes=0
    for i, fp_meta in enumerate(fingerprints_meta):
        
        if outliers=='remove' and fp_meta[mapping['size']] <= _MIN_DENSITY: is_outlier = True 
        else: is_outlier = False
        
        if i not in processed and not is_outlier:
            merged_fingerprints_meta.append(fp_meta)
            allnodes+=fingerprints_meta[i][mapping['size']]
            
        elif i not in processed and is_outlier:
            out_fingerprints_meta.append(fp_meta)
            
    if outliers == 'remove' and  len(fingerprints_meta) > 0:
        with open(save_path+'outliers.pkl', 'wb') as f:  
            pickle.dump(out_fingerprints_meta, f, protocol=-1)
            print("# Fingerprints removed: {}".format(get_field(out_fingerprints_meta, 'size')))

    #print("Num nodes {}".format(sum([len(merged_fmap[listNodes]) for listNodes in merged_fmap])))
            
    print("# Nodes in fmap: {}\n".format(allnodes))
    return merged_fingerprints_meta


def mergeFingerprints(fingerprints_meta_temp, outliers, save_path, similarity='dotsim', threshold=0.3):
    '''
    finds similar clusters and merges them  
    '''
    print("MERGE FINGERPRINTS {}".format(outliers))
    _MIN_DENSITY = 2 # minimum number of elements in cluster to not be flushed
    fingerprints_meta = []
    merged_fingerprints_meta = []

    if outliers == 'remove' or outliers== 'add':
        try:
            with open(save_path+'outliers.pkl', 'rb') as f: 
                out_fingerprints_meta = pickle.load(f)
        except:
            out_fingerprints_meta = []
                
    nmore=0
    if outliers == 'add':      
        for i, fp in enumerate(out_fingerprints_meta):
            fingerprints_meta.append(out_fingerprints_meta[i])
            nmore+=1
        out_fingerprints_meta.clear() #to empty the file
        
        if nmore>0:
            print("# Adding back {} fingerprints".format(nmore))
            os.remove(save_path+'outliers.pkl')
                
              
    # sort fingerprints from less to more density        
    idxFP = np.argsort([fp[mapping['size']] for fp in fingerprints_meta_temp])

    #idxFP = np.argsort([ len(fmap_temp[listNodes]) for listNodes in fmap_temp])
    # list_lens=np.sort([ len(fmap_temp[listNodes]) for listNodes in fmap_temp])

    # sort fingerprints 
    for i in idxFP:
        fingerprints_meta.append(fingerprints_meta_temp[i])

    fps = get_field(fingerprints_meta, 'fp')
    row_array=np.array(fps)
        
    idxs_all=row_array.any(axis=0) #0
    array_fps=np.transpose(row_array)

    processed = []
    for ai, afp in enumerate(fps):
        
        # skip fingerprints that have already been merged
        if ai in processed: continue

        # dot similarity
        if similarity == 'dotsim':
            idxs=np.nonzero(row_array[ai,:])[0]
            sum_sonzeros=row_array[ai,idxs].sum()
            result=row_array[ai,idxs].dot(array_fps[idxs,:])  / sum_sonzeros
              
        elif similarity == 'jaccard':
            sum_sonzeros=row_array[ai,idxs_all].sum()
            sums=np.sum(array_fps[idxs_all,:],axis=0)+sum_sonzeros # array_fps is transposed
            result=np.divide(row_array[ai,idxs_all].dot(array_fps[idxs_all,:]),sums)
        
        elif similarity == 'euclidean':
            # 1/1+d(v1,v2) ie inverse of Euclidean distance = similarity score
            # using all the vector, not only idxs
            result = np.divide(1, 1 + np.sqrt(np.sum(np.square(np.subtract(row_array[ai,idxs_all],array_fps[idxs_all,:].transpose()))))) 
            score=np.amax(result)
            fi=np.where(result == np.amax(result))[0][-1]

        # normalized mutual info score         

        # ===========================
        result[ai]=0.0 #make 0 it's own product
        score=np.amax(result)
        bi=np.where(result == score)[0][0]
        sum_result=np.sum(result)
            
        while (bi in processed or score<threshold) and sum_result>0:
            result[bi]=0.0
            sum_result=np.sum(result)
            score=np.amax(result)
            bi=np.where(result == np.amax(result))[0][0]
                        
        if score<threshold or bi in processed:
            continue     
        else:
            #print("score {}, theshold {}".format(score,threshold))

            # merge fingerprints in proportion of densities
            fingerprints_meta[bi][mapping['fp']][:] = updateFingerprint(True, afp, fps[bi], fingerprints_meta[ai][mapping['size']], fingerprints_meta[bi][mapping['size']])
            fingerprints_meta[bi][mapping['fmap']] = list(set(fingerprints_meta[bi][mapping['fmap']] + fingerprints_meta[ai][mapping['fmap']]))

            ''' avoid inconsistencies across ranks by keeping min id and rank of merged fingerprints
            - no need for this here because it's the final merge (no more broadcasts) '''
            # min_fp_id = min(fingerprints_meta[ai][mapping['id']],fingerprints_meta[bi][mapping['id']])
            # min_fp_rank = min(fingerprints_meta[ai][mapping['rank']],fingerprints_meta[bi][mapping['rank']])
            # fingerprints_meta[bi][mapping['id']] = min_fp_id
            # fingerprints_meta[bi][mapping['rank']] = min_fp_rank

            # mark as processed
            processed += [ai]
            
    print("# Fingerprints merged: {}".format(len(processed))) 
    # add fingerprints that were not merged
    allnodes=0
    for i, fp_meta in enumerate(fingerprints_meta):
        
        if outliers=='remove' and fp_meta[mapping['size']] <= _MIN_DENSITY: is_outlier = True 
        else: is_outlier = False
        
        if i not in processed and not is_outlier:
            merged_fingerprints_meta.append(fp_meta)
            allnodes+=fingerprints_meta[i][mapping['size']]
            
        elif i not in processed and is_outlier:
              out_fingerprints_meta.append(fp_meta)
                
    if outliers == 'remove' and len(fingerprints_meta) >0:
        with open(save_path+'outliers.pkl', 'wb') as f:  
            pickle.dump(out_fingerprints_meta, f, protocol=-1)
            print("# Fingerprints removed: {}".format(get_field(out_fingerprints_meta, 'size')))

    #print("Num nodes {}".format(sum([len(merged_fmap[listNodes]) for listNodes in merged_fmap])))
            
    print("# Nodes in fmap: {}\n".format(allnodes))
    return merged_fingerprints_meta


'''  
step n -> broadcast
fps -> [0 0 1 2..] 0 0 2 from rank 0

step n + 1
step n + 2

....

step n + K -> broadcast
	fps        creation rank - id
fps -> [0 1 1 1 ...] 0 0 4 from rank 0
fps -> [0 1 0 1 ...] 0 0 3 from rank 1
fps -> [0 1 1 0 ...] 0 0 5 from rank 2

----------------------------------
rank 0

n1 [0 0 1 0]
n2 [0 0 1 1]

fp -> [0 0 1 0] 1
fp -> [0 0 1 .5] 2

bradcast

[0 0 1 1] 
[0 0 1 1]

fp -> [0 0 1 .6666] 3
fp -> [0 0 1 .75] 4


rank 1

fp -> [0 0 1 .5] 2 

n1 [0 0 1 0]

fp -> [0 0 1 .333] 3

broadcast

rank 0

previous version -> [ 0 0 1 .5] 2 -> [0 0 2 1]

[0 0 1 .75] 4 -> [0 0 4 3] - [0 0 2 1] = [0 0 2 2]
[0 0 1 .3333] 3 -> [0 0 3 1] - [0 0 2 1] = [0 0 1 0]

[0 0 2 1] + [0 0 2 2] + [0 0 1 0]  = [0 0 5 3]/5 = [0 0 1 3/5]

rank 0 
[0 0 1 1]

fps [0 0 1 1] 0 2

rank 1
[0 0 1 1]
fps [0 0 1 .5] -1 0

every rank will do the same merge
merge -> 
fps_new -> [0 0 1 .75] -> -1 0 
'''


# def mergeFingerprints_old(fps, fmap, similarity='dotsim', threshold=0.3):
#     '''
#     finds similar clusters and merges them  
#     '''
#     # same kind of mapping of nodes to fingerprints
#     merged_fps = []
#     merged_fmap = {}

#     processed = []
#     for ai, afp in enumerate(fps):
#         # skip fingerprints that have already been merged
#         if ai in processed: continue
            
#         # dot similarity
#         if similarity == 'dotsim':
#             score, bi, bfp = sorted([(dotSimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps) if bi != ai]).pop()

#         # normalized mutual info score
#         elif similarity == 'nmi':
#             score, bi, bfp = sorted([(NMISimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps) if bi != ai]).pop()
     
#         # cosine Similarity
#         elif similarity == 'cosine':
#             score, bi, bfp = sorted([(cosineSimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps) if bi != ai]).pop() 
                      
#         # same for second fingerprint
#         if bi in processed: continue

#         if score > threshold:
#             # merge fingerprints
#             fp = updateFingerprint(afp, bfp, len(fmap[ai]), len(fmap[bi]))
#             merged_fps.append(fp)
#             # merge node references
#             i = len(merged_fps) - 1
#             merged_fmap[i] = list(set(fmap[ai] + fmap[bi]))
#             # mark as processed
#             processed += [ai, bi]
            
#     # add fingerprints that were not merged
#     for i, fp in enumerate(fps):
#         if i not in processed:
#             merged_fps.append(fp)
#             merged_fmap[len(merged_fps) - 1] = fmap[i]

#     return merged_fps, merged_fmap


# def findProbabilisticClusters(nodes, csr_matrix, fps =[], similarity='dotsim', threshold=0.5):
#     ''' 
#     input
#         nodes: list of nodes
#         csr_matrix: sparse adjacency matrix
#         threshold: initial merging threshold
        
#     returns:
#         fps:  list of all found fingerprints
#         fmap: (dict) fingerprint mapping of nodes to fingerprints 
#               to keep track of what node belongs to what fp
#         {
#             fp_index: [
#                 row_index,
#                 ...
#             ],
#             ...
#         }
#         '''
#     fmap = defaultdict(list)

#     ''' fingerprints '''
#     #fps = []

#     for node in nodes:
#         ri=node
#         row = csr_matrix[ri]
        
#         # initialize fingerprints
#         if len(fps) == 0:
#             fmap[len(fps)].append(node)
#             fps.append(row.A[0].astype(np.float))
#             continue
        
#         # get best scoring fingerprint using dotSimilarity
#         if similarity == 'dotsim':        
#             # sorted and pop gets me the best scoring one (find something more elegant!)
#             similarfps = sorted([(dotSimilarity(fp, row), fi, fp) for fi, fp in enumerate(fps)])

#         # get best scoring fingerprint using cosine Similarity
#         elif similarity == 'cosine':
#             similarfps = sorted([(cosineSimilarity(fp, row), fi, fp) for fi, fp in enumerate(fps)])
        
#         # get best scoring fingerprint using mutual_info_score
#         elif similarity == 'nmi':
#             similarfps = sorted([(NMISimilarity(fp, row), fi, fp) for fi, fp in enumerate(fps)])  

            
#         score, fi, fp = similarfps.pop() # pop the last one
#         total_score=score
#         not_assigned=True  # use this to assign a node only to one cluster
            
#         if score < threshold:  # if less threshold, add as a new fingerprint
#             fmap[len(fps)].append(node)
#             fps.append(row.A[0].astype(np.float))
#         else:
#             for k in reversed(range(len(similarfps))):
#                 if similarfps[k][0] >= threshold: 
#                     total_score+=similarfps[k][0]
#                 else:
#                     break
        
#         while score > threshold:  # otherwise, pop as long as score > threshold and add to other fingerprints proportionally
            
#             proportion=score/total_score
#             # map node to fingerprint
#             if not_assigned: # use this flag to assign a node only to one cluster, remove for multiples
#                 fmap[fi].append(node)
#                 not_assigned=False
                
#             # update fingerprint with row weights
#             fp[:] = updateFingerprint(fp, row*proportion, len(fmap[fi]))
            
#             score, fi, fp = similarfps.pop()
            
            
#     return fps, fmap
