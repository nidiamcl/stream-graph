import networkx as nx
import scipy as sp
import copy
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import sparse
from collections import defaultdict
from scipy.sparse import csr_matrix, spmatrix
import numpy as np
from sklearn.metrics import *
from scipy import spatial
from scipy.sparse import csr_matrix
import scipy.sparse
import scipy as sp
import tqdm as tqdm
from reader import *
from mpi4py import MPI

import warnings
warnings.filterwarnings('ignore')

def dotSimilarity(fp, vec):
    ''' gets similarity between a fingerprint and a row vector
        the number of non-zero components they share 
        divided by the total number of non-zero components of the vector '''
    return vec.dot(fp).max() / vec.sum()

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

def updateFingerprint(fp, vec, count):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''
    
    # for sparse matrices in csr format
    if scipy.sparse.isspmatrix_csr(vec):
        return (fp * ((count-1)/count)) + (vec.A.astype(np.float) * (1/count))
    
    # for numpy.matrix
    else:
        return (fp * ((count-1)/count)) + (vec*(1/count))
    
    
# broadcast_stride : broadcast every broadcast_stride steps
def findClusters(nodes, csr_matrix, similarity='dotsim', threshold=0.5, broadcast_stride = 10):
    ''' 
    input
        nodes: list of nodes
        csr_matrix: sparse adjacency matrix
        threshold: initial merging threshold
        
    returns:
        fps:  list of all found fingerprints
        fmap: (dict) fingerprint mapping of nodes to fingerprints 
              to keep track of what node belongs to what fp
        {
            fp_index: [
                row_index,
                ...
            ],
            ...
        }
        '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fmap = defaultdict(list)
    
    fps_before = []
    ranks_before = []
    id_before = []
    sizes_before = []
    fmaps_before = []

    fps = []
    ranks = []
    id_ = []
    sizes = []
    fmaps = []

    fingerprints = []
    fingerprints_before = []

    # [fingerprint, rank, id, size, fmap]
    ''' fingerprints '''
    for ri, node in enumerate(nodes):
        row = csr_matrix[ri]
    
        # initialize fingerprints
        if len(fingerprints) == 0:
            fingerprints.append([row.A[0].astype(np.float), rank, 0, 1, [node]])
            continue
        
        # get best scoring fingerprint using dotSimilarity
        if similarity == 'dotsim':        
            # sorted and pop gets me the best scoring one (find something more elegant!)
            score, fi, fp = sorted([(dotSimilarity(fp, row), fi, fp) for fi, fp in enumerate(fps)]).pop() 

        # get best scoring fingerprint using cosine Similarity
        elif similarity == 'cosine':
            score, fi, fp = sorted([(cosineSimilarity(fp, row), fi, fp) for fi, fp in enumerate(fps)]).pop() 
        
        # get best scoring fingerprint using mutual_info_score
        elif similarity == 'nmi':
            score, fi, fp = sorted([(NMISimilarity(fp, row), fi, fp) for fi, fp in enumerate(fps)]).pop()  

        if score > threshold:
            # map node to fingerprint
            fmaps[fi].append(node)
            sizes[fi] = sizes[fi] + 1
            # update fingerprint with row weights
            fp[:] = updateFingerprint(fp, row, len(fmaps[fi]))
        else:
            #fmap[len(fps)].append(node)
            fmaps.append([node])
            sizes.append(1)
            id_.append(ri)
            ranks.append(rank)
            fps.append(row.A[0].astype(np.float))

        #-------------- MPI ------------#
        if (ri + 1) % broadcast_stride == 0:
            new_fps = []
            new_ranks = []
            new_id = []
            new_sizes = []
            for i in range(size):
                if i == rank:
                    size_list = [sizes[i] for i in range(len(sizes))]
                    data = {
                        'fps' : fps,
                        'size' : size_list,
                        'id': id_,
                        'ranks': ranks                
                        }
                else:
                    data = None
                    
                data = comm.bcast(data, root = i)
                
                if i != rank:
                    new_sizes = new_sizes + data['size']
                    new_fps = new_fps + data['fps']
                    new_id = new_id + data['id']
                    new_ranks = new_ranks + data['ranks']

            # add new fingerprints to fps
            # and update the count
            for i in range(len(new_fps)):
                fps.append(new_fps[i])
                ranks.append(new_ranks[i])
                id_.append(new_id[i])
                sizes.append(new_sizes[i])
            fmaps = fmaps + [[]]*len(new_fps)

            def get_idx(r, fid, ranks, ids):
                idxsr = [i for i,r_ in enumerate(ranks) if r_ == r]
                idxsi = [i for i,j in enumerate(ids) if j == fid]
                return list(set(idxsr) & set(idxsi))
            
            fps_ = []
            ranks_ = []
            id__ = []
            sizes_ = []
            fmaps_ = []
            for r in range(size):
                for fid in set(id_):
                    idx = get_idx(r, fid, ranks, id_)
                    idx_before = get_idx(r, fid, ranks_before, id_before)
                    if len(idx_before) != 0:
                        new_f, new_s, new_fm = broadcast_merge(list(np.array(fps_before)[idx_before]), list(np.array(sizes_before)[idx_before]), list(np.array(fmaps_before)[idx_before]),
                                                                list(np.array(fps)[idx]), list(np.array(sizes)[idx]), list(np.array(fmaps)[idx]))

                        fps_.append(new_f)
                        ranks_.append(r)
                        id__.append(fid)
                        sizes_.append(new_s)
                        fmaps_.append(new_fm)

            fps = fps_
            ranks = ranks_
            id_ = id__
            sizes = sizes_
            fmaps = fmaps_

            fps_before = copy.deepcopy(fps)
            ranks_before = copy.deepcopy(ranks)
            id_before = copy.deepcopy(id_)
            sizes_before = copy.deepcopy(sizes)
            fmaps_before = copy.deepcopy(fmaps)

        #-------------- MPI ------------#

    print(rank, np.shape(fps), np.shape(fmaps), len(fmap))
    return fps, fmaps

def broadcast_merge(fp, fp_size, fp_fmap, fps, sizes, fmaps):
    if len(fp) == 0:
        return fps[0], sizes[0], fmaps[0]
    else:
        fp = np.array(fp)

        size_diff = [s-fp_size for s in sizes]

        fps_weighted = [sizes[i]*(np.array(fps[i]))-fp*fp_size for i in range(len(fps))]
        fps_sum = fp.copy()
        for fp_tmp in fps_weighted:
            fps_sum += fp_tmp

        size = sum([sizes[i]-fp_size for i in range(len(fps))]) + fp_size

        new_fmaps = []
        for fm in fmaps:
            new_fmaps += fm
        print(new_fmaps, fp_fmap)
        new_fmaps = list(set(new_fmaps + fp_fmap[0]))
        
        return fps_sum/size, size, new_fmaps


def mergeFingerprints(fps, fmap, similarity='dotsim', threshold=0.3):
    '''
    finds similar clusters and merges them  
    '''
    # same kind of mapping of nodes to fingerprints
    merged_fps = []
    merged_fmap = {}

    processed = []
    for ai, afp in enumerate(fps):
        # skip fingerprints that have already been merged
        if ai in processed: continue
            
        # dot similarity
        if similarity == 'dotsim':
            score, bi, bfp = sorted([(dotSimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps) if bi != ai]).pop()

        # normalized mutual info score
        elif similarity == 'nmi':
            score, bi, bfp = sorted([(NMISimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps) if bi != ai]).pop()
     
        # cosine Similarity
        elif similarity == 'cosine':
            score, bi, bfp = sorted([(cosineSimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps) if bi != ai]).pop() 
                      
        # same for second fingerprint
        if bi in processed: continue

        if score > threshold:
            # merge fingerprints
            fp = updateFingerprint(afp, bfp, 2)
            merged_fps.append(fp)
            # merge node references
            i = len(merged_fps) - 1
            merged_fmap[i] = list(set(fmap[ai] + fmap[bi]))
            # mark as processed
            processed += [ai, bi]
            
    # add fingerprints that were not merged
    for i, fp in enumerate(fps):
        if i not in processed:
            merged_fps.append(fp)
            merged_fmap[len(merged_fps) - 1] = fmap[i]

    return merged_fps, merged_fmap


if __name__ == "__main__":
    
#     trajectory = np.load('../sample_data/trajectory.npy')
#     t = np.matrix(trajectory)
#     csr_trajectory = sparse.csr_matrix(trajectory)
        
    file = '../sample_data/test.tsv'
    g = nx.read_edgelist(file, delimiter=' ', nodetype=int, edgetype=int, create_using=nx.Graph())
    nodes = g.nodes()
    print(nodes)
    
#     print(nx.info(g))
#     print('Density: ' + str(nx.density(g)))

    gr = GraphReader('../sample_data/test.tsv', '../sample_data/test_node_edges.txt')
    csr_matrix = gr.read() # csr sparse matrix from the reader
    print(np.array(csr_matrix.todense()))
    
    csr_networkx = nx.to_scipy_sparse_matrix(g, format='csr', dtype=np.float64) # csr sparse matrix from networkx
    print(np.array(csr_networkx.todense()))
        
    # choose thresholds
    first_th = 0.3
    second_th = 0.7
    
    # could be 'dotsim', 'cosine', 'nmi'
    similarity1 = 'dotsim' 
    similarity2 = 'nmi'
                
    print('Initial similarity metric: ' + similarity1)  
    print('Merging similarity metric: ' + similarity2)  
        
    # find initial clusters, findClusters returns fps (list) and fmap (dict)
    fps, fmap = findClusters(nodes, csr_matrix, similarity='nmi', threshold=first_th)
    print('clusters found: ' + str(len(fmap)))

    # merge similar clusters
    merged_fps, merged_fmap = mergeFingerprints(fps, fmap, similarity='nmi', threshold=second_th)
    print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
    print('remaining clusters: ' + str(len(merged_fmap)))

#     pickle.dump(merged_fps, open('merged_fps_{}_{}.pkl'.format(first_th,second_th), 'wb')) 
#     pickle.dump(merged_fmap, open('merged_fmap_{}_{}.pkl'.format(first_th,second_th), 'wb')) 
    
#     ------------------------------------------------------------
#     the csr matrix from the reader is not the same as one created from the networkx library
#     you can check it out with the comparison below

#     print(csr_matrix.todense() == csr_A.todense())

    
