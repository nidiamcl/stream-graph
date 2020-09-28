import networkx as nx
import scipy as sp
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
    counts = defaultdict(lambda : 0)

    # will probably use these but not yet
    ranks = defaultdict(lambda : 0)
    indices = defaultdict(lambda : 0)

    ''' fingerprints '''
    fps = []

    for ri, node in enumerate(nodes):
        row = csr_matrix[ri]
        
        # initialize fingerprints
        if len(fps) == 0:
            fmap[len(fps)].append(node)
            counts[len(fps)] = 1
            fps.append(row.A[0].astype(np.float))
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
#             print(score, threshold)
            
            # map node to fingerprint
            fmap[fi].append(node)
            counts[fi] = counts[fi] + 1
            # update fingerprint with row weights
            fp[:] = updateFingerprint(fp, row, len(fmap[fi]))
        else:
            fmap[len(fps)].append(node)
            counts[len(fps)] = 1
            fps.append(row.A[0].astype(np.float))

        #-------------- MPI ------------#
        if (ri + 1) % broadcast_stride == 0:
            new_counts = []
            new_fps = []
            for i in range(size):
                if i == rank:
                    counts_list = [counts[i] for i in range(len(counts))]
                    data = {
                        'fps' : fps,
                        'count' : counts_list
                        }
                else:
                    data = None
                    
                data = comm.bcast(data, root = i)
                
                if i != rank:
                    new_counts = new_counts + data['count']
                    new_fps = new_fps + data['fps']

        # add new fingerprints to fps
        # and update the count
        current_fps_size = len(fps)
        for i,f in enumerate(new_fps):
            fps.append(f)
            counts[i+current_fps_size] = new_counts[i] 


        #TODO decide how to merge otherwise this will blow up
        #-------------- MPI ------------#

    return fps, fmap

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

    
