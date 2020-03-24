import numpy as numpy
import networkx as nx
import scipy as sp
import pickle
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csr_matrix, spmatrix
import numpy as np

def getSimilarity(fp, vec):
    ''' gets similarity between a fingerprint and a row vector
        the number of non-zero components they share 
        divided by the total number of non-zero components of the vector '''
    return vec.dot(fp).max() / vec.sum()

def updateFingerprint(fp, vec, count):
    ''' updates a fingerprint when a node vector is added to the cluster
        weighted merge of the node vector with the fingerprint '''
    if isinstance(vec, spmatrix):
        return (fp * ((count-1)/count)) + (vec.A.astype(np.float) * (1/count))
    else:
        return (fp * ((count-1)/count)) + (vec*(1/count))

def findClusters(g, threshold=0.3):
    ''' fingerprint map
    {
        fp_index: [
            row_index,
            ...
        ],
        ...
    }
    '''
    # mapping of nodes to fingerprints to keep track of what node belongs to what fp
    fmap = defaultdict(list)
    
    # I could (should?) be creating this matrix from edge tuples directly
    matrix = nx.to_numpy_matrix(g)

    ''' fingerprints '''
    fps = []

    for ri, node in enumerate(g.nodes):
        row = matrix[ri]
        
        # initialize fingerprints
        if len(fps) == 0:
            fmap[len(fps)].append(node)
            fps.append(row.A[0].astype(np.float))
            continue
        
        # get best scoring fingerprint
        # sorted and pop gets me the best scoring one (I should find something more elegant)
        score, fi, fp = sorted([(getSimilarity(fp, row), fi, fp) for fi, fp in enumerate(fps)]).pop() 
        
        if score > threshold:
            # map node to fingerprint
            fmap[fi].append(node)
            # update fingerprint with row weights
            fp[:] = updateFingerprint(fp, row, len(fmap[fi]))
            
        else:
            fmap[len(fps)].append(node)
            fps.append(row.A[0].astype(np.float))
            
    return fps, fmap

def mergeFingerprints(fps, fmap, threshold=0.3):
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

        # get best scoring fingerprint
        score, bi, bfp = sorted([(getSimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps) if bi != ai]).pop()

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

# --------------------------------------------------------------------------------------------------------------
# -------------------ALGORITHM ENDS HERE, THE REST IS JUST LAST MINUTE PLOTTING --------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

def drawClusters(g, fmap, color, title, fname):
    nodes = list(g.nodes)
    
    order = []
    for ci, cnodes in sorted(fmap.items()):
        for cnode in cnodes:
            order.append(nodes.index(cnode))
            
    matrix = nx.to_numpy_array(g)[order][:,order]
    x, y = np.argwhere(matrix == 1).T
    
#     fig = go.Figure(data=go.Scatter(
#         x=x,
#         y=y,
#         mode='markers'
#     ))
    
#     fig.show()

     
    plt.scatter(x, y, s=0.1, c=color)

    # Add title and axis names
    plt.title(title, fontsize=25)
    plt.xlabel('Nodes', fontsize=18)
    plt.ylabel('Nodes', fontsize=18)

    plt.show()

    out_path = '../../stream_graph_data/plots/'
    plt.savefig(out_path + fname + '.png')    
    return matrix, x, y
                
# fps, fmap = findClusters(G)

def showClusters(in_path):
    ''' plots clusters for all nx network objects stored in pkl files'''
    
    all_files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
    
    for f in all_files: 
        fn = f.split('-')
#         print(fn)
        fname = fn[1] + '-' + fn[2].split(' ')[0]
#         print(fname)
        print('network: ' + fname)

        graph = nx.read_gpickle(in_path + f)
        fps, fmap = findClusters(graph, 0.3)
        print('clusters found: ' + str(len(fmap)))

        merged_fps, merged_fmap = mergeFingerprints(fps, fmap, 0.05)
        print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
        print('remaining clusters: ' + str(len(merged_fmap)))
        
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (20, 20)
              
        flegend_dict = {str(k): len(v) for k, v in fmap.items()}
        name1 = 'Graph clusters before merging fingerprints: ' + fname
        m, x, y = drawClusters(graph, fmap, 'rosybrown', name1, fname)

        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (20, 20)

        mlegend_dict = {k: len(v) for k, v in merged_fmap.items()}
        name2 = 'Graph clusters after merging fingerprints: ' + fname
        m, x, y = drawClusters(graph, merged_fmap, 'black', name2 , fname+'_merged')
