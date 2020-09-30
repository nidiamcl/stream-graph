from reader import *
from mpi4py import MPI
from find_clusters import *
import pickle as pkl
from timeit import default_timer as timer
from datetime import timedelta
import argparse
# from th import *

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', required=True)                   # network
parser.add_argument('--t1', required=True, type=float)      # first threshold
parser.add_argument('--t2', required=True, type=float)      # second threshold
args = parser.parse_args()

network = args.n
t1 = args.t1
t2 = args.t2

# mpi stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start = MPI.Wtime()

# edge_list_path: path to file with the list of edges e.g., 
# 1 2       i.e., there is an edge between node 1 and 2
# 1 3
# 2 3
# 2 5
# ....
edge_list_path = '../sample_data/'

# node_edges_path: path to file number of edges per node (needed for reader to read in all edges on a node)
# 1,364        i.e., node 1 has 364 edges (there will be 364 adjacent lines to read in for this node)
# 2,254
# 3,202
# 4,221
# 5,103
node_edges_path = '../sample_data/'

# clusters_path: where resulting clusters will be saves
clusters_path = '../sample_data/'

sim1='dotsim'
sim2='nmi'

mapping = {'fp':0, 'rank':1, 'id':2, 'size':3, 'fmap':4}
def get_field(fingerprints, field):
    i = mapping[field]
    return [fingerprints[j][i] for j in range(len(fingerprints))]

def get_idx(r, fid, ranks, ids):
    idxsr = [i for i,r_ in enumerate(ranks) if r_ == r]
    idxsi = [i for i,j in enumerate(ids) if j == fid]
    return list(set(idxsr) & set(idxsi))

# actual algorithm
gr = GraphReader(edge_list_path + network + '.tsv', node_edges_path + network + '_node_edges.txt')
csr_matrix = gr.read() # csr sparse matrix from the reader
nodes = gr.local_vertices

# find initial clusters, findClusters returns fps (list) and fmap (dict)
fingerprints_meta = findClusters(nodes, csr_matrix, similarity=sim1, threshold=t1)

'''
[[fps, rank, id, size, map],[fps, rank, id, size, map],...]
fps : [.02, .43, ...]
rank : rank where it came frome
id : id in rank
size : how many nodes are in this fingerprint
map : [232, 34, 12...]

f1 : rank 0 id 0 currently in rank 0 [0, 1]
f1 : rank 0 id 0 currently in rank 1 [2]

f1 [0, 1, 2]
'''

''' ------------------- START MPI DATA TRANSFER ------------------------'''
all_fingerprints = comm.gather(fingerprints_meta, root = 0)
if rank == 0:
    fingerprints_meta = []
    for fp in all_fingerprints:
        fingerprints_meta = fingerprints_meta + fp

    fingerprints_meta_merged = []
    for r in range(size):
        for i in set(get_field(fingerprints_meta, 'id')):
            idx = get_idx(r, i, get_field(fingerprints_meta, 'rank'), get_field(fingerprints_meta, 'id'))
            if len(idx) > 0:
                metas = list(np.array(fingerprints_meta)[idx])
                maps = []
                for tmp in metas:
                    maps += tmp[4]
               
                fp = metas[0][0]
                rank = metas[0][1]
                idx = metas[0][2]
                size = metas[0][3]

                fp_meta = [fp, rank, idx, size, maps]
                fingerprints_meta_merged.append(fp_meta)

    # fingerprints has a list of fingerprint meta data, including: 
    # fingerprint, the actual representative vector
    # rank, which rank this fingerprint started in
    # id, fingerprint id number. it is local to the rank
    # size, number of nodes in the fingerprint
    # nodes list, nodes that belong to the fingerprint
    
    fingerprints = get_field(fingerprints_meta_merged, 'fp') # fingerprints
    fmap = get_field(fingerprints_meta_merged, 'fmap') # which nodes belong to the fingerprint

    # TODO merge
    '''
    # merge similar clusters
    merged_fps, merged_fmap = mergeFingerprints(fps, fmap, similarity=sim2, threshold=t2)

    # print(network)
    # print('Calculating similarity: using ' + sim1 + ' and ' + sim2)
    # print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
    # print('remaining clusters: ' + str(len(merged_fmap)))

    end = MPI.Wtime()
    print(network, timedelta(seconds=end-start))

    pickle.dump(merged_fps, open(clusters_path + '{}_fps.pkl'.format(network), 'wb'))
    pickle.dump(merged_fmap, open(clusters_path + '{}_fmap.pkl'.format(network), 'wb'))
    '''

