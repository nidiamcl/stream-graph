from reader import *
from mpi4py import MPI
from find_probabilistic_clusters import *
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
edge_list_path = '../network_data/'

# node_edges_path: path to file number of edges per node (needed for reader to read in all edges on a node)
# 1,364        i.e., node 1 has 364 edges (there will be 364 adjacent lines to read in for this node)
# 2,254
# 3,202
# 4,221
# 5,103
node_edges_path = '../network_data/'

# clusters_path: where resulting clusters will be saved
clusters_path = '../network_data/'

# log will be in same directory as clustering
log_path = clusters_path 

sim1='dotsim'
sim2='dotsim'

fps = []
outliers_action = 'remove'


# actual algorithm
gr = GraphReader(edge_list_path + network + '.tsv', node_edges_path + network + '_node_edges.txt')
csr_matrix = gr.read() # csr sparse matrix from the reader
nodes = gr.local_vertices

# find initial clusters, findClusters returns fps (list) and fmap (dict)
fps, fmap, fmap_oultiers = findProbabilisticClusters(network, nodes, csr_matrixg, fps, log_path, similarity=sim1, threshold=t1)

''' ------------------- START MPI DATA TRANSFER ------------------------'''
local_num_fps = len(fps)
num_fps_per_rank = comm.gather(local_num_fps, root=0)

if rank == 0:
    for r, num_fps in enumerate(num_fps_per_rank):
        if r != 0:
            for _ in range(num_fps_per_rank[r]):
                f = np.empty(len(fps[0]), dtype=np.float64)
                comm.Recv(f, source=r, tag=0)
                fps.append(f)
                members = comm.recv(source=r, tag=1)
                fmap[len(fps)-1] = members
else:
    for i, f in enumerate(fps):
        comm.Send(f, dest=0, tag=0)
        comm.send(fmap[i], dest=0, tag=1)
''' ------------------- END MPI DATA TRANSFER ------------------------'''

if rank == 0:
    # merge similar clusters    
    merged_fps, merged_fmap = mergeProbabilisticFingerprints(fps, fmap, outliers_action, log_path, similarity=sim2, threshold=t2)
    pruned_fmap = getNonOverlappingClusters(merged_fmap)

    # print(network)
    # print('Calculating similarity: using ' + sim1 + ' and ' + sim2)
    # print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
    # print('remaining clusters: ' + str(len(merged_fmap)))

    end = MPI.Wtime()
    print(network, timedelta(seconds=end-start))

    pickle.dump(merged_fps, open(clusters_path + '{}_fps.pkl'.format(network), 'wb'))
    pickle.dump(merged_fmap, open(clusters_path + '{}_fmap.pkl'.format(network), 'wb'))


# how to run:
# mpiexec -n 2 python run_clustering_mpi.py --n 'zebra' --t1 0.3 --t2 0.6