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

edge_list_path = '../../../Code/stream_graph_data/all_networks/'
node_edges_path = '../../../Code/stream_graph_data/node_edges_all_networks/'
clusters_path = '../../../Code/stream_graph_data/clustered_networks/mpi_clusters/best_scores_newman/4_process/'

sim1='dotsim'
sim2='nmi'

# actual algorithm
gr = GraphReader(edge_list_path + network + '.tsv', node_edges_path + network + '_node_edges.txt')
csr_matrix = gr.read() # csr sparse matrix from the reader
nodes = gr.local_vertices

# find initial clusters, findClusters returns fps (list) and fmap (dict)
fps, fmap = findClusters(nodes, csr_matrix, similarity=sim1, threshold=t1)

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
    merged_fps, merged_fmap = mergeFingerprints(fps, fmap, similarity=sim2, threshold=t2)

    # print(network)
    # print('Calculating similarity: using ' + sim1 + ' and ' + sim2)
    # print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
    # print('remaining clusters: ' + str(len(merged_fmap)))

    end = MPI.Wtime()
    print(network, timedelta(seconds=end-start))

    pickle.dump(merged_fps, open(clusters_path + '{}_fps.pkl'.format(network), 'wb'))
    pickle.dump(merged_fmap, open(clusters_path + '{}_fmap.pkl'.format(network), 'wb'))


# mpiexec -n 4 python run_clustering_mpi.py
