from reader import *
from mpi4py import MPI
from find_clusters import *
import pickle as pkl


# mpi stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# choose thresholds
first_th = 0.3
second_th = 0.8

data_path = '../../../Code/stream_graph_data/test_networks/'
network = 'facebook_caltech'

# gr = GraphReader('../sample_data/test.tsv', '../sample_data/test_node_edges.txt')
gr = GraphReader(data_path + network + '.tsv', data_path + network + '_node_edges.txt')
csr_matrix = gr.read() # csr sparse matrix from the reader
nodes = gr.local_vertices

# find initial clusters, findClusters returns fps (list) and fmap (dict)
fps, fmap = findClusters(nodes, csr_matrix, similarity='nmi', threshold=first_th)

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
    merged_fps, merged_fmap = mergeFingerprints(fps, fmap, similarity='nmi', threshold=second_th)
    print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
    print('remaining clusters: ' + str(len(merged_fmap)))

    pickle.dump(merged_fps, open('merged_fps_{}_{}.pkl'.format(first_th,second_th), 'wb')) 
    pickle.dump(merged_fmap, open('merged_fmap_{}_{}.pkl'.format(first_th,second_th), 'wb')) 

# to run this script
# mpiexec -n 4 python mpi_usage_example.py 
