from find_clusters import *
from cdlib.algorithms import louvain, label_propagation
import scipy 
import pandas as pd
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import timedelta
import argparse

from reader import *
from mpi4py import MPI

# -------------------------------------------------------------------------

def buildGraph(edge_list_path, network):
    '''
    Create nx graph from edge list
    '''
    with open(edge_list_path + network + '.tsv') as f:
        edges = f.readlines()
        edges = [tuple(line.strip().split(' ')) for line in edges][2:]
        edges = [(int(x[0]), int(x[1])) for x in edges]

    graph = nx.Graph()
    graph.name = network

    graph.add_edges_from(edges)
    print(nx.info(graph))
    print("Network density:", nx.density(graph))  
    return graph

def clusterGraph(g, edge_list_path, node_edges_path, network, sim1, sim2, t1, t2):

    network_name = network
    nodes = g.nodes()

    gr = GraphReader(edge_list_path + network + '.tsv', node_edges_path + network + '_node_edges.txt')
    csr_matrix = gr.read() # csr sparse matrix from the reader
       
    initial_threshold = t1
    merging_threshold = t2
    
    print('Calculating similarity: using ' + sim1 + ' and ' + sim2)    

    # find clusters
    fps, fmap = findClusters(nodes, csr_matrix, similarity='nmi', threshold=initial_threshold)

    # merge similar clusters
    merged_fps, merged_fmap = mergeFingerprints(fps, fmap, similarity='nmi', threshold=merging_threshold)
    
    partition = dict()

    for cluster, nodes in merged_fmap.items():
        for node in nodes:
            partition[node] = cluster
        
    return partition, merged_fmap, merged_fps

def runAlgorithm(edge_list_path, node_edges_path, network, sim1, sim2, t1, t2):
    graph = buildGraph(edge_list_path, network)
    partition, merged_fmap, merged_fps = clusterGraph(graph, edge_list_path, node_edges_path, network, sim1, sim2, t1, t2)
    return partition, merged_fmap, merged_fps


# --------------------------------------------------------------------------

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', required=True)                   # network
parser.add_argument('--t1', required=True, type=float)      # first threshold
parser.add_argument('--t2', required=True, type=float)      # second threshold
args = parser.parse_args()

network = args.n
t1 = args.t1
t2 = args.t2

edge_list_path = '../../../Code/stream_graph_data/all_networks/'
node_edges_path = '../../../Code/stream_graph_data/node_edges_all_networks/'
thresholds_file = '../../../Code/stream_graph_data/newman_mod_thresholds_for_best_scores.txt'
clusters_path = '../../../Code/stream_graph_data/clustered_networks/good_clusters/best_scores_newman'
# clusters_path = '../../../Code/stream_graph_data/clustered_networks/good_clusters/best_scores_newman_2/'

sim1='dotsim'
sim2='nmi'

# with open(thresholds_file, 'r') as f:
#     lines = f.readlines()
#     lines = [line.strip() for line in lines]
#     networks = [line.split(',') for line in lines]
#     networks = [ [n[0],n[2],n[3]] for n in networks]

# for net in networks:
#     network = net[0]
#     t1 =  float(net[1])
#     t2 =  float(net[2])
    
start = timer()
partition, merged_fmap, merged_fps = runAlgorithm(edge_list_path, node_edges_path, network, sim1=sim1 , sim2=sim2, t1=t1, t2=t2)
end = timer()
print(timedelta(seconds=end-start))
print('')

with open(clusters_path + network + '_fmap'+ '.pkl', 'wb') as f:
    pickle.dump(merged_fmap, f)
    
with open(clusters_path + network + '_fps'+ '.pkl', 'wb') as f:
    pickle.dump(merged_fps, f)
    
with open(clusters_path + network + '_partition'+ '.pkl', 'wb') as f:
    pickle.dump(partition, f)