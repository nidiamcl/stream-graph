from find_clusters import *
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from networkx.algorithms import community 

import cdlib
from cdlib.algorithms import louvain
from cdlib import evaluation

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
   
def thresholdSearch(graph, csr_matrix, network_name, output_path, sim1='dotsim', sim2='nmi', initial_start=0, initial_stop=1, initial_num=10, merging_start=0, merging_stop=1, merging_num=10):
    ''' calculates fitness score for combinations of thresholds values
    
        g:              nx graph  
        csr_matrix      sparse adjacency matrix
        network_name    network name
        sim1:           similarity metric between nodes and fps (could be 'dotsim', 'cosine', 'nmi')         
        sim2:           merging similarity metric between fps (could be 'dotsim', 'cosine', 'nmi')
        initial_start:  starting value of the initial threshold sequence
        initial_stop:   end value of the initial threshold sequence
        initial_num:    number of evenly spaced threshold values to generate
        merging_start:  the starting value of the merging threshold sequence
        merging_stop:   the end value of the merging threshold sequence
        merging_num:    number of evenly spaced threshold values to generate
    
        returns a scores (dict) with scores and info about clusters for every 
        combination of initial and merging threshold values
        
         {0: {'initial_theshold': 0.25,
              'merging_threshold': 0.5,
              'score': 0.28831587944489795,
              'clusters_found': 98,
              'clusters_merged': 0,
              'remaining_clusters': 98},
          1: {...}
              ...
              ... } 
    '''

    graph.name = network_name.split('.')[0]
    nodes = graph.nodes()

    initial_thresholds = list(np.linspace(initial_start,initial_stop,initial_num, endpoint=True))[1:] 
    merging_thresholds = list(np.linspace(merging_start, merging_stop, merging_num, endpoint=True))[1:]
    combined_thresholds = [[i,m] for i in initial_thresholds for m in merging_thresholds]

    method = 'nidia'
    scores = defaultdict(dict)
    
    for i, pair in enumerate(tqdm(combined_thresholds)):
        initial_threshold = pair[0]
        merging_threshold = pair[1]

        # find initial clusters, findClusters returns fps (list) and fmap (dict)
        fps, fmap = findClusters(nodes, csr_matrix, similarity='nmi', threshold=initial_threshold)

        # merge similar clusters
        merged_fps, merged_fmap = mergeFingerprints(fps, fmap, similarity='nmi', threshold=merging_threshold)

        # need to create a NodeClustering object
        # clusters:     list of communities (list of elements)
        # graph:        networkx/igraph object
        # method_name: community discovery algorithm name

        clusters = list(merged_fmap.values())
        c = cdlib.classes.node_clustering.NodeClustering(clusters, graph, method)

        # modularity fitness functions
        newman_modularity = evaluation.newman_girvan_modularity(graph,c).score
        erdos_renyi_mod = evaluation.erdos_renyi_modularity(graph,c).score
        link_modularity = evaluation.link_modularity(graph,c).score

        # other fitness_functions
        average_internal_degree = evaluation.average_internal_degree(graph,c).score
        conductance = evaluation.conductance(graph,c).score
        cut_ratio = evaluation.cut_ratio(graph,c).score
        edges_inside = evaluation.edges_inside(graph,c).score
        internal_edge_density = evaluation.internal_edge_density(graph,c).score        

        scores[i]['initial_threshold'] = initial_threshold
        scores[i]['merging_threshold'] = merging_threshold
        scores[i]['clusters_found'] = len(fmap)
        scores[i]['clusters_merged'] = len(fmap)-len(merged_fmap)
        scores[i]['remaining_clusters'] = len(merged_fmap)

        scores[i]['newman_modularity'] = newman_modularity
        scores[i]['erdos_renyi_mod'] = erdos_renyi_mod
        scores[i]['link_modularity'] = link_modularity
        scores[i]['average_internal_degree'] = average_internal_degree
        scores[i]['conductance'] = conductance
        scores[i]['cut_ratio'] = cut_ratio
        scores[i]['edges_inside'] = edges_inside
        scores[i]['internal_edge_density'] = internal_edge_density

    with open(output_path + network_name.split('.')[0] + '_scores'+ '.pkl', 'wb') as f:
        pickle.dump(scores, f)

    return scores

if __name__ == "__main__":

    print('hey')
