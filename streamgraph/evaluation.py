from clusters import *
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

# path to save all scores
output_path = '../../stream_graph_data/clustered_networks/dotSimilarityHarvey/'

# path to save plots
plots_path = '../../stream_graph_data/plots/'

def plotGraph(graph, network_name, color='#A4CACA', plots_path=plots_path):
    plt.figure(3,figsize=(20,20))
    nx.draw(graph, with_labels=True, node_color=color, figsize=(500,500), node_size=500, font_size=6)
    plt.savefig(plots_path + network_name + '_graph.png')
    
def thresholdSearch(graph, network_name, output_path=output_path, initial_start=0, initial_stop=1, initial_num=10, merging_start=0, merging_stop=1, merging_num=10):
    ''' calculates fitness score for combinations of thresholds values
    
        graph:           networkx graph 
        initial_start:   starting value of the initial threshold sequence
        initial_stop:    end value of the initial threshold sequence
        initial_num:     number of evenly spaced threshold values to generate
        merging_start:   the starting value of the merging threshold sequence
        merging_stop:    the end value of the merging threshold sequence
        merging_num:     number of evenly spaced threshold values to generate
    
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
    initial_thresholds = list(np.linspace(initial_start,initial_stop,initial_num, endpoint=True))[1:] 
    merging_thresholds = list(np.linspace(merging_start, merging_stop, merging_num, endpoint=True))[1:]
    combined_thresholds = [[i,m] for i in initial_thresholds for m in merging_thresholds]

    method = 'nidia'
    scores = defaultdict(dict)
    
    for i, pair in enumerate(tqdm(combined_thresholds)):
        initial_threshold = pair[0]
        merging_threshold = pair[1]

        fps, fmap = findClusters(graph, initial_threshold)
        merged_fps, merged_fmap = mergeFingerprints(fps, fmap, merging_threshold)

        
        # need to create a NodeClustering object
        # clusters:     list of communities (list of elements)
        # graph:        networkx/igraph object
        # method_name: community discovery algorithm name

        clusters = list(merged_fmap.values())
        c = cdlib.classes.node_clustering.NodeClustering(clusters, graph, method)


        # modularity fitness functions
        newman_mod = evaluation.newman_girvan_modularity(graph,c).score
        erdos_renyi_mod = evaluation.erdos_renyi_modularity(graph,c).score
        link_mod = evaluation.link_modularity(graph,c).score
        modularity_density = evaluation.modularity_density(graph,c).score
        z_modularity = evaluation.z_modularity(graph,c).score

        # other fitness_functions
        avg_internal_degree = evaluation.average_internal_degree(graph,c).score
        conductance = evaluation.conductance(graph,c).score
        cut_ratio = evaluation.cut_ratio(graph,c).score
        edges_inside = evaluation.edges_inside(graph,c).score
        expansion = evaluation.expansion(graph,c).score
        fraction_over_median_degree = evaluation.fraction_over_median_degree(graph,c).score
        internal_edge_density = evaluation.internal_edge_density(graph,c).score
        normalized_cut = evaluation.normalized_cut(graph,c).score
        max_odf = evaluation.max_odf(graph,c).score
        avg_odf = evaluation.avg_odf(graph,c).score
        flake_odf = evaluation.flake_odf(graph,c).score
        significance = evaluation.significance(graph,c).score
        surprise = evaluation.surprise(graph,c).score
        triangle_participation_ratio = evaluation.triangle_participation_ratio(graph,c).score


        scores[i]['initial_threshold'] = initial_threshold
        scores[i]['merging_threshold'] = merging_threshold
        scores[i]['clusters_found'] = len(fmap)
        scores[i]['clusters_merged'] = len(fmap)-len(merged_fmap)
        scores[i]['remaining_clusters'] = len(merged_fmap)

        scores[i]['newman_mod'] = newman_mod
        scores[i]['erdos_renyi_mod'] = erdos_renyi_mod
        scores[i]['link_mod'] = link_mod
        scores[i]['modularity_density'] = modularity_density
        scores[i]['z_modularity'] = z_modularity
        scores[i]['avg_internal_degree'] = avg_internal_degree
        scores[i]['conductance'] = conductance
        scores[i]['cut_ratio'] = cut_ratio
        scores[i]['edges_inside'] = edges_inside
        scores[i]['expansion'] = expansion
        scores[i]['fraction_over_median_degree'] =  fraction_over_median_degree
        scores[i]['internal_edge_density'] = internal_edge_density
        scores[i]['normalized_cut'] = normalized_cut
        scores[i]['max_odf'] = max_odf
        scores[i]['avg_odf'] = avg_odf
        scores[i]['flake_odf'] = flake_odf
        scores[i]['significance'] = significance
        scores[i]['surprise'] = surprise
        scores[i]['triangle_participation_ratio'] = triangle_participation_ratio

    with open(output_path + network_name.split('.')[0] + '_scores'+ '.pkl', 'wb') as f:
        pickle.dump(scores, f)

    return scores

if __name__ == "__main__":

    # build networkx graph from file
    g = buildGraph(data_path, network_name)
    print('')
    

