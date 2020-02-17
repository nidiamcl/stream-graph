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

amazon = 'amazon/com-amazon.ungraph.txt'
email = 'email/email-Eu-core.txt'
dblp = 'dblp/com-dblp.ungraph.txt'
wikipedia = 'wikipedia/wiki-topcats.txt'
youtube = 'youtube/com-youtube.ungraph.txt'

data_path = '../../stream_graph_data/networks_ground_truth_communities/'

def buildGraph(data_path, network_name):
    ''' builds networkx graph from file and returns it as a networkx object '''
  
    with open(data_path + network_name) as f:
        
        if network_name == email:
            edges = f.readlines()
            edges = [tuple(line.strip().split(' ')) for line in edges]
            edges = [(int(x[0]), int(x[1])) for x in edges]
            
            graph = nx.Graph()
            graph.name = 'Email network'
            graph.add_edges_from(edges)
            
        elif network_name == wikipedia:      
            edges = f.readlines()
            edges = [tuple(line.strip().split(' ')) for line in edges]
            edges = [(int(x[0]), int(x[1])) for x in edges]     

            graph = nx.Graph()
            graph.name = 'Wikipedia network'
            graph.add_edges_from(edges)
        
        elif network_name == youtube:
            edges = f.readlines()[4:]
            edges = [tuple(line.strip().split('\t')) for line in edges]  
            edges = [(int(x[0]), int(x[1])) for x in edges]
            
            graph = nx.Graph()
            graph.name = 'Youtube network'
            graph.add_edges_from(edges)            
            
        elif network_name == amazon:
            edges = f.readlines()[4:]
            edges = [tuple(line.strip().split('\t')) for line in edges]  
            edges = [(int(x[0]), int(x[1])) for x in edges]
            
            graph = nx.Graph()
            graph.name = 'Amazon network'
            graph.add_edges_from(edges)            
            
        elif network_name == dblp:
            edges = f.readlines()[4:]
            edges = [tuple(line.strip().split('\t')) for line in edges]  
            edges = [(int(x[0]), int(x[1])) for x in edges]
            
            graph = nx.Graph()
            graph.name = 'DBLP collaboration network'
            graph.add_edges_from(edges)            
        
        else:
            print('not available')
    
    print(nx.info(graph))
    print("Network density:", nx.density(graph))
    
    return graph

def plotGraph(graph, network_name, data_path=data_path):
    plt.figure(3,figsize=(20,20))
    nx.draw(graph, with_labels=True, node_color='#A4CACA', figsize=(500,500), node_size=500, font_size=6)
    plt.savefig(data_path + network_name + 'graph.png')
    
def thresholdSearch(graph, network_name, data_path=data_path, initial_start=0, initial_stop=1, initial_num=5, merging_start=0, merging_stop=1, merging_num=5, log=False):
    ''' calculates fitness score for combinations of thresholds values
    
        graph:           networkx graph 
        initial_start:   starting value of the initial threshold sequence
        initial_stop:    end value of the initial threshold sequence
        initial_num:     number of evenly spaced threshold values to generate
        merging_start:   the starting value of the merging threshold sequence
        merging_stop:    the end value of the merging threshold sequence
        merging_num:     number of evenly spaced threshold values to generate
        log:             set to True to see command line prints with info
    
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
#     scores = defaultdict(list)
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

        # evaluate fitness score for clusters
        score = evaluation.newman_girvan_modularity(graph,c).score

        scores[i]['initial_threshold'] = initial_threshold
        scores[i]['merging_threshold'] = merging_threshold
        scores[i]['score'] = score
        scores[i]['clusters_found'] = len(fmap)
        scores[i]['clusters_merged'] = len(fmap)-len(merged_fmap)
        scores[i]['remaining_clusters'] = len(merged_fmap)
        
        # set to true to see info in command line
        if log == True:
            print(i)
            print('clusters found: ' + str(len(fmap)))   
            print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
            print('remaining clusters: ' + str(len(merged_fmap)))
            print('initial_threshold: ' + str(initial_threshold))
            print('merging threshold: ' + str(merging_threshold))
            print('score: ' + str(score))
            print('')
        else:
            pass

    with open(data_path + network_name + '.pkl', 'wb') as f:
        pickle.dump(scores, f)

    return scores

def plotScores(df, network_name, data_path=data_path):
    sns.set(font_scale=1.2, style="ticks") #set styling preferences

    plt.figure(figsize=(20,10)) 
    points = plt.scatter(df['initial_threshold'], df['merging_threshold'],
                        c=df['score'], s=40, cmap="Spectral") #set style options

    # #set limits
    plt.xlim(-0.04, 0.55)
    plt.ylim(-0.04, 1.15)

    #add a color bar
    plt.colorbar(points)
    sns.regplot('initial_threshold', 'merging_threshold',  data=df, scatter=False, color=".1", fit_reg=False)
    plt.savefig(data_path + network_name + 'scores.png')


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", required=True, action='store')
    # args = parser.parse_args()
    # network_name = args.n

    network_name = email

    # build networkx graph from file
    g = buildGraph(data_path, network_name)
    print('')
    
    # plot image of the graph and get info about size, density, etc
    # plotGraph(g, network_name)
    # print('graph saved to png')
    # print('')
    
    # calculate evaluation scores for several combination of input parameters 
    # and save scores (dict) as pkl file
    print('calculating scores for threshold combinations')
    d = thresholdSearch(g, network_name = network_name, 
                        initial_start=0, initial_stop=0.5, initial_num=10, 
                        merging_start=0, merging_stop=1, merging_num=10, log=True)
    

    # load scores (dict) from pkl file
    # with open(data_path + network_name + '.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # turn scores (dict) into dataframe
    df = pd.DataFrame.from_dict(d, orient='index')

    # sort by score to see best score
    df = df.sort_values(by='score', ascending=False)
    df.head()

    print('highest score: ' + str(df['score'].iloc[0]))
    print('initial threshold: ' + str(df['initial_threshold'].iloc[0]))
    print('merging threshold: ' + str(df['merging_threshold'].iloc[0]))
    print('clusters initially found: ' + str(df['clusters_found'].iloc[0]))
    print('clusters merged: ' + str(df['clusters_merged'].iloc[0]))
    print('remaining clusters: ' + str(df['remaining_clusters'].iloc[0]))

    # score with Louvain
    communities = louvain(g)
    mod = evaluation.newman_girvan_modularity(g,communities)
    print('score with louvain: ' + str(mod.score))
    print('')

    # plot threshold combination against scores and save image to png
    plotScores(df, network_name)
    print('plot saved to png')
