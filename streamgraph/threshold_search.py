from evaluation import *
import matplotlib.pyplot as plt
import seaborn as sb
from cdlib import evaluation
from cdlib.algorithms import louvain, label_propagation
from reader import *
from mpi4py import MPI

# env 
# source python37_env/bin/activate  
# cd Projects/stream-graph/streamgraph


# path to edge list
data_path = '../../../Code/stream_graph_data/test_networks/'

# path to save scores
output_path = '../../../Code/stream_graph_data/clustered_networks/test_dS_mI/'

# get all files in datapath 
all_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
# networks = [file.split('.')[0] for file in all_files if 'tsv' in file]

networks = ['syn4com2']

for network in networks:
    
    network_name = network
    g = nx.read_edgelist(data_path + network + '.tsv', delimiter=' ', nodetype=int, edgetype=int, create_using=nx.Graph())

    gr = GraphReader(data_path + network + '.tsv', data_path + network + '_node_edges.txt')
    csr_matrix = gr.read() # csr sparse matrix from the reader
    
    sim1='dotsim'
    sim2='nmi'

    print('Similarity: using ' + sim1 + ' and ' + sim2)
    g.name = network_name
    info = nx.info(g)
    density = nx.density(g)
    print(info)
    print('density: ' + str(density))

    # thresholdSearch function returns the scores (dict) with scores and info about clusters 
    # for every combination of initial and merging threshold values
    # also saves all scores to pkl
    
    d = thresholdSearch(g, csr_matrix, network_name = network_name, 
                    output_path=output_path, 
                    sim1='dotsim', sim2='nmi', 
                    initial_start=0, initial_stop=1, initial_num=30, 
                    merging_start=0, merging_stop=1, merging_num=30)

    # ----------------------------------------------------------------------------
    # the part below will just print some info of the run

    # turn scores (dict) into dataframe
    df = pd.DataFrame.from_dict(d, orient='index')

    # sort by score to get best score
    df = df.sort_values(by='newman_modularity', ascending=False)
    initial_t = df.initial_threshold.iloc[0]
    merging_t = df.merging_threshold.iloc[0]
    
    # type of fitness here
    highest_score = df.newman_modularity.iloc[0]
    
    initial_clusters = df.clusters_found.iloc[0]
    clusters_merged = df.clusters_merged.iloc[0]
    final_clusters = df.remaining_clusters.iloc[0]
    
    print('initial_threshold: ' + str(initial_t))
    print('merging_threshold: ' + str(merging_t))
    print('highest_score:' + str(highest_score))
    print('initial_clusters:' + str(initial_clusters))
    print('clusters_merged:' + str(clusters_merged))
    print('remaining_clusters:' + str(final_clusters))
    
    communities = louvain(g)
    lou_mod = evaluation.newman_girvan_modularity(g,communities)
    print('louvain: ' +  str(lou_mod.score))

    coms = label_propagation(g)
    lp_mod = evaluation.newman_girvan_modularity(g,coms)
    print('label_propagation: ' +  str(lp_mod.score))
    print('')



    