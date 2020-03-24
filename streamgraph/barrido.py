from evaluation import *
import matplotlib.pyplot as plt
import seaborn as sb

# data_path = '../../stream_graph_data/nx_graphs/harvey/'
data_path = '../../stream_graph_data/nx_graphs/harvey/'

# path to save all scores
# output_path = '../../stream_graph_data/clustered_networks/dotSimilarityHarvey/'
output_path = '../../stream_graph_data/clustered_networks/dotSimilarityHarvey/'

# get all files in datapath 
# all_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

step = 1/9

with open('../../stream_graph_data/harvey5.csv') as f:
    networks = f.readlines()
    networks = [network.strip().split(',') for network in networks]

for network in networks:
    network_name = network[0] + '.pkl'
    init = float(network[1])
    merg = float(network[2])
    
    g = nx.read_gpickle(data_path + network_name)
    g.name = network_name.split('.')[0]
    
    info = nx.info(g)
    density = nx.density(g)
    print(info)
    print('density: ' + str(density))
          
    # thresholdSearch function returns the scores (dict) with scores and info about clusters 
    # for every combination of initial and merging threshold values
    # also saves all scores to pkl
    
    
    d = thresholdSearch(g, network_name = network_name, output_path=output_path,
                    initial_start=init-step, initial_stop=init+step, initial_num=10, 
                    merging_start=merg-step, merging_stop=merg+step, merging_num=10)
    
    # ----------------------------------------------------------------------------
    # the part below will just create a file with thresholds for best modularity scores
    # along with modularity produced by louvain

    # turn scores (dict) into dataframe
    df = pd.DataFrame.from_dict(d, orient='index')

    # sort by score to get best score
    df = df.sort_values(by='newman_mod', ascending=False)
    initial_t = df.initial_threshold.iloc[0]
    merging_t = df.merging_threshold.iloc[0]
    
    # type of fitness here
    highest_score = df.newman_mod.iloc[0]
    
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
    mod = evaluation.newman_girvan_modularity(g,communities)
    print('louvain: ' +  str(mod.score))
    print('')
    
    f = open('out_small_newman_mod.txt', 'a+')
    f.write('{},{},{},{},{},{},{},{},{}\n'.format(g.name, density, initial_t, merging_t, highest_score, initial_clusters, clusters_merged, final_clusters, mod.score))
    f.close()

 