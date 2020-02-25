from evaluation import *
import matplotlib.pyplot as plt
import seaborn as sb

data_path = '../../stream_graph_data/built_networks_nx/'
out_path = '../../stream_graph_data/experiments/'

all_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for f in all_files:
    network_name = f
    g = nx.read_gpickle(data_path + network_name)
    g.name = network_name
    
    info = nx.info(g)
    density = nx.density(g)
    print(info)
    print('density: ' + str(density))
          
    d = thresholdSearch(g, network_name = network_name, 
                    initial_start=0, initial_stop=1, initial_num=10, 
                    merging_start=0, merging_stop=1, merging_num=10, log=False)
    
        # turn scores (dict) into dataframe
    df = pd.DataFrame.from_dict(d, orient='index')

    # sort by score to see best score
    df = df.sort_values(by='score', ascending=False)
    initial_t = df.initial_threshold.iloc[0]
    merging_t = df.merging_threshold.iloc[0]
    highest_score = df.score.iloc[0]
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
    
    f = open('output.txt', 'a+')
    f.write('{}\n'.format(info))
    f.write('density: {}\n'.format(density))
    f.write('initial_threshold: {}\n'.format(initial_t))
    f.write('merging_threshold: {}\n'.format(merging_t))
    f.write('highest_score: {}\n'.format(highest_score))
    f.write('initial_cluster: {}\n'.format(initial_clusters))
    f.write('clusters_merged: {}\n'.format(clusters_merged))
    f.write('final_clusters: {}\n'.format(final_clusters))
    f.write('louvain_score: {}\n'.format(mod.score))
    f.write('\n')
    f.close()

 