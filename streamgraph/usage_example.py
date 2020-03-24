from clusters import *

# you need a networkx graph, this is a small one for testing
g = nx.read_gpickle('../sample_data/harveysept17.pkl')
print(nx.info(g))
print(nx.density(g))

# choose thresholds
first_th = 0.222
second_th = 0.05

# find initial clusters
# findClusters returns fps (list) and fmap (dict)
fps, fmap = findClusters(g, first_th)
print('clusters found: ' + str(len(fmap)))

# merge similar clusters
merged_fps, merged_fmap = mergeFingerprints(fps, fmap, second_th)
print('clusters merged: ' + str(len(fmap)-len(merged_fmap)))
print('remaining clusters: ' + str(len(merged_fmap)))

# that's it