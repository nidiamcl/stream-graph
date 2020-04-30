import pandas as pd
import csv
from mpi4py import MPI
import os.path
import numpy as np
import h5py
import math

def to_hdf5(path):
  file_reader = pd.read_csv(path, sep='\t', delimiter=' ', chunksize=5000, header=None)
  dirname = os.path.dirname(path)
  basename = os.path.basename(path)
  with h5py.File(os.path.join(dirname,basename.split('.')[0]+'.h5'), 'w') as hf:
    dataset = hf.create_dataset('edges')
    for df_chunk in file_reader:
      hf.append('edges', df_chunk)

class GraphReader:
  def __init__(self, path, partition_path):
    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()
    self.rank = 10
    self.size = self.comm.Get_size()
    self.size = 20

    tsv_file = open(partition_path)
    read_tsv = csv.reader(tsv_file, delimiter=",")
    partitions = list(read_tsv)
    partitions = np.array([[int(a[0]), int(a[1])] for a in partitions])
    partitions = np.array(sorted(partitions, key=lambda x: x[0]))
    tsv_file.close()

    nums = partitions[:,0]
    counts = partitions[:,1]
    edges_ = np.sum(counts)
    local_edge_count = math.ceil(edges_ / self.size)

    local_vertices = self.get_vertices(partitions, local_edge_count)

    local_edges = []
    with open(path) as tsv_file:
      read_tsv = csv.reader(tsv_file, delimiter=" ")
      for row in read_tsv:
        r = list(map(int, row))
        for l in local_vertices:
          if l in r:
            local_edges.append(r)
            break

    # here we need to create a local sparse matrix
    # 1) shuffle edges so that local vertices are on right hand side
    # 2) create a local mapping so that local vertices start at 0
    # 3) create sparse matrix

    # does this need to be backward?
    self.global_mapping = {}
    for k, i in enumerate(partitions[:,0]):
      self.global_mapping[i] = k

  def get_vertices(self, partitions, local_edges):
    dist = [[] for i in range(self.size)]
    current_rank = 0

    count = 0
    for i, entry in enumerate(partitions): 
      dist[current_rank].append(entry[0])
      count += entry[1]
      if count > local_edges:
        current_rank += 1
        count = 0

    return dist[self.rank]
    
if __name__ == "__main__":
  #to_hdf5('../sample_data/protein.tsv') 
  reader = GraphReader('../sample_data/protein.tsv', '../sample_data/protein_node_edges.txt')
  

