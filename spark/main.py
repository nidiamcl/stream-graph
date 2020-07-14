from pyspark.sql.functions import collect_list, udf
from pyspark.ml.linalg import VectorUDT, SparseVector
import pandas as pd
import numpy as np

from streamgraph.graph import Graph, spark, sc



g = Graph('/mnt/d/Datasets/harvey_streams/2017-08-17 00_10_03.csv')


def getNodeAdjacency(edges, num_nodes):
    # get dataframe with bi-directional edges
    a = edges.groupby('src').agg(collect_list('dst').alias('dst'))
    b = edges \
        .withColumnRenamed('src', 'tmp') \
        .withColumnRenamed('dst', 'src') \
        .withColumnRenamed('tmp', 'dst') \
        .groupby('src').agg(collect_list('dst').alias('dst'))
    neighbors = a.union(b)

    @pandas_udf('node long, neighbors array<long>', PandasUDFType.GROUPED_MAP)
    def joinArrays(a):
        dst = np.unique(a.dst)
        return pd.DataFrame([(a.src.iloc[0], dst)], columns=['node', 'neighbors'])

    neighbors = neighbors.groupby('src').apply(joinArrays)

    # convert to sparse vectors
    @udf(VectorUDT())
    def toSparse(a):
        vector = [(dst, 1) for dst in a]
        return SparseVector(num_nodes, vector)

    neighbors = neighbors.withColumn('neighbors', toSparse('neighbors'))
    return neighbors

# @udf
# def createSparseVector(neighbors, size=g.):
#     lst_dst_val = [(dst, 1) for dst in neighbors] # I'm creting a list of tuples [(index, value), ...] to pass as args
#     return SparseVector(size, lst_dst_val)
