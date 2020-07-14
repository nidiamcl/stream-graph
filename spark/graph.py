import logging
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import time

import pandas
from graphframes import GraphFrame
from pyspark.context import SparkContext
from pyspark.sql.functions import (PandasUDFType, array_contains, broadcast,
                                   explode, lit, monotonically_increasing_id,
                                   pandas_udf, regexp_replace, split)
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               StringType, StructField, StructType)

from . import schemas

# Fix PyArrow IPC
os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'

# Create Spark Context
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


class Logging():
  
  def info(self, string):
    print(string)
    
logger = Logging()


class Graph():

    edge_types = ['topic', 'authority', 'hashtags', 'mentions']

    def __init__(self, path):
        ''''''
        self.nodes = None
        self.edges = None

        if path:
            self._loadFromCSV(path)

    @property
    def gf(self):
        if not hasattr(self, '_gf'):
            self._gf = GraphFrame(self.nodes, self.edges)
        return self._gf

    @property
    def num_nodes(self):
        if not hasattr(self, '_num_nodes'):
            self._num_nodes = sc.broadcast(self.nodes.count())
        return self._num_nodes

    @property
    def num_edges(self):
        if not hasattr(self, '_num_edges'):
            self._num_edges = sc.broadcast(self.edges.count())
        return self._num_edges
        
    def _loadFromCSV(self, path: Path):
        '''Loads a CSV into a Spark DataFrame.'''
        start_time = time()

        # parse csv and load in dataframe
        self.nodes = spark.read.csv(path, header=True, inferSchema=False, schema=schemas.NODE, escape='"').drop('unnamed')
        logger.info('[{:3.2f}] Loaded {} nodes.'.format(time() - start_time, self.nodes.count()))

        # add monotonically increasing id
        self.nodes = self.nodes.withColumnRenamed('id', 'tweetId').withColumn('id', monotonically_increasing_id())

        # parse lists in columns
        self.nodes = self.nodes.withColumn('hashtags', split(regexp_replace(self.nodes.hashtags, r"[\[\]'\s]", ''), ','))
        self.nodes = self.nodes.withColumn('mentions', split(regexp_replace(self.nodes.mentions, r"[\[\]'\s]", ''), ','))
        logger.info('[{:3.2f}] Parsed lists within hashtags and mentions.'.format(time() - start_time))

        # create empty edge dataframe
        self.edges = spark.createDataFrame([], schema=schemas.EDGE)
        self.edges.createOrReplaceTempView('main_edges_temp')
        self.edges.write.mode('append')
        
        # generate edges
        for name in self.edge_types:
            edges = self._genEdgesByAttributeWithGroups(name)
            logger.info('[{:3.2f}] Found {} edges by {}.'.format(time() - start_time, edges.count(), name))

        # drop duplicates
        self.edges = self.edges.dropDuplicates()
        spark.catalog.dropTempView('main_edges_temp')
        logger.info('[{:3.2f}] Found {} edges in total.'.format(time() - start_time, self.edges.count()))

    @staticmethod
    @pandas_udf('src long, dst long', PandasUDFType.GROUPED_MAP)
    def _genEdgesFromGroups(group):
        return pandas.DataFrame(list(combinations(group.id, 2)), columns=['src', 'dst'])

    def _genEdgesByAttributeWithGroups(self, attr_name):
        '''Connectes edges sharing a given attribute.'''
        attr_type = self.nodes.schema[attr_name].dataType

        if isinstance(attr_type, ArrayType):
            groups = self.nodes.withColumn('exploded', explode(self.nodes[attr_name])).groupby('exploded')
        else:
            groups = self.nodes.groupby(self.nodes[attr_name])

        edges = groups.apply(self._genEdgesFromGroups)
        self.edges = self.edges.union(edges)
        return edges
