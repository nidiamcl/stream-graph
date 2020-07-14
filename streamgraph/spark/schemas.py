from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               StringType, StructField, StructType)

NODE = StructType([
    StructField('unnamed', LongType(), False),
    StructField('id', LongType(), False),
    StructField('twitterID', LongType(), False),
    StructField('timestamp', StringType(), False),
    StructField('user', StringType(), False),
    StructField('originalText', StringType(), False),
    StructField('topic', LongType(), False),
    StructField('reply', BooleanType(), False),
    StructField('inReplyToUser', StringType(), True),
    StructField('authority', StringType(), True),
    StructField('hashtags', StringType(), True),
    StructField('mentions', StringType(), True)
])

EDGE = StructType([
    StructField('src', LongType(), False),
    StructField('dst', LongType(), False)
])