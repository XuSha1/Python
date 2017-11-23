#from pyspark import SparkConf,SparkContext
#from pyspark.sql.types import *
#from pyspark.sql import SQLContext,Row
#from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import functions as F
import pandas as pd
spark=SparkSession.builder.master("local").appName("my app").config(conf=SparkConf()).getOrCreate()
sc=SparkSession.builder.config(conf=SparkConf())
df=spark.read.format('json').load(['/home/xs/Documents/Weblog.1457006400155.gz','/home/xs/Documents/Weblog.1457006400158.gz',
'/home/xs/Documents/Weblog.1457006401774.gz'])
g1=df.groupBy("captcha_id",F.substring("request_time",1,19).alias("time")).count().filter(df.captcha_id!='')
pandas_df=g1.toPandas()#转换成pandas的dataframe
data_pivot=pandas_df.pivot_table(index=["captcha_id"],columns=["time"],values=["count"])
data_pivot.to_csv("/home/xs/Documents/data.csv",header=True)