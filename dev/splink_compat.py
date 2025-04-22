from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pandas as pd

# from pyspark.sql import types
from splink import ColumnExpression, SparkAPI, Linker, splink_datasets, block_on
import splink.comparison_library as cl
from splink.internals.spark.jar_location import similarity_jar_location

conf = SparkConf()
# This parallelism setting is only suitable for a small toy example
conf.set("spark.driver.memory", "12g")
conf.set("spark.default.parallelism", "16")

# Add custom similarity functions, which are bundled with Splink
# documented here: https://github.com/moj-analytical-services/splink_scalaudfs
path = "target/scala-udf-similarity-0.1.2.jar"
# cf existing version:
# path = similarity_jar_location()
conf.set("spark.jars", path)

sc = SparkContext.getOrCreate(conf=conf)

spark = SparkSession(sc)
spark.sparkContext.setCheckpointDir("./tmp_checkpoints")
pandas_df = splink_datasets.fake_1000
df = spark.createDataFrame(pandas_df)

settings = {
    "link_type": "dedupe_only",
    "comparisons": [
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison("dob", input_is_string=True),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
}

db_api = SparkAPI(spark_session=spark)

linker = Linker(df, settings, db_api)
df_pred = linker.inference.predict()

print(df_pred.as_pandas_dataframe(10))
