import os
import sys

from pyspark import __version__, SparkContext, SparkConf
from pyspark.sql import SparkSession

from splink import SparkAPI, Linker, block_on, splink_datasets
import splink.comparison_library as cl

print(f"Checking pyspark version: {__version__}")

conf = SparkConf()
# This parallelism setting is only suitable for a small toy example
conf.set("spark.driver.memory", "12g")
conf.set("spark.default.parallelism", "16")

# Add custom similarity functions, which are bundled with Splink
# documented here: https://github.com/moj-analytical-services/splink_scalaudfs
path = os.environ.get(
    "SCALA_UDF_JAR",
    sys.argv[1] if len(sys.argv) > 1 else "jars/scala-udf-similarity-0.2.2_spark4.jar",
)
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
        cl.JaroAtThresholds("first_name"),
        cl.JaroWinklerAtThresholds("substr(first_name, 1, 10)"),
        cl.DamerauLevenshteinAtThresholds("surname"),
        cl.LevenshteinAtThresholds("substr(surname, 2, 3)"),
        cl.JaccardAtThresholds("substr(surname, 1, 10)"),
        cl.ExactMatch("Dmetaphone(city)"),
        cl.ExactMatch("DmetaphoneAlt(city)"),
    ],
    "blocking_rules_to_generate_predictions": [
        block_on("first_name"),
        block_on("surname"),
        block_on("city"),
    ]
}

db_api = SparkAPI(spark_session=spark)

linker = Linker(df, settings, db_api)
df_pred = linker.inference.predict()

print(df_pred.as_pandas_dataframe(10))
