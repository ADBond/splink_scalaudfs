from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType

from splink import SparkAPI, Linker, splink_datasets
import splink.comparison_library as cl

conf = SparkConf()
# This parallelism setting is only suitable for a small toy example
conf.set("spark.driver.memory", "12g")
conf.set("spark.default.parallelism", "16")

# Add custom similarity functions, which are bundled with Splink
# documented here: https://github.com/moj-analytical-services/splink_scalaudfs
path = "jars/scala-udf-similarity-0.2.0.jar"
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
        cl.JaccardAtThresholds("substr(surname, 1, 10)"),
        cl.ExactMatch("Dmetaphone(city)"),
        cl.ExactMatch("DmetaphoneAlt(city)"),
    ],
}

db_api = SparkAPI(spark_session=spark)

linker = Linker(df, settings, db_api)
df_pred = linker.inference.predict()

print(df_pred.as_pandas_dataframe(10))
