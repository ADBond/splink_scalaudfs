#  splink_scalaudfs

Scala functions, for use in [Apache Spark](https://spark.apache.org/), designed for use with the data-linking python package [Splink](https://moj-analytical-services.github.io/splink/).

## Jars

Built packages can be found in the [jars folder](./jars/).

These are not published in any package repository, but relevant ones are bundled with Splink.

## Dev

The goal is to build the package into a `.jar` file, which can then be used by Splink.

This is done using [maven](https://maven.apache.org/), via docker. Just run `make package`.

You can check if it is working with Splink by running `make check-splink`.

For Spark 3 compatible versions, which need to be compiled against a different version of Scala, set the environment variable `SPARK_COMPAT=spark3`

### Further info

For more information check out the [dev readme](./dev/README.md).

## Source

This module was started as an extension of an example provided in [1]
Phillip Lee (ONS) has created an example of a UDF defined in Scala, callable from PySpark,
that wraps a call to JaroWinklerDistance from Apache commons.

## Package version history

See [the changelog](./CHANGELOG.md).

---

## References:

[1]: [Using a Scala UDF Example](https://github.com/ONSBigData/scala_udf_example)  @philip-lee-ons
