#  splink_scalaudfs

Scala functions, for use in Spark, designed for use with [Splink](https://moj-analytical-services.github.io/splink/).

## Jars

Built packages can be found in the [jars folder](./jars/).

These are not published in any package repository, but relevant ones are bundled with Splink.

## Dev

The goal is to build the package into a `.jar` file, which can then be used by Splink.

This is done using [maven](https://maven.apache.org/), via docker. Just run `make package`.

You can check if it is working with Splink by running `make check-splink`.

### Further info

For more information check out the [dev readme](./dev/README.md).

## Source

This module was started as an extension of an example provided in [1]
Phillip Lee (ONS) has created an example of a UDF defined in Scala, callable from PySpark,
that wraps a call to JaroWinklerDistance from Apache commons.

## Package version history

v.0.2.0

* Spark 4 compatible version
* removed Qngramtokenizer n > 1

v.0.1.2

* Updated package dependency versions

v.0.1.1

* Added levenstein-damerau distance to the UDFs provided.

v.0.1.0

* ensured databricks installations got working jaro_winkler as there was a problem manifesting only on those spark installations.
* took out some not used udfs in order to make fatjar a bit smaller

v.0.0.10

* added BeiderMorseEncode UDF
* added NysiisEncode UDF
* added guessNameLanguage UDF

v.0.0.9

* added null handling on UDFs of the form UDF(string1,string2)

v.0.0.8

* Removed Logit and Expit UDFs
* added latlongexplode UDF
* added escapeSQL


v.0.0.7

* Added DualArrayExplode UDF . Also added Logit and Expit UDFs (experimental). Added alternate encoding of Double Metaphone from Apache Commons 


v.0.0.6

* Added QgramTokenisers for Q3grams,Q4grams,Q5grams,Q6grams 

v.0.0.5

* Added a small QgramTokeniser 

v.0.0.4

* Added Double Metaphone from Apache Commons  ( org.apache.commons.codec.language._ )


v.0.0.3

* cleaning up and housekeeping

v.0.0.2

* JaroWinklerSimilarity has been used instead of JaroWinklerDistance 
* Added CosineDistance and JaccardSimilarity from Apache Commons

v.0.0.1

* get this mechanism working and output JaroWinklerDistance jar. Test that its working on AP.

---

## References:

[1] [Using a Scala UDF Example](https://github.com/ONSBigData/scala_udf_example)  @philip-lee-ons
