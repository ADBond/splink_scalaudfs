package uk.gov.moj.dash.linkage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalactic.TolerantNumerics


class JaroWinklerSimilarityTest extends AnyFlatSpec with Matchers {

    implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.01)

    "Wrapped JaroWinkler" should "return same results" in {

        val distance = JaroWinklerSimilarity()

        distance.call("","") should equal (1.0)
        distance.call("","a") should equal (0.0)
        distance.call("aaapppp", "") should equal (0.0)
        distance.call("frog", "fog") should equal (0.93)
        distance.call("fly", "ant") should equal (0.0)
        distance.call("elephant", "hippo") should equal (0.44)
        distance.call("hippo", "elephant") should equal (0.44)
        distance.call("hippo", "zzzzzzzz") should equal (0.0)
        distance.call("hello", "hallo") should equal (0.88)
        distance.call("ABC Corporation", "ABC Corp") should equal (0.907)
        distance.call("D N H Enterprises Inc", "D & H Enterprises, Inc.") should equal (0.95)
        distance.call("My Gym Children's Fitness Center", "My Gym. Childrens Fitness") should equal (0.942)
        distance.call("PENNSYLVANIA", "PENNCISYLVNIA") should equal (0.898)
    }
}


