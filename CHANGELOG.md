# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-10-23

### Changed

- Compatible with Spark 4
- Updated dependency versions
- Greatly reduced `.jar` size

### Removed

- Removed `sqlEscape`, `Q2gramTokeniser`, `Q3gramTokeniser`, `Q4gramTokeniser`, `Q5gramTokeniser`, `Q6gramTokeniser`

## [0.1.2] - 2025-04-24

### Changed

- Updated dependency versions

## [0.1.1] - 2023-04-10

### Added

- Added `LevDamerauDistance` for computing Damerau-Levenshtein distance
- Added `JaroSimilarity`

## [0.1.0] - 2023-01-08

### Changed

- Compatible with Spark 3
- Updated dependency versions

### Removed

- Removed `guessNameLanguage`, `NysiisEncode`, `BeiderMorseEncode`

## [0.0.10] - 2021-11-06

### Added

- Added  `guessNameLanguage`, `NysiisEncode`, `BeiderMorseEncode`

## [0.0.9] - 2021-10-05

### Fixed

- Null handling on UDFs of the form UDF(string1,string2)

## [0.0.8] - 2021-03-01

### Added

- Added `sqlEscape` and `latlongexplode`

### Fixed

- Null handling on UDFs of the form UDF(string1,string2)

## [0.0.x]

See [./dev/README.md](historic README) for some details on older versions.

[0.2.0]: https://github.com/ADBond/splink_scalaudfs/pull/2
[0.1.2]: https://github.com/ADBond/splink_scalaudfs/pull/1
[0.1.1]: https://github.com/moj-analytical-services/splink_scalaudfs/pull/6
