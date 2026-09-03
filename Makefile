export PYTHONDONTWRITEBYTECODE=1

.PHONY: check-commons check-splink build package build-and-check

SERVICE       := build
CONTAINER     := splink_scalaudfs_build
VERSION       := 0.2.2
SPARK_COMPAT  ?= spark4
MAVEN_PROFILE = $(if $(filter spark3,$(SPARK_COMPAT)),-Pspark3,)
JAR           = scala-udf-similarity-$(VERSION)_$(SPARK_COMPAT).jar

check-splink:
	uv run --group ${SPARK_COMPAT} python dev/splink_compat.py jars/$(JAR)

package:
	docker compose build --build-arg MAVEN_PROFILE="$(MAVEN_PROFILE)"
	docker compose create $(SERVICE)
	docker compose start $(SERVICE)
	docker exec -t $(CONTAINER) ls -ls /app/target/
	docker cp $(CONTAINER):/app/target/$(JAR) jars/$(JAR)
	docker compose stop $(SERVICE)
	docker compose rm -f $(SERVICE)

build-and-check: package check-splink
