export PYTHONDONTWRITEBYTECODE=1

SERVICE       := build
CONTAINER     := splink_scalaudfs_build

check-splink:
	uv run dev/splink_compat.py

package:
	docker compose build
	docker compose create $(SERVICE)
	docker compose start $(SERVICE)
	docker cp $(CONTAINER):/app/target/scala-udf-similarity-0.2.0.jar jars/scala-udf-similarity-0.2.0.jar
	docker compose stop $(SERVICE)
	docker compose rm -f $(SERVICE)

build-and-check: package check-splink
