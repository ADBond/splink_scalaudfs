export PYTHONDONTWRITEBYTECODE=1

check-splink:
	uv run dev/splink_compat.py

package:
	mvn package
