#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.state"
# Uses committed source only, never copies x86 binaries, venvs, datasets or tokens.
git -C "$ROOT" archive --format=tar.gz --prefix=factory_mapping/ HEAD > "$ROOT/.state/factory_mapping-source.tar.gz"
echo "$ROOT/.state/factory_mapping-source.tar.gz"
