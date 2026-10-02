#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.state"
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || { echo 'Commit source changes before packaging; the archive contains committed source only.' >&2; exit 1; }
# Uses committed source only, never copies x86 binaries, venvs, datasets or tokens.
git -C "$ROOT" archive --format=tar.gz --prefix=factory_mapping/ HEAD > "$ROOT/.state/factory_mapping-source.tar.gz"
echo "$ROOT/.state/factory_mapping-source.tar.gz"
