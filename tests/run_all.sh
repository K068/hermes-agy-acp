#!/bin/sh
# Full mock acceptance suite. No real `agy acp`, no quota, no network.
set -eu
cd "$(dirname "$0")/../.."          # ~/.hermes/plugins
exec python3 -W ignore::ResourceWarning -m unittest discover -s agy_acp/tests -t . "$@"
