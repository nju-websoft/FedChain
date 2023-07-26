#!/bin/bash
# Exit on first error
set -ex

# Bring the test network down
pushd ../test-network
./network.sh down
popd

