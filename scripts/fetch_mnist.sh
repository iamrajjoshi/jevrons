#!/usr/bin/env bash
# Download MNIST into data/ from Google's CVDF mirror and verify checksums.
set -euo pipefail

mkdir -p "$(dirname "$0")/../data"
cd "$(dirname "$0")/../data"

base=https://storage.googleapis.com/cvdf-datasets/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "$f.gz" ] || curl -sfLO "$base/$f.gz"
done

shasum -a 256 -c <<'EOF'
440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  train-images-idx3-ubyte.gz
3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c  train-labels-idx1-ubyte.gz
8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6  t10k-images-idx3-ubyte.gz
f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6  t10k-labels-idx1-ubyte.gz
EOF
