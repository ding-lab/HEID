#!/bin/bash
set -u
W=$($PY $SCRIPTS/weak_pairs.py $AVOL $ALIGN/pairs)
PAIRS=$(echo "$W" | grep -v '^sec ' | grep .)
if [ -s "$ALIGN/retry_skip.txt" ]; then
  PAIRS=$(echo "$PAIRS" | grep -vxF -f "$ALIGN/retry_skip.txt" | grep .)
  echo "retry skips $(grep -c . "$ALIGN/retry_skip.txt") pairs listed in $ALIGN/retry_skip.txt"
fi
SECS=$(echo "$W" | grep '^sec ' | cut -d' ' -f2)
[ -z "$PAIRS" ] && { echo "no pair below 6 core anchors: nothing to retry"; exit 0; }
echo "retry at 1 um: $(echo "$PAIRS" | wc -l) pairs, $(echo "$SECS" | wc -l) sections"
AT="--bases G_withdrawn --anchor $ANCH --volume $AVOL --levels 1 --prep-root $PREP --manifest $MAN --edge-runs $EDGES --out $AP/volume_tiles"
echo "$SECS" | xargs -P ${RETRY_PAR:-4} -I{} bash -c "TILE_WORKERS=2 $PY $SCRIPTS/build_tiles.py --planes {} $AT && TILE_WORKERS=2 $PY $SCRIPTS/build_tiles.py --planes {} $AT --rgb" || { echo "### FAILED: 1 um alignment planes"; exit 1; }
echo "$PAIRS" | xargs -P ${RETRY_PAR:-4} -I{} bash -c "P={}; OMP_NUM_THREADS=2 $PY $SCRIPTS/duct_anchor_align.py --volume $AVOL --sections \$P --out $ALIGN/pairs/\${P/,/_} --figures 3,4 --lumen-cache $ALIGN/lumens_1um --res 1 > $ALIGN/pairs/\${P/,/_}/retry_1um.log 2>&1 || echo \"### FAILED pair \$P\""
for p in $PAIRS; do d=$ALIGN/pairs/${p/,/_}; grep -E "anchors \|" $d/retry_1um.log | tail -1 | sed "s/^/  /"; done
echo "$W" > $ALIGN/retry_1um_pairs.txt
