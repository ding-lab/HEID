#!/bin/bash
set -u
L=$CFGDIR/pair_examples.txt
[ -s "$L" ] || { echo "no pair_examples.txt: nothing to draw"; exit 0; }
while read -r P; do
  [ -z "$P" ] && continue
  D=$DELIV/align/2_pair_examples/${P/,/-}
  mkdir -p $D; rm -f $D/[1234]_*.png $D/*.tsv $D/pair_fits.json $D/crop_box.json
  OMP_NUM_THREADS=8 $PY $SCRIPTS/duct_anchor_align.py --volume $VOL --sections $P --out $D --figures 1,2,3,4 --res ${RES:-1} \
    || { echo "### FAILED pair example $P"; exit 1; }
  E=$EDGES/${P/,/-}
  for f in overlay.png overlay.pdf results.json tile_residuals.parquet; do [ -f $E/$f ] && cp $E/$f $D/0_coarse_$f; done
  grep -E "anchors \|" $D/../../../raw/logs/*_examples_*.log 2>/dev/null | tail -0
  echo "example $P: $(ls $D | wc -l) files"
done < "$L"
