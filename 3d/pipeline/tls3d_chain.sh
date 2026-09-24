#!/bin/bash
set -uo pipefail
SCRIPTS=${SCRIPTS:-$RELEASE_ROOT/3d/scripts}
R3D=${THREED_ROOT:-$PROJECTS_ROOT/3d}


PY=${GEOMETRY_PYTHON:-${PYTHON:-python3}}
export OPENCV_IO_MAX_IMAGE_PIXELS=8589934592
export HTAN3D_OBJ3D=$OBJ3D HTAN3D_TLSROOT=$TLSROOT TLS_PRED_DIR=$PRED
S=$SAMPLE
TABLE=$RAW/tls_3d_table.tsv
S9=$OBJ3D/S9_tls_objects; SD=$OBJ3D/S13_structures_3d
G5=$(dirname $(dirname $RAW))
mkdir -p $S9 $DELIV/tls/1_objects_report $RAW
[ -n "${TLS_FROM:-}" ] && RESCUE_REUSE=1
step() { echo; echo "=== $1  $(date)"; [ "${1%% *}" = "${TLS_FROM:-}" ] && TLS_FROM=""; [ -n "${TLS_FROM:-}" ] && echo "(skipped: TLS_FROM=$TLS_FROM)"; return 0; }
run() { [ -n "${TLS_FROM:-}" ] && return 0; "$@"; rc=$?; [ $rc -ne 0 ] && { echo "### FAILED rc=$rc: $*"; exit $rc; }; }

step "1 per-section TLS table (canvas centroids)"
cd $SCRIPTS
case "$S" in
  HT891Z1|HT913Z1) run $PY aggregate_cohort.py --root HT891Z1=$G5/HT891Z1/raw/tls_per_slide \
                       --root HT913Z1=$G5/HT913Z1/raw/tls_per_slide --volume $S=$VOL --out-dir $RAW ;;
  S22-27909) run env HTAN3D_VOL=$VOL HTAN3D_TLSROOT=$TLSROOT HTAN3D_PRED=$PRED HTAN3D_GEN=$R3D/front/S22-27909 HTAN3D_TABLE=$TABLE $PY aggregate_two_block.py ;;
  *) run $PY aggregate_single_chain.py --sample $S --volume $VOL --pred $PRED --sections $TLS_SECTIONS --tls-root $TLSROOT --out $TABLE ;;
esac

step "2 link the 2-D TLS across sections (the linked-object table)"
cd $SCRIPTS
run $PY tls_linker.py --sample $S --volume $VOL --table $TABLE

step "2r transition fit: sections inside a linked column that carry no TLS get a relaxed second look"
cd $SCRIPTS
run $PY tls_rescue_targets.py --sample $S --objects $S9/objects.tsv --table $TABLE --volume $VOL --pred $PRED --tls-root $TLSROOT \
    --bridges $S9/bridge_candidates.tsv --out $S9/rescue_targets.json
RESCUE_SLIDES=$($PY -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))))" $S9/rescue_targets.json)
if [ -n "$RESCUE_SLIDES" ]; then
  rescue_done() { tail -n 1 "$RAW/logs/tls2d_rescue_$1.log" 2>/dev/null | grep -q -- "-> TLS"; }
  TODO=""; for sl in $RESCUE_SLIDES; do { [ "${RESCUE_REUSE:-0}" = 1 ] && rescue_done $sl; } || TODO="$TODO $sl"; done
  echo "rescue reruns: $(echo $TODO | wc -w) of $(echo $RESCUE_SLIDES | wc -w) slides"
  echo "$TODO" | tr ' ' '\n' | grep . | xargs -P ${RESCUE_PAR:-6} -I{} sh -c \
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TLS_RESCUE_JSON=$S9/rescue_targets.json TLS_OUT_ROOT=$TLSROOT $PY tls_define_he.py $S {} --outdir $TLSROOT/{} > $RAW/logs/tls2d_rescue_{}.log 2>&1"
  for sl in $RESCUE_SLIDES; do
    rescue_done $sl && continue
    echo "rescue retry (serial): $sl"
    run env TLS_RESCUE_JSON=$S9/rescue_targets.json TLS_OUT_ROOT=$TLSROOT $PY tls_define_he.py $S $sl --outdir $TLSROOT/$sl > $RAW/logs/tls2d_rescue_$sl.log 2>&1
    rescue_done $sl || { echo "### FAILED rc=1: rescue $sl (see $RAW/logs/tls2d_rescue_$sl.log)"; exit 1; }
  done
  step "2r-1 table and links again, with the rescued sections"
  case "$S" in
    HT891Z1|HT913Z1) run $PY aggregate_cohort.py --root HT891Z1=$G5/HT891Z1/raw/tls_per_slide \
                         --root HT913Z1=$G5/HT913Z1/raw/tls_per_slide --volume $S=$VOL --out-dir $RAW ;;
    S22-27909) run env HTAN3D_VOL=$VOL HTAN3D_TLSROOT=$TLSROOT HTAN3D_PRED=$PRED HTAN3D_GEN=$R3D/front/S22-27909 HTAN3D_TABLE=$TABLE $PY aggregate_two_block.py ;;
  *) run $PY aggregate_single_chain.py --sample $S --volume $VOL --pred $PRED --sections $TLS_SECTIONS --tls-root $TLSROOT --out $TABLE ;;
  esac
  cd $SCRIPTS
  run $PY tls_linker.py --sample $S --volume $VOL --table $TABLE
fi

step "2a ids from the table: tier-A objects are 3d-tls-NN in table order, everything else 2d-tls-NN"
cd $SCRIPTS
FROZEN=$G5/$S/configs/tls3d_frozen.tsv; [ -f "$FROZEN" ] || FROZEN=""
run $PY tls_tier_ids.py --sample $S --objects $S9/objects.tsv --table $TABLE ${FROZEN:+--frozen $FROZEN}

step "2b per-slide products relabelled: csv columns + the detector's own figure with ids"
run $PY relabel_tls_2d.py --sample $S --objects $S9/objects.tsv --pred $PRED --tls-root $TLSROOT
mkdir -p $DELIV/tls/3_per_slide
for f in $TLSROOT/*/*_tls.png; do c=${f%_tls.png}_tls.csv; [ -f "$f" ] && [ "$(wc -l < "$c" 2>/dev/null || echo 0)" -gt 1 ] && cp -p "$f" $DELIV/tls/3_per_slide/; done

step "2c the TLS ring layers per plane (3-D red, 2-D green) for the section stack"
run $PY $SCRIPTS/build_tls_planes.py --volume $VOL --tls-root $TLSROOT --pred $PRED

step "3 class bodies (meshes) and the cell point cloud the solid panel reads"
cd $SCRIPTS
run $PY build_meshes.py --volume $VOL --sample $S --down 6
[ -s $OBJ3D/S12_cloud/cloud_metadata.json ] || run $PY build_cloud.py --volume $VOL --sample $S

step "4 report"
cd $SCRIPTS
run $PY tls_3d_report.py --sample $S --objects $S9/objects.tsv --table $TABLE \
    --meshes $OBJ3D/S11_meshes/meshes.json --out $DELIV/tls/1_objects_report/1_objects_report

step "5 per-object crops"
[ -d $DELIV/tls/2_objects ] && $PY -c "import shutil,sys; shutil.rmtree(sys.argv[1])" $DELIV/tls/2_objects
CP=${CROP_PAR:-8}; rcs=0
for k in $(seq 0 $((CP-1))); do
  $PY tls_object_crops.py --sample $S --objects $S9/objects.tsv --table $TABLE \
    --volume $VOL --tiles ${TILES_ROOT:-$RAW}/$TILES/G_withdrawn --out $DELIV/tls/2_objects \
    ${SWAP_TILES:+--swap-tiles ${TILES_ROOT:-$RAW}/$SWAP_TILES/G_withdrawn} --shard $k/$CP > $DELIV/tls/.crops_shard_$k.log 2>&1 &
done
wait
for k in $(seq 0 $((CP-1))); do grep -q "^wrote " $DELIV/tls/.crops_shard_$k.log || { echo "### FAILED crops shard $k"; tail -5 $DELIV/tls/.crops_shard_$k.log; rcs=1; }; done
grep -h "^wrote " $DELIV/tls/.crops_shard_*.log; rm -f $DELIV/tls/.crops_shard_*.log
[ $rcs -eq 0 ] || exit 1

step "6 page"
run bash -c "$PAGE_CMD"
step "7 consistency gate: page / 2-D layers / delivery describe the same objects"
[ -n "${CFG:-}" ] && run $PY $SCRIPTS/sync_check.py $CFG
echo; echo "### chain done for $S  $(date)"
