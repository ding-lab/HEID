#!/bin/bash
set -uo pipefail
SCRIPTS=${SCRIPTS:-$RELEASE_ROOT/3d/scripts}
R3D=${THREED_ROOT:-$PROJECTS_ROOT/3d}
PY=${GEOMETRY_PYTHON:-${PYTHON:-python3}}
T=$((LSB_JOBINDEX+1))
UA=$(awk -F'\t' -v n=$T 'NR==n{print $1}' "$EDGES"); UB=$(awk -F'\t' -v n=$T 'NR==n{print $2}' "$EDGES")
TAG="U${UA}-U${UB}"
if [ -f "$OUT/$TAG/results.json" ] && $PY -c "
import json,sys
d=json.load(open(sys.argv[1])); d=d[0] if isinstance(d,list) else d
sys.exit(0 if (d.get('ok') and d.get('M_moving_to_fixed')) else 1)" "$OUT/$TAG/results.json"; then echo "SKIP: $TAG already solved"; exit 0; fi
PRE=""; [ -n "${GEOM:-}" ] && PRE="--precompensate --geometry $GEOM"
echo "=== edge $TAG  $(hostname)  $(date) ==="; t0=$SECONDS
cd $SCRIPTS
$PY pairwise_registration.py --sections "$SECTIONS" --prep-root "$PREP" --out "$OUT" --pairs "${UA}-${UB}" --index 0 \
  --mpp 2.0 --try-flip --boxes decoupled --tol-iso 0.03 --tol-dev 0.015 --tol-shear 0.01 \
  --intensity ncc --t-star-um 80.0 --tile-um 220.0 --search-um 80.0 --rot-step 10.0 --rot-range 180 \
  --epochs 40 --patience 12 --seed 42 $PRE --climb-levels none --search-mpp 16
rc=$?; echo "### $TAG rc=$rc in $((SECONDS-t0)) s"; exit $rc
