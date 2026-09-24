R3D=${THREED_ROOT:-$PROJECTS_ROOT/3d}
SCRIPTS=${SCRIPTS:-$RELEASE_ROOT/3d/scripts}
SAMPLE=SAMPLE_ID
TAG=sample
G5=$R3D/samples/$SAMPLE
GEN=$G5
MAN=$G5/configs/sections.tsv
RAW=$G5/raw
PREP=$RAW/prep/sections
INFER=$R3D/inference/blocks
RUNMAN=$INFER/configs/blocks/run_manifest.tsv
JOIN=$RAW/celltype_join/M; WMAPS=$JOIN/wmaps; SUFFIX=""
JOIN_MATCH=section
EDGES=$RAW/reg_M
EDGES_TSV=$G5/configs/edges_M.tsv
SECTIONS_DIR=$RAW/sections
SWAP=0
SWAP_MOD=codex
SWAP_RESIDUAL=fit
XENIUM=0; XENIUM9=0
VOL=$RAW/volume_8um
TILES=volume_tiles
TILES_ROOT=$RAW
PAGE=$G5/viewer
DELIV=$G5/delivery
PRED=$JOIN/predictions
EXCLUDE=$G5/configs/exclude_sections_3d.json
OBJ3D=$RAW/objects_3d
S14=$OBJ3D/S14_nerve_3d
S12=$OBJ3D/S12_cloud
TLSROOT=$RAW/tls_per_slide
TLS_SECTIONS=$MAN
LOGS=$RAW/logs
QUEUE=general
LIGHT_QUEUE=general
NERVE=1
ANCHOR_MEM=12; ANCHOR_QUEUE=general; ANCHOR_CORES=2
AFFINE_MIN_N=6
SWAP_TILES=volume_tiles_heswap

