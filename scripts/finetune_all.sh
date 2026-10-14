#!/usr/bin/env bash
# Train the six supervised rows of the paper under the shared protocol — three seeds each, one after another — and evaluate
# every run once on the test slide (whole regions; predictions stored under outputs/ for notebooks/02-05).
#
# The protocol (configs/dataset.yaml): leave-slide-out. SLIDE-0235 is the development slide — KODIAK pretraining, training and
# checkpoint selection (min val/loss on its reviewed regions 14 + 15) — and SLIDE-0236 is the test slide, held out at every
# stage and reported once. Every row reads the same data, input (LAP2 · — · DAPI), optimizer and budget from the YAMLs; the
# seed changes the initialization and the augmentation draws, nothing else.
#
#     conda activate dl
#     scripts/finetune_all.sh                          # the paper's six rows x every seed, train then evaluate, skipping runs already done
#     scripts/finetune_all.sh --epochs 10               # the same with a 10-epoch budget for every row (the YAMLs say 20)
#     scripts/finetune_all.sh kodiak_flows unet_flows  # a subset of rows, in the order given
#     scripts/finetune_all.sh --seeds "0"              # one seed
#     scripts/finetune_all.sh --dry-run                # print the commands only
#     scripts/finetune_all.sh --smoke                  # the same path end to end in a few minutes: unet_flows, 1 epoch, 1 test region
#
# Built for an unattended run: it re-launches itself under `caffeinate -is` (no idle / system sleep while it works), a run that
# fails is logged and listed in the final summary while the next one starts, and every evaluation stores the predictions and
# the PNG panels (see "Where things land" at the end of this header).
#
# Rows (paper order) and the commands; runs are named <row>_s<seed>:
#     unet_flows     mxseg train -c configs/train_unet.yaml     --name unet_flows_s0 --set seed=0
#     kodiak_flows   mxseg train -c configs/train_kodiak.yaml   --name kodiak_flows_s0 --set seed=0
#     dinov3_flows   mxseg train -c configs/train_kodiak.yaml   --name dinov3_flows_s0 --set seed=0 --set kodiak.checkpoint=<DINOv3 .pth>
#     dinov3_frozen  ... --set kodiak.checkpoint=<DINOv3 .pth> --set kodiak.train_blocks=0
#     kodiak_frozen  ... --set kodiak.train_blocks=0
#     cellpose_ft    mxseg train -c configs/train_cellpose.yaml --name cellpose_ft_s0 --set seed=0   (cpdino-vitb, full fine-tune)
#     kodiakb_flows  ... --set kodiak.vit_embed_dim=768 --set kodiak.vit_heads=12 --set kodiak.checkpoint=<KODIAK ViT-B ckpt>
#     dinov3b_flows  ... --set kodiak.vit_embed_dim=768 --set kodiak.vit_heads=12 --set kodiak.checkpoint=<DINOv3 ViT-B .pth>
#     kodiakb_frozen ... the same, plus --set kodiak.train_blocks=0   (encoder frozen, 4.3M UNETR head only)
#     dinov3b_frozen ... the same, plus --set kodiak.train_blocks=0
#     then, per run:  mxseg eval -b configs/baselines/<row>.yaml -p weights=runs/<row>_s<seed>/weights.pt --name <row>_s<seed> --split test --unit region
#                     (cellpose_ft: -p variant=...)
#     finally:        mxseg compare --split test --unit region --seeds      # mean ± std over the seeds, one row per method
#
# Where things land, per run <row>_s<seed>:
#     runs/<run>/                     weights.pt (min val/loss), checkpoints/, history.csv, resolved_config.yaml, data_manifest.json, summary.json
#     results/region/test/<run>/      metrics.json (the paper numbers), per_region.csv, run.json
#     outputs/region/test/<run>/      one uint16 instance map per test region (the notebooks read these)
#     outputs/figures/region/test/<region>/   PNG panels: image, ground truth, QC review, and one verdict map per method
#     runs/finetune_all.log           everything, with timestamps
#
# A run counts as trained once runs/<run>/.finetune_all.done exists (written here after a clean exit; weights.pt alone is not
# enough, it is rewritten whenever the validation loss improves, so an interrupted run has one too) and as evaluated once
# results/region/test/<run>/metrics.json exists. Everything is logged to runs/finetune_all.log.
set -euo pipefail                                               # bash 3.2 (macOS): empty arrays expand as ${arr[@]+"${arr[@]}"}

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; cat <<'USAGE'

Options:
  --seeds "0 1 2"      seeds to train per row (default: 0 1 2)
  --epochs N           training budget for every row (default: the YAMLs' 20); the schedule follows it
  --smoke              end-to-end check in minutes: unet_flows, seed 0, 1 epoch, 1 test region, runs named _smoke_<row>_s<seed>
  --no-caffeinate      do not re-launch under caffeinate
  --dry-run            print the commands, run nothing
  --no-eval            train only
  --eval-only          evaluate only (runs whose runs/<run>/weights.pt exists)
  --force              retrain / re-evaluate runs that are already done
  --zero-shot          after the trained rows, also evaluate the prompted and zero-shot rows: cellpose, cpdino, instanseg, sam3, sam3_text, sam2
  --kodiak-ckpt PATH   KODIAK pretraining checkpoint for kodiak_flows / kodiak_frozen (default: the one in configs/train_kodiak.yaml)
  --kodiak-b-ckpt PATH KODIAK ViT-B pretraining checkpoint for kodiakb_flows (default: runs/kodiak_mxif_vitb/checkpoints/last.ckpt)
  -h, --help
USAGE
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DINOV3=weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
DINOV3_B=weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth      # ViT-B/16 (85.7M): the size-matched pair of rows below
KODIAK_B=runs/kodiak_mxif_vitb/checkpoints/last.ckpt               # KODIAK pretrained at ViT-B (--kodiak-b-ckpt to override)
ALL=(unet_flows cellpose_ft kodiakb_flows dinov3b_flows kodiakb_frozen dinov3b_frozen)      # the paper's rows (ViT-B); the ViT-S rows
VITS=(kodiak_flows dinov3_flows dinov3_frozen kodiak_frozen)                                 # need runs/kodiak_mxif (make pretrain) and are named explicitly
ZERO_SHOT=(cellpose cpdino instanseg sam3 sam3_text sam2)
SEEDS="0 1 2"; EPOCHS=""; DRY=0; DO_TRAIN=1; DO_EVAL=1; FORCE=0; WITH_ZERO=0; KODIAK_CKPT=""; SMOKE=0; CAFF=1
ORIG_ARGS=("$@")
ROWS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --smoke) SMOKE=1 ;;
    --no-caffeinate) CAFF=0 ;;
    --dry-run) DRY=1 ;;
    --no-eval) DO_EVAL=0 ;;
    --eval-only) DO_TRAIN=0 ;;
    --force) FORCE=1 ;;
    --zero-shot) WITH_ZERO=1 ;;
    --kodiak-ckpt) KODIAK_CKPT="$2"; shift ;;
    --kodiak-b-ckpt) KODIAK_B="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *) ROWS+=("$1") ;;
  esac
  shift
done
PREFIX=""; EXTRA_TRAIN=(); EXTRA_EVAL=()
if [ "$SMOKE" = 1 ]; then                                        # the same path, shrunk: one row, one seed, one epoch, one region
  ROWS=(unet_flows); SEEDS="0"; PREFIX="_smoke_"; WITH_ZERO=0
  EXTRA_TRAIN=(--epochs 1 --set val.n_metric_patches=4); EXTRA_EVAL=(--n 1)
fi
[ ${#ROWS[@]} -eq 0 ] && ROWS=("${ALL[@]}")
for r in "${ROWS[@]}"; do
  case " ${ALL[*]} ${VITS[*]} " in *" $r "*) ;; *) echo "unknown row '$r' (rows: ${ALL[*]} ${VITS[*]})" >&2; exit 2 ;; esac
done
if [ -n "$EPOCHS" ]; then
  case "$EPOCHS" in ''|*[!0-9]*) echo "bad --epochs '$EPOCHS'" >&2; exit 2 ;; esac
  EXTRA_TRAIN+=(--epochs "$EPOCHS")
fi
for s in $SEEDS; do case "$s" in ''|*[!0-9]*) echo "bad seed '$s' in --seeds \"$SEEDS\"" >&2; exit 2 ;; esac; done

command -v mxseg >/dev/null || { echo "mxseg not on PATH — conda activate dl (or pip install -e .) first" >&2; exit 2; }
if [ "$CAFF" = 1 ] && [ "$DRY" = 0 ] && [ -z "${FINETUNE_ALL_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export FINETUNE_ALL_CAFFEINATED=1                              # keep the machine awake for the whole job (idle + system sleep)
  exec caffeinate -is "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
fi
DATA_ROOT="${MXDATA_ROOT:-$HOME/Desktop/MedAI/Data/mxdata}"       # the same default mxdata uses (src/mxdata/io/paths.py)
for s in SLIDE-0235 SLIDE-0236; do
  [ -d "$DATA_ROOT/$s" ] || { echo "slide folder not found: $DATA_ROOT/$s — set MXDATA_ROOT to the folder holding the SLIDE-* folders" >&2; exit 2; }
done
[ -f "$DATA_ROOT/dataset/patches.csv" ] || { echo "no dataset at $DATA_ROOT/dataset — run: mxdata sample -c configs/dataset.yaml" >&2; exit 2; }
mkdir -p runs
LOG=runs/finetune_all.log
say() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
run() {                                                          # print, then run (unless --dry-run), logging everything
  say "\$ $*"
  [ "$DRY" = 1 ] && return 0
  "$@" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

VITB="--set kodiak.vit_embed_dim=768 --set kodiak.vit_heads=12"   # DINOv3 ViT-B/16; the checkpoint must match exactly or training refuses
train_cmd() {                                                    # the train command of a row and seed, as words
  local r="$1" seed="$2" name="${PREFIX}$1_s$2"
  case "$r" in
    unet_flows)    echo mxseg train -c configs/train_unet.yaml --name "$name" --set seed="$seed" ;;
    kodiak_flows)  echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" ${KODIAK_CKPT:+--set kodiak.checkpoint="$KODIAK_CKPT"} ;;
    dinov3_flows)  echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" --set kodiak.checkpoint="$DINOV3" ;;
    dinov3_frozen) echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" --set kodiak.checkpoint="$DINOV3" --set kodiak.train_blocks=0 ;;
    kodiak_frozen) echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" --set kodiak.train_blocks=0 ${KODIAK_CKPT:+--set kodiak.checkpoint="$KODIAK_CKPT"} ;;
    cellpose_ft)   echo mxseg train -c configs/train_cellpose.yaml --name "$name" --set seed="$seed" ;;
    # ---- ViT-B/16 pair: the size-matched partner of cellpose_ft (cpdino-vitb, 85.7M). Same protocol as every row above; only the
    #      encoder differs, and it differs between these two rows only by which weights it starts from. --------------------------
    kodiakb_flows) echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" $VITB --set kodiak.checkpoint="$KODIAK_B" ;;
    dinov3b_flows) echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" $VITB --set kodiak.checkpoint="$DINOV3_B" ;;
    kodiakb_frozen) echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" $VITB --set kodiak.checkpoint="$KODIAK_B" --set kodiak.train_blocks=0 ;;
    dinov3b_frozen) echo mxseg train -c configs/train_kodiak.yaml --name "$name" --set seed="$seed" $VITB --set kodiak.checkpoint="$DINOV3_B" --set kodiak.train_blocks=0 ;;
  esac
}
weights_key() { case "$1" in cellpose_ft) echo variant ;; *) echo weights ;; esac; }   # the baseline param that names the weights

# ---- preflight: what the rows need must exist before the first minute of training is spent --------------------------------
if [ "$DO_TRAIN" = 1 ]; then
  for r in "${ROWS[@]}"; do
    case "$r" in
      dinov3b_*) [ -f "$DINOV3_B" ] || { echo "missing $DINOV3_B (DINOv3 ViT-B/16 weights)" >&2; exit 2; } ;;
      kodiakb_*) [ -f "$KODIAK_B" ] || { echo "missing $KODIAK_B (KODIAK ViT-B pretraining checkpoint, or --kodiak-b-ckpt PATH)" >&2; exit 2; } ;;
      dinov3_*) [ -f "$DINOV3" ] || { echo "missing $DINOV3 (DINOv3 ViT-S/16 weights)" >&2; exit 2; } ;;
      kodiak_*) ck="${KODIAK_CKPT:-$(sed -n 's/^ *checkpoint: *\([^ #]*\).*/\1/p' configs/train_kodiak.yaml | head -1)}"
                [ -f "$ck" ] || { echo "missing KODIAK pretraining checkpoint $ck (pretrain first, or --kodiak-ckpt PATH)" >&2; exit 2; } ;;
    esac
  done
fi

say "==== finetune_all: rows ${ROWS[*]} | seeds $SEEDS | epochs ${EPOCHS:-yaml} | train=$DO_TRAIN eval=$DO_EVAL force=$FORCE zero-shot=$WITH_ZERO dry-run=$DRY smoke=$SMOKE caffeinate=${FINETUNE_ALL_CAFFEINATED:-0}"
T0=$(date +%s)
SUMMARY=()
for r in "${ROWS[@]}"; do
  for seed in $SEEDS; do
    name="${PREFIX}${r}_s${seed}"
    t=$(date +%s)
    if [ "$DO_TRAIN" = 1 ]; then
      if [ -f "runs/$name/.finetune_all.done" ] && [ "$FORCE" = 0 ]; then
        say "---- $name: trained already (runs/$name/.finetune_all.done) — skipping training"
      else
        if [ -d "runs/$name" ] && [ "$FORCE" = 0 ]; then
          say "---- $name: runs/$name exists but never finished here (interrupted?) — remove it, or rerun with --force to train over it; skipping"
          SUMMARY+=("$name: skipped (unfinished run present)"); continue
        fi
        say "---- $name: training"
        # shellcheck disable=SC2046
        if ! eval run $(train_cmd "$r" "$seed") ${EXTRA_TRAIN[@]+"${EXTRA_TRAIN[@]}"}; then
          say "---- $name: TRAINING FAILED (see $LOG) — moving on to the next run"
          SUMMARY+=("$name: TRAINING FAILED"); continue
        fi
        [ "$DRY" = 1 ] || date > "runs/$name/.finetune_all.done"
      fi
    fi
    if [ "$DO_EVAL" = 1 ]; then
      if [ ! -f "runs/$name/weights.pt" ] && [ "$DRY" = 0 ]; then
        say "---- $name: no runs/$name/weights.pt — nothing to evaluate"
        SUMMARY+=("$name: not evaluated (no weights)"); continue
      fi
      if [ -f "results/region/test/$name/metrics.json" ] && [ "$FORCE" = 0 ]; then
        say "---- $name: evaluated already (results/region/test/$name) — skipping evaluation"
      else
        say "---- $name: evaluating on the test slide (predictions + panels stored under outputs/)"
        if ! run mxseg eval -b "configs/baselines/$r.yaml" -p "$(weights_key "$r")=runs/$name/weights.pt" --name "$name" --split test --unit region --panels ${EXTRA_EVAL[@]+"${EXTRA_EVAL[@]}"}; then
          say "---- $name: EVALUATION FAILED (see $LOG) — moving on"
          SUMMARY+=("$name: trained, EVALUATION FAILED"); continue
        fi
      fi
    fi
    SUMMARY+=("$name: done in $(( ($(date +%s) - t) / 60 )) min")
  done
done

if [ "$WITH_ZERO" = 1 ] && [ "$DO_EVAL" = 1 ]; then
  for r in "${ZERO_SHOT[@]}"; do
    t=$(date +%s)
    if [ -f "results/region/test/$r/metrics.json" ] && [ "$FORCE" = 0 ]; then
      say "---- $r: evaluated already — skipping"; continue
    fi
    say "---- $r: evaluating on the test slide (no training)"
    if ! run mxseg eval -b "configs/baselines/$r.yaml" --split test --unit region --panels; then
      say "---- $r: EVALUATION FAILED (see $LOG) — moving on"; SUMMARY+=("$r: EVALUATION FAILED"); continue
    fi
    SUMMARY+=("$r: evaluated in $(( ($(date +%s) - t) / 60 )) min")
  done
fi

say "==== finished in $(( ($(date +%s) - T0) / 60 )) min"
for s in ${SUMMARY[@]+"${SUMMARY[@]}"}; do say "  $s"; done
if [ "$DO_EVAL" = 1 ] && [ "$DRY" = 0 ]; then
  say "==== test-slide table, mean ± std over seeds (results/region/test):"
  mxseg compare --split test --unit region --seeds 2>&1 | tee -a "$LOG"
  say "next: notebooks/02 -> 03 -> 04 -> 05 read outputs/region/test/<run>/ (06 reads runs/)"
fi
if [ "$SMOKE" = 1 ] && [ "$DRY" = 0 ]; then
  say "==== smoke run left its files under runs/_smoke_*, results/region/test/_smoke_*, outputs/region/test/_smoke_* — remove them with:"
  say "     rm -rf runs/_smoke_* results/region/test/_smoke_* outputs/region/test/_smoke_* outputs/figures/region/test/*/_smoke_*"
fi
