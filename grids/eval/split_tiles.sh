#!/bin/bash
# Split a 3328x960 grid video into 7 variant tiles (832x480 each), cropping off top 32px label strip
set -e
IN="$1"; OUTDIR="$2"; base=$(basename "$IN" _grid.mp4)
mkdir -p "$OUTDIR"
declare -A POS=( [pca8]="0:0" [pca4]="832:0" [pca2]="1664:0" [16node]="2496:0" [4node]="0:480" [noatok]="832:480" [noadaln]="1664:480" )
for v in pca8 pca4 pca2 16node 4node noatok noadaln; do
  IFS=: read x y <<< "${POS[$v]}"
  ffmpeg -v error -y -i "$IN" -vf "crop=832:448:${x}:$((y+32))" -c:v libx264 -crf 10 -preset fast -pix_fmt yuv420p "$OUTDIR/${base}__${v}.mp4"
done
