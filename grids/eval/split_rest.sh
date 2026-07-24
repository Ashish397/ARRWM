#!/bin/bash
for g in /home/ashish/ARRWM/grids/grids_A/A/*_grid.mp4; do
  base=$(basename "$g" _grid.mp4)
  if [ ! -f "tiles/${base}__pca4.mp4" ] && [ ! -f "tiles_new/${base}__pca4.mp4" ]; then
    ffmpeg -v error -y -i "$g" \
      -filter_complex "[0:v]crop=832:448:832:32[p4];[0:v]crop=832:448:1664:32[p2];[0:v]crop=832:448:0:512[n4];[0:v]crop=832:448:832:512[nat];[0:v]crop=832:448:1664:512[nad]" \
      -map "[p4]" -c:v libx264 -crf 12 -preset veryfast "tiles_new/${base}__pca4.mp4" \
      -map "[p2]" -c:v libx264 -crf 12 -preset veryfast "tiles_new/${base}__pca2.mp4" \
      -map "[n4]" -c:v libx264 -crf 12 -preset veryfast "tiles_new/${base}__4node.mp4" \
      -map "[nat]" -c:v libx264 -crf 12 -preset veryfast "tiles_new/${base}__noatok.mp4" \
      -map "[nad]" -c:v libx264 -crf 12 -preset veryfast "tiles_new/${base}__noadaln.mp4"
  fi
done
echo "split done: $(ls tiles_new | wc -l) files in tiles_new"
