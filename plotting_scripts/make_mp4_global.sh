#!/bin/bash

#ffmpeg -r 60 -s 1920x1080 -i pic%0000d.jpeg -vcodec libx264 -crf 25  -pix_fmt yuvj420p Global_T.mp4
ffmpeg -framerate 60 -i Global_cff_%0000d.jpeg -c:v libx264 -profile:v high -crf 15 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuvj420p Global_cff.mp4
