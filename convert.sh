#!/bin/bash

in=$1
fname=$(echo "$in"|awk -F/ '{print $NF}'|awk -F'.' '{print $1}')

if [ -d $fname ]
then
        echo "Directory $fname already exists"
        exit
fi
mkdir $fname
ffmpeg -y -i "$in" -q:v 1 -r 25 $fname/%08d.jpg