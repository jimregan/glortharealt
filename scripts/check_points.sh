#!/bin/bash

ls|awk -F'.' '{print $1}'|sort|uniq|while read i
do
    wcids=0
    wcfd=0
    wcpts=0

    if [ ! -e "$i.id.txt" ]
    then
        echo "Missing IDs: $i"
    else
        wcids=$(wc -l "$i.id.txt"|awk '{print $1}')
    fi
    if [ ! -e "$i.txt" ]
    then
        echo "Missing face detection: $i"
    else
        wcfd=$(wc -l "$i.txt"|awk '{print $1}')
    fi
    if [ ! -e "$i.68.txt" ]
    then
        echo "Missing facial keypoints: $i"
    else
        wcpts=$(wc -l "$i.68.txt"|awk '{print $1}')
    fi

    if [ "$wcids" == 0 ]
    then
        echo "Empty IDs: $i"
    fi
    if [ "$wcfd" != "$wcpts" ]
    then
        echo "Mismatch: $i"
    fi
    if [ "$wcfd" != "$wcids" ]
    then
        echo "Mismatch: $i"
    fi
done