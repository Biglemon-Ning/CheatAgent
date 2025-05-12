#!/bin/sh

# for i in "LastFM" "ML1M" "Taobao";do
for attack_mode in "PA" "FPRP" ;do
    echo Start \'$attack_mode\' $(date "+%Y-%m-%d %H:%M:%S")
    sh ./shell/evaluate.sh 0,1 $attack_mode
    echo End \'$attack_mode\' $(date "+%Y-%m-%d %H:%M:%S")
    echo '\n\n\n'
done