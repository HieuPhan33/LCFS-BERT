#!/bin/bash
counter=1
until [ $counter -gt 10 ]
do
output_file="srd_log-$counter.txt"
python train.py --model_name lcfs_bert --dataset restaurant \
 --pretrained_bert_name bert-base-cased \
 --batch_size 32 --local_context_focus cdw --SRD $counter >> $output_file
((counter++))
done