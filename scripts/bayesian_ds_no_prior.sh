#!/bin/bash

datasets=("SummEval")

for dataset in ${datasets[@]}
do
    mkdir -p results/plots/${dataset}-bayds
    mkdir -p results/logs/${dataset}-bayds
    python main.py --estimator None \
                    --dataset ${dataset} \
                    --calibrator BayesianDawidSkene \
                    --compare_models GPT-2___All \
                    --plot_dir results/plots/${dataset}-bayds/ood_prior | tee results/logs/${dataset}-bayds/ood_prior.log
done
