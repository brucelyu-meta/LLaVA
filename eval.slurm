#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition learnai
#SBATCH --nodes=1

python llava/eval/model_vqa_loader.py \
    --model-path checkpoints/frs-v1.5-13b-pretrain \
    --model-base lmsys/vicuna-13b-v1.5 \
    --question-file \
    playground/data/ml-32m_eval/ml-32m_eval_eval_text_keyword.jsonl \
    --image-folder \
    ./playground/data/ml-32m/behavior \
    --answers-file playground/response/answer-file-our-keyword-no_behavior.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --no_behavior
