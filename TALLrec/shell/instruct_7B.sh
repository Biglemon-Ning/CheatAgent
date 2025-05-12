echo $1, $2
seed=$2
output_dir="./Saved_model"
# base_model="./Llama-2-7b-hf"
base_model="linhvu/decapoda-research-llama-7b-hf"
train_data="/raid/home/wenqi/Liangbo/TALLRec/data/ML1M/train.json"
val_data="/raid/home/wenqi/Liangbo/TALLRec/data/ML1M/valid.json"
instruction_model="/home/lemon/wenqi/Liangbo/TALLRec/alpaca-lora-7B"
# instruction_model="tloen/alpaca-lora-7b"
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 64
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u /raid/home/wenqi/Liangbo/TALLRec/finetune_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir $output_dir/$seed-$sample \
                    --batch_size 128 \
                    --micro_batch_size 32 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $2
        done
    done
done

