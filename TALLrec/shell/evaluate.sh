CUDA_ID=$1
attack_mode=$2
if [ ! -d './shell/'$2 ];then
    mkdir './shell/'$2
    cp -r './shell/Benign/2048-64' './shell/'$2
fi

if [ -f './shell/'$2'/2048-64.json' ];then
    if [ ! -d './shell/'$2'/Previous_results' ];then
        mkdir './shell/'$2'/Previous_results'
    fi
    mv './shell/'$2'/2048-64.json' './shell/'$2'/Previous_results/'$(date "+%Y_%m_%d_%H_%M_%S").json
fi
output_dir='./shell/'$2'/2048-64'
model_path=$(ls -d $output_dir*)
base_model="linhvu/decapoda-research-llama-7b-hf"
# base_model="./alpaca-lora-7B"
test_data='/raid/home/wenqi/Liangbo/TALLRec/data/ML1M/test_filtered.json'
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model $base_model \
        --lora_weights $path \
        --test_data_path $test_data \
        --result_json_data $output_dir.json \
        --attack_mode $2
done
