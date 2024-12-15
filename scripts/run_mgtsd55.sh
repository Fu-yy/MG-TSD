#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr3


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118




#
#
#if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result" ]; then
#    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result
#fi
#
#
#if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result" ]; then
#    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result"
#fi
#if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log" ]; then
#    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log"
#fi
#
#export model_name='mgtsd'
##export dataset="elec"
# export dataset="solar"
## export dataset="cup"
## export dataset="taxi"
## export dataset="traf"
## export dataset="wiki"
#
#export batch_size=128
#export num_cells=128
#export diff_steps=100
#
#export cuda_num=0
#export epoch=30
##export mg_dict='1_4'
##export num_gran=2
##export share_ratio_list='1_0.6'
##export weight_list='0.9_0.1'
#export mg_dict='1'
#export num_gran=1
#export share_ratio_list='1'
#export weight_list='1'
#
#
#
#
#
#if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938" ]; then
#    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938
#fi
#if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938" ]; then
#    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938
#fi
#
#export result_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938/${model_name}_${dataset}"
#export log_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938/${model_name}_${dataset}"
#if [ ! -d $result_path ]; then
#    mkdir $result_path
#fi
#if [ ! -d $log_path ]; then
#    mkdir $log_path
#fi
#
#for i in {1..10};
#do
#    echo "run $dataset _ $i"
#    python /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/src/run_mgtsd.py \
#    --result_path $result_path \
#    --model_name $model_name \
#    --epoch $epoch \
#    --cuda_num $cuda_num \
#    --dataset $dataset \
#    --diff_steps $diff_steps\
#    --batch_size $batch_size\
#    --num_cells $num_cells\
#    --mg_dict $mg_dict\
#    --num_gran $num_gran\
#    --share_ratio_list $share_ratio_list\
#    --weight_list $weight_list\
#    --run_num $i\
#    --log_metrics False \
#    >>"${log_path}/gran_${num_gran}_share_${share_ratio_list}_weight_${weight_list}_run_${i}.txt" 2>&1
#done


#-------------------------------------------------------------

#
#export model_name='mgtsd'
#export dataset="elec"
## export dataset="solar"
## export dataset="cup"
## export dataset="taxi"
## export dataset="traf"
## export dataset="wiki"
#
#export batch_size=128
#export num_cells=128
#export diff_steps=100
#
#export cuda_num=0
#export epoch=30
##export mg_dict='1_4'
##export num_gran=2
##export share_ratio_list='1_0.6'
##export weight_list='0.9_0.1'
#export mg_dict='1'
#export num_gran=1
#export share_ratio_list='1'
#export weight_list='1'
#
#
#if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938" ]; then
#    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938
#fi
#if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938" ]; then
#    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938
#fi
#
#export result_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938/${model_name}_${dataset}"
#export log_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938/${model_name}_${dataset}"
#if [ ! -d $result_path ]; then
#    mkdir $result_path
#fi
#if [ ! -d $log_path ]; then
#    mkdir $log_path
#fi
#
#for i in {1..10};
#do
#    echo "run $dataset _ $i"
#    python /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/src/run_mgtsd.py \
#    --result_path $result_path \
#    --model_name $model_name \
#    --epoch $epoch \
#    --cuda_num $cuda_num \
#    --dataset $dataset \
#    --diff_steps $diff_steps\
#    --batch_size $batch_size\
#    --num_cells $num_cells\
#    --mg_dict $mg_dict\
#    --num_gran $num_gran\
#    --share_ratio_list $share_ratio_list\
#    --weight_list $weight_list\
#    --run_num $i\
#    --log_metrics False \
#    >>"${log_path}/gran_${num_gran}_share_${share_ratio_list}_weight_${weight_list}_run_${i}.txt" 2>&1
#done

#-------------------------------------------------------------
export model_name='mgtsd'
#export dataset="elec"
# export dataset="solar"
 export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

export batch_size=128
export num_cells=128
export diff_steps=100

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'
export mg_dict='1'
export num_gran=1
export share_ratio_list='1'
export weight_list='1'


if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938"
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938"
fi

export result_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938/${model_name}_${dataset}"
export log_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..10};
do
    echo "run $dataset _ $i"
    python /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --mg_dict $mg_dict\
    --num_gran $num_gran\
    --share_ratio_list $share_ratio_list\
    --weight_list $weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/gran_${num_gran}_share_${share_ratio_list}_weight_${weight_list}_run_${i}.txt" 2>&1
done

#-------------------------------------------------------------
export model_name='mgtsd'
#export dataset="elec"
# export dataset="solar"
# export dataset="cup"
 export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

export batch_size=128
export num_cells=128
export diff_steps=100

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'
export mg_dict='1'
export num_gran=1
export share_ratio_list='1'
export weight_list='1'


if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938"
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938"
fi

export result_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938/${model_name}_${dataset}"
export log_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..10};
do
    echo "run $dataset _ $i"
    python /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --mg_dict $mg_dict\
    --num_gran $num_gran\
    --share_ratio_list $share_ratio_list\
    --weight_list $weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/gran_${num_gran}_share_${share_ratio_list}_weight_${weight_list}_run_${i}.txt" 2>&1
done

#-------------------------------------------------------------
export model_name='mgtsd'
#export dataset="elec"
# export dataset="solar"
# export dataset="cup"
# export dataset="taxi"
 export dataset="traf"
# export dataset="wiki"

export batch_size=128
export num_cells=128
export diff_steps=100

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'
export mg_dict='1'
export num_gran=1
export share_ratio_list='1'
export weight_list='1'


if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938"
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938"
fi

export result_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938/${model_name}_${dataset}"
export log_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..10};
do
    echo "run $dataset _ $i"
    python /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --mg_dict $mg_dict\
    --num_gran $num_gran\
    --share_ratio_list $share_ratio_list\
    --weight_list $weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/gran_${num_gran}_share_${share_ratio_list}_weight_${weight_list}_run_${i}.txt" 2>&1
done

#-------------------------------------------------------------

export model_name='mgtsd'
#export dataset="elec"
# export dataset="solar"
# export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
 export dataset="wiki"

export batch_size=128
export num_cells=128
export diff_steps=100

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'
export mg_dict='1'
export num_gran=1
export share_ratio_list='1'
export weight_list='1'


if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938"
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938" ]; then
    mkdir "/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938"
fi

export result_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/result/log_202412130938/${model_name}_${dataset}"
export log_path="/mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/log/log_202412130938/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..10};
do
    echo "run $dataset _ $i"
    python /mnt/nfs/data/home/1120231455/home/fuy/python/MG-TSDclone/src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --mg_dict $mg_dict\
    --num_gran $num_gran\
    --share_ratio_list $share_ratio_list\
    --weight_list $weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/gran_${num_gran}_share_${share_ratio_list}_weight_${weight_list}_run_${i}.txt" 2>&1
done

#-------------------------------------------------------------