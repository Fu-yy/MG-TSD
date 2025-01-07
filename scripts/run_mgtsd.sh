

if [ ! -d "./result" ]; then
    mkdir "./result"
fi
if [ ! -d "./log" ]; then
    mkdir "./log"
fi

export freq_rate_list='1'
export freq_weight_list='1'



export freq_ranges='0,60'





#------------------------------------

export model_name='mgtsd'
#export dataset="elec"
 export dataset="solar"
# export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

export batch_size=128
export num_cells=128
export diff_steps=100
export end_ratio=0.3
export rate=0.5

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501061215" ]; then
    mkdir "./result/log_202501061215"
fi
if [ ! -d "./log/log_202501061215" ]; then
    mkdir "./log/log_202501061215"
fi

export result_path="./result/log_202501061215/${model_name}_${dataset}"
export log_path="./log/log_202501061215/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _ $freq_rate_list"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --end_ratio $end_ratio\
    --rate $rate\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/end_ratio_${end_ratio}_rate_${rate}_run_${i}.txt" 2>&1
done






#------------------------------------

export model_name='mgtsd'
export dataset="elec"
# export dataset="solar"
# export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

export batch_size=128
export num_cells=128
export diff_steps=100
export end_ratio=0.3
export rate=0.5

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501061215" ]; then
    mkdir "./result/log_202501061215"
fi
if [ ! -d "./log/log_202501061215" ]; then
    mkdir "./log/log_202501061215"
fi

export result_path="./result/log_202501061215/${model_name}_${dataset}"
export log_path="./log/log_202501061215/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _ $freq_rate_list"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --end_ratio $end_ratio\
    --rate $rate\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/end_ratio_${end_ratio}_rate_${rate}_run_${i}.txt" 2>&1
done








