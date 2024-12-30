

if [ ! -d "./result" ]; then
    mkdir "./result"
fi
if [ ! -d "./log" ]; then
    mkdir "./log"
fi

export freq_rate_list='1'
export freq_weight_list='1'

# 测试 1
for freq_rate_list in 0.9 0.8 0.7  0.6 0.5 0.4 0.3 0.2 0.1;
do

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

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'

export freq_ranges='0,60'




if [ ! -d "./result/log_202412301832" ]; then
    mkdir "./result/log_202412301832"
fi
if [ ! -d "./log/log_202412301832" ]; then
    mkdir "./log/log_202412301832"
fi

export result_path="./result/log_202412301832/${model_name}_${dataset}"
export log_path="./log/log_202412301832/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi
#

for i in {1..2};
do
    echo "run $dataset _ $i"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/freq_rate_list_${freq_rate_list}_freq_weight_list_${freq_weight_list}_run_${i}.txt" 2>&1
done
#

#-------------------------------------------------------------
#

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

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202412301832" ]; then
    mkdir "./result/log_202412301832"
fi
if [ ! -d "./log/log_202412301832" ]; then
    mkdir "./log/log_202412301832"
fi

export result_path="./result/log_202412301832/${model_name}_${dataset}"
export log_path="./log/log_202412301832/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/freq_rate_list_${freq_rate_list}_freq_weight_list_${freq_weight_list}_run_${i}.txt" 2>&1
done

#-------------------------------------------------------------
export model_name='mgtsd'
#export dataset="elec"
# export dataset="solar"
 export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

export batch_size=64
export num_cells=128
export diff_steps=100

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202412301832" ]; then
    mkdir "./result/log_202412301832"
fi
if [ ! -d "./log/log_202412301832" ]; then
    mkdir "./log/log_202412301832"
fi

export result_path="./result/log_202412301832/${model_name}_${dataset}"
export log_path="./log/log_202412301832/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/freq_rate_list_${freq_rate_list}_freq_weight_list_${freq_weight_list}_run_${i}.txt" 2>&1
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



if [ ! -d "./result/log_202412301832" ]; then
    mkdir "./result/log_202412301832"
fi
if [ ! -d "./log/log_202412301832" ]; then
    mkdir "./log/log_202412301832"
fi

export result_path="./result/log_202412301832/${model_name}_${dataset}"
export log_path="./log/log_202412301832/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/freq_rate_list_${freq_rate_list}_freq_weight_list_${freq_weight_list}_run_${i}.txt" 2>&1
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



if [ ! -d "./result/log_202412301832" ]; then
    mkdir "./result/log_202412301832"
fi
if [ ! -d "./log/log_202412301832" ]; then
    mkdir "./log/log_202412301832"
fi

export result_path="./result/log_202412301832/${model_name}_${dataset}"
export log_path="./log/log_202412301832/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/freq_rate_list_${freq_rate_list}_freq_weight_list_${freq_weight_list}_run_${i}.txt" 2>&1
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



if [ ! -d "./result/log_202412301832" ]; then
    mkdir "./result/log_202412301832"
fi
if [ ! -d "./log/log_202412301832" ]; then
    mkdir "./log/log_202412301832"
fi

export result_path="./result/log_202412301832/${model_name}_${dataset}"
export log_path="./log/log_202412301832/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/freq_rate_list_${freq_rate_list}_freq_weight_list_${freq_weight_list}_run_${i}.txt" 2>&1
done

#-------------------------------------------------------------



done