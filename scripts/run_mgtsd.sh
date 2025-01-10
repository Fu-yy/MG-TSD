

if [ ! -d "./result" ]; then
    mkdir "./result"
fi
if [ ! -d "./log" ]; then
    mkdir "./log"
fi

export freq_rate_list='1_1'
export freq_weight_list='1'



export freq_ranges='0,60'













#-------------------------------------------------------------

# export dataset="solar"
# export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

# stack mean  low,high
#30	 mgtsd	0.063674414	0.075844457	0.639238516	0.033485001	0.040899262	0.04815147  0.8_0.2 fourier0.5  100 20
#30	 mgtsd	0.087821094	0.103559215	0.988085511	0.067284391	0.079618577	0.094004765 0.2_0.8 fourier0.5  100 20
#30, mgtsd, 0.24101308, 0.259562107, 2.87571266, 0.2336544, 0.24420406, 0.29372846  0.8_0.2 fourier0.7  100 20
#30, mgtsd, 0.06203823, 0.075905238, 0.70401479, 0.036123, 0.045642486, 0.05894445  0.8_0.2 fourier0.7  100 20
#30, mgtsd, 0.05741508, 0.07026296, 0.61973680, 0.032590402, 0.0406667453, 0.0526693682  0.9_0.1 fourier0.9  100 20
#30, mgtsd, 0.08979863, 0.1066426, 1.08059687,  0.073492101, 0.085744067, 0.1007337073  0.9_0.1 fourier0.9  100 20
#30, mgtsd, 0.0779760444, 0.091776, 0.89417,    0.0581388269, 0.06755162344372696, 0.078   0.8_0.2 fourier0.5 100 20
#30	 mgtsd	0.078180289	0.09586089	0.956010474	0.04822537	0.059873088	0.071663789     0.9_0.1 fourier_rate=0.5

# 高频数据低频数据测试  低0.5高1


export model_name='mgtsd'
export dataset="elec"
export freq_rate_list=1_1  # 样本权重

export batch_size=64
export num_cells=128
export diff_steps_low=100
export diff_steps_high=20
export fourier_rate=0.5
export freq_weight_list=1_1 # loss权重
export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done











# 高频数据低频数据测试 低1高0.5

export model_name='mgtsd'
export dataset="elec"
export freq_rate_list=1_0.5  # 样本权重

export batch_size=64
export num_cells=128
export diff_steps_low=100
export diff_steps_high=20
export fourier_rate=0.5
export freq_weight_list=1_1 # loss权重
export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done














# 高频loss低频loss测试 低0.5高1

export model_name='mgtsd'
export dataset="elec"
export freq_rate_list=1_1  # 样本权重

export batch_size=64
export num_cells=128
export diff_steps_low=100
export diff_steps_high=20
export fourier_rate=0.5
export freq_weight_list=0.5_1 # loss权重
export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done




# 高频loss低频loss测试 低1高0.5

export model_name='mgtsd'
export dataset="elec"
export freq_rate_list=1_1  # 样本权重

export batch_size=64
export num_cells=128
export diff_steps_low=100
export diff_steps_high=20
export fourier_rate=0.5
export freq_weight_list=1_0.5 # loss权重
export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done




# 高频扩散个数 低100 高50

export model_name='mgtsd'
export dataset="elec"
export freq_rate_list=1_1  # 样本权重

export batch_size=64
export num_cells=128
export diff_steps_low=100
export diff_steps_high=50
export fourier_rate=0.5
export freq_weight_list=1_1 # loss权重
export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done







# 傅里叶rate 0.2

export model_name='mgtsd'
export dataset="elec"
export freq_rate_list=1_1  # 样本权重

export batch_size=64
export num_cells=128
export diff_steps_low=100
export diff_steps_high=50
export fourier_rate=0.2
export freq_weight_list=1_1 # loss权重
export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done




# 傅里叶rate 0.8

export model_name='mgtsd'
export dataset="elec"
export freq_rate_list=1_1  # 样本权重

export batch_size=64
export num_cells=128
export diff_steps_low=100
export diff_steps_high=50
export fourier_rate=0.8
export freq_weight_list=1_1 # loss权重
export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'



if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done




#-------------------------------------------------------------
#export model_name='mgtsd'
##export dataset="elec"
# export dataset="solar"
## export dataset="cup"
## export dataset="taxi"
## export dataset="traf"
## export dataset="wiki"
##30	 mgtsd	0.487046084	0.537264034	1.126754751	0.439450391	0.472045464	0.938661122  0.8_0.2  fourier_rate=0.5  100 20
##30	 mgtsd	0.552554886	0.601633502	1.278591079	0.503475571	0.542161792	1.081564252  0.5_0.5  fourier_rate=0.5  100 20
##30	 mgtsd	0.532446294	0.580848308	1.223877591	0.482939715	0.524737908	1.02873966   1_1      fourier_rate=0.5  100 20
##30, mgtsd, 0.5418789768105592, 0.5905029092039484, 1.2257307531032375, 0.49995616728096265, 0.5277237342801936, 1.036278357414605
#
#
#
#export batch_size=128
#export freq_rate_list='1_1'
#export num_cells=128
#export diff_steps_low=100
#export diff_steps_high=20
#export fourier_rate=0.5
#export freq_weight_list=0.5_0.5
#
#export cuda_num=0
#export epoch=30
##export mg_dict='1_4'
##export num_gran=2
##export share_ratio_list='1_0.6'
##export weight_list='0.9_0.1'
#
#
#
#if [ ! -d "./result/log_202501080825" ]; then
#    mkdir "./result/log_202501080825"
#fi
#if [ ! -d "./log/log_202501080825" ]; then
#    mkdir "./log/log_202501080825"
#fi
#
#export result_path="./result/log_202501080825/${model_name}_${dataset}"
#export log_path="./log/log_202501080825/${model_name}_${dataset}"
#if [ ! -d $result_path ]; then
#    mkdir $result_path
#fi
#if [ ! -d $log_path ]; then
#    mkdir $log_path
#fi
#
#for i in {1..2};
#do
#    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
#    python ../src/run_mgtsd.py \
#    --result_path $result_path \
#    --model_name $model_name \
#    --epoch $epoch \
#    --cuda_num $cuda_num \
#    --dataset $dataset \
#    --diff_steps_low $diff_steps_low\
#    --diff_steps_high $diff_steps_high\
#    --fourier_rate $fourier_rate \
#    --batch_size $batch_size\
#    --num_cells $num_cells\
#    --freq_ranges $freq_ranges\
#    --freq_rate_list $freq_rate_list\
#    --freq_weight_list $freq_weight_list\
#    --run_num $i\
#    --log_metrics False \
#    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
#done


#-------------------------------------------------------------
export model_name='mgtsd'
#export dataset="elec"
# export dataset="solar"
 export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

export batch_size=32
export num_cells=128
export diff_steps_low=100
export diff_steps_high=20
export fourier_rate=0.5
export freq_weight_list=0.5_0.5
export freq_rate_list=1_1

export cuda_num=0
export epoch=30
#export mg_dict='1_4'
#export num_gran=2
#export share_ratio_list='1_0.6'
#export weight_list='0.9_0.1'

#30	 mgtsd	0.614406005	0.719099115	1.43760025	0.332472663	0.438684865	0.533001382  fourier_rate=0.5  freq_weight_list=0.2_0.8


if [ ! -d "./result/log_202501080825" ]; then
    mkdir "./result/log_202501080825"
fi
if [ ! -d "./log/log_202501080825" ]; then
    mkdir "./log/log_202501080825"
fi

export result_path="./result/log_202501080825/${model_name}_${dataset}"
export log_path="./log/log_202501080825/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..2};
do
    echo "run $dataset _ $i _fourier_rate_ $fourier_rate"
    python ../src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps_low $diff_steps_low\
    --diff_steps_high $diff_steps_high\
    --fourier_rate $fourier_rate \
    --batch_size $batch_size\
    --num_cells $num_cells\
    --freq_ranges $freq_ranges\
    --freq_rate_list $freq_rate_list\
    --freq_weight_list $freq_weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/diff_steps_low_${diff_steps_low}_diff_steps_high_${diff_steps_high}_fourier_rate_${fourier_rate}_freq_weight_list_${freq_weight_list}_freq_rate_list_${freq_rate_list}_run_${i}.txt" 2>&1
done


