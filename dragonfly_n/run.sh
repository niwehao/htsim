#!/bin/bash
mkdir -p log

start=6
end=100
# 计算总的任务数量 (100 - 6 + 1 = 95)
total_steps=$((end - start + 1))

make clean

# 直接从 start 循环到 end
for n in $(seq $start $end); do
    # 执行任务
    make run > "log/${n}.txt" 
    # 为了演示，这里用 sleep 代替实际任务
    
    # --- 进度条逻辑 ---
    # 计算当前完成了第几个任务 (从1开始计数)
    current_step=$((n - start + 1))
    
    # 计算百分比
    percent=$((current_step * 100 / total_steps))
    
    # 计算进度条长度 (20个字符宽)
    completed=$((current_step * 20 / total_steps))
    remaining=$((20 - completed))
    
    # 生成字符串
    bar=$(printf "%${completed}s" | tr ' ' '#')
    space=$(printf "%${remaining}s" | tr ' ' '-')
    
    # \r 使光标回到行首，-n 不换行
    printf "\rProgress: [%-20s] %d%% (%d/%d)" "${bar}${space}" "$percent" "$current_step" "$total_steps"
done

# 完成后换行
echo -e "\nDone!"