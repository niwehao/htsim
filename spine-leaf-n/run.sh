#!/bin/bash
mkdir -p log

total=100
make clean
for n in $(seq 1 $total); do
    # 执行任务
    make run > "log/${n}.txt"
    
    # --- 进度条逻辑 ---
    # 计算百分比
    percent=$((n * 100 / total))
    # 计算进度条长度 (20个字符宽)
    completed=$((n * 20 / total))
    remaining=$((20 - completed))
    
    # 生成字符串
    bar=$(printf "%${completed}s" | tr ' ' '#')
    space=$(printf "%${remaining}s" | tr ' ' '-')
    
    # \r 使光标回到行首，-n 不换行
    printf "\rProgress: [%-20s] %d%% (%d/%d)" "${bar}${space}" "$percent" "$n" "$total"
done

# 完成后换行
echo -e "\nDone!"