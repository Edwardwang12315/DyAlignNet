#!/bin/bash

# 创建目标目录（如果不存在）
mkdir -p ../../dataset/DarkFace_new/label ../../dataset/DarkFace_new/image

# 循环复制1-500编号的图片
for i in {1..500}; do
    cp "../../dataset/DarkFace/image/$i.png" "../../dataset/DarkFace_new/image" 2>/dev/null
    cp "../../dataset/DarkFace/label/$i.txt" "../../dataset/DarkFace_new/label" 2>/dev/null
done

# 忽略未找到文件的错误（2>/dev/null）

# 如果文件名严格匹配数字，可直接用通配符批量操作：

# bash
# # 复制1-500（无前导零）
# cp source_dir/{1..500}.jpg target_dir/ 2>/dev/null

# # 复制1-500（4位前导零）
# cp source_dir/{0001..0500}.jpg target_dir/ 2>/dev/null