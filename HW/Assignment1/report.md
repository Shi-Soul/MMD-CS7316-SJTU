# Assignment 1 Report
谢炜基 @ 2025/04/25

## 数据预处理
- **验证集划分**：在非测试区的已知评分中随机抽取 5%，使用掩码矩阵 `mask_val` 标记。
- **训练矩阵**：将测试区与验证集对应位置置零，生成 `train` 矩阵；对应 `mask_mat` 为二进制标记。

## 算法
采用Item-CF算法，使用余弦相似度计算电影间的相似度，基于用户评分预测未评分电影的分数。
1. **中心化处理**：
   - 计算每部电影的平均评分（仅对未遮蔽条目求和计数），
   - 对训练矩阵减去对应电影平均值，遮蔽条目保持 0,得到矩阵 \(R\)。
2. **相似度计算**：
   - 使用矩阵运算：\(S = R^T R\)，归一化为余弦相似度；
   - 保留每行 Top-\(K\)（\(K=10\)）邻居，置其余相似度为 0。
3. **预测评分**：
   - 对所有用户-电影对向量化计算：\(\hat{R} = R S_k^T / \|S_k\|_1 + \mu\)，$\mu$ 是每个电影的平均分
   - 四舍五入并截断到 [0, 5]。


## 环境
- **硬件平台**：Intel i5-1135F7 CPU，16 GB RAM；
- **软件环境**：Windows 10，Python 3.8，NumPy 1.22.4。

## 结果
- **相似度计算**：耗时 1.79 秒；
- **总运行时间**：9.30 秒；
- **内存使用**：1545.25 MB；

- **验证集准确率**：
  - 样本数量：44155；
  - 精确匹配率：40.13% (即完全相等预测占比)；
  - 均方误差：0.9509。


