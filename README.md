# matrix

一个只依赖 Python 标准库的稠密矩阵内核：带部分主元的 LU 分解、前代回代、线性方程组
求解、行列式、秩、逆矩阵与粗略条件数估计。它不读写文件、不联网，只在内存里处理浮点矩阵。

## 目录

- matrix/core.py：矩阵内核
- tests/test_core.py：行为测试

## 跑测试

在项目根目录执行：

    python3 -m unittest discover -s tests -v
