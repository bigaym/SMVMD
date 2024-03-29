### SMVMD(Successive Multivariate Variational Mode Decomposition) 

MVMD的改进版本，主要改进计算速度；将其从MATLAB中移植到Python中。

原MATLAB版本：https://github.com/Yoricko/SMVMD/blob/main/SMVMD.m

参数介绍：
signal    - 需要分解的二维数组；

alphaMin  -  与MVMD中alpha类似，相当于初始alpha；

alphaMax  - 与MVMD中alpha类似，相当于迭代中允许的最大alpha；

beta     -  alphaMin在迭代过程中的变化比率，需要大于1，相当于每次迭代后alphaMin *=alphaMin  * beta，知道alphaMin  和alphaMax一样大；

 init      - 0 = 初始中心频率为0，否则随机设置；

tau       - 一般为0；

eps1      - 阈值容限，1e-6或1e-7即可；

eps2      - 同时；

K        - 需要分解的分量数

返回值：
u       - 分解结果，维度0为分量
u_hat   - 分量对应频谱
omega   - 分量的中心频率

