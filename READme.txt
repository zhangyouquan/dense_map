高博十四讲的单目稠密建图demo，我重新调整结构，进行分文件编写，将函数声明和具体功能实现分为两个文件。使得源文件更加简洁，可读性更强。这是大型项目应该的做法。

单目稠密建图知识点有：极线搜索（块匹配）；因为观测是受噪声影响的，demo里考虑了相邻帧的观测会有一个像素的误差，基于此求出每次相邻帧匹配时的观测不确定度；最后用了高斯分布的深度滤波器来更新深度值的分布。
1）极线搜索
2）观测不确定性
3）高斯分布的深度滤波器更新深度分布

画出了在整个建图过程中，深度值的均值和方差的变化，逐渐收敛。