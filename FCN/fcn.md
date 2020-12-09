---
typora-root-url: ../PIC
typora-copy-images-to: ../PIC
---

### FCN

FCN是全卷积网络，用来处理图像的语义分割等问题，最后将生成密集的图划分标签，其中的核心思想是，因为普通卷积网络，最后通过全连接层，会使得二维信息造成缺失。

所以，FCN最后将使用反卷积来进行上采样，最后得到一个与输入图像分辨率相同的结果。

其中，对于上采样的过程，采用了skip architecture，具体的图演示为：

![image-20201209150002591](/image-20201209150002591.png)

这幅图的意思是，会选择其中的几层池化结果，进行一些处理与拼接，最后得到的池化图层，作为反卷积操作的基础矩阵。
