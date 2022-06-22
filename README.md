# 机器学习

**机器学习**

1. 回归和分类：回归预测具体数值；分类预测所属分类。
2. 逻辑回归：损失函数用交叉熵（非凸函数）不能用最小二乘法，因为非凸函数的特性使得其会陷入局部最优。（待定）交叉熵的原因是，loss的区别。cross-entropy可以随着目标轻松的梯度下降，而mse(mean square error)无法做到，因为mse在loss大的时候，难以轻松下降。

![](<.gitbook/assets/image (6).png>)

1. 梯度下降：初始$$\theta$$,$$\theta$$针对各维度/参数求偏导，得到梯度L($$\theta$$)；=argmin L
2. Batch：随机分的批，一批一批的$$\theta$$梯度更新$$\theta$$。一个batch叫update，所有batch叫epoch
3. Sigmod指的是logisitic回归，relu指的是max(0,wx+b);relu更好。
4. 发现问题：（1）训练集loss大：欠拟合（模型不够复杂，因子/参数不够多，梯度下降的不够彻底；解决：可以通过不同的模型测试下）(2)训练集loss小测试集loss大：过拟合。
5. 过拟合解决方法：（1）更少的参数（特指神经元），更多的共享参数。（2）更少的特征。（3）早停止（early stopping）（4）正则化（5）dropout
6. gradient等于/接近0的原因：（1）local minima （2）saddle point

**局部最小值local minima和鞍点 saddle point**

1. 判断local minima/saddle point方法：采用泰勒二阶展开式。H(hessian)是正(eigen正)，local minima;H(hessian)是负(eigen负)，local maxima;反之，saddle point。
2. saddle point比local minima更加常见，因为存在许多负的eigen value。
3. saddle point解决方法：寻找负的eigen value（H）,顺着eigenvector寻找更小loss的数值。（运算量过大，不建议使用）

**Batch批处理和动量Momentum**

1. 1 **epoch** = see all the **batches** once -> **shuffle** after each epoch
2. small batch size，小批次，每步计算loss；big batch size，大批次，积累再计算loss。
3. **计算速度：**batch size过大的情况下，时间会很长；batch size处理合理区间，由于并行计算，时间差不多。但是整体来说，一个epoch，肯定是batch size大，计算更快。
4. **梯度下降：**small batch会noisy，large batch更稳定。
5. **准确度：**一般情况，batch size越大，训练和测试的准确度越低，由于过大的batch size会容易困在local minima，但是batch很多的情况下，相当于有很多条梯度下降曲线，一条卡住下一条不一定卡住。其中，测试的时候，small batch size表现更好，由于过拟合的情况。主要由于，训练的sharp minima 可能对应着测试的loss较大。large batch 会导致更容易走到sharp minuma, small batch 会导致更容易走到flat minima。
6. **动量：**能一定程度解决local minima和saddle point问题。gradient descent + momentum。

**自动调整学习速率**

1. training stuck != small gradient；训练卡住了，不一定是到了local minima或者saddle point，有可能是learning rate过大导致在跳跃（越过了最低点）
2. 损失函数陡峭处，希望小的learning rate；损失函数平缓处，希望大的learning rate。
3. 采用root mean square的方法，过去所有的g平方和开根号当作learning rate的分母。这个方法被用作adagrad。但没有考虑到一个theta也存在陡峭和平缓（gradient的变化）
4. RMSProp加入了alpha，改变了每个gradient的重要性。
5. Adam: RMSProp+Momentum。
6. 问题一，有些时候，theta会乱喷/偏离过大，所以通过learning rate scheduling解决，包括learning rate delay(即随着t增加,learning rate变小)；warm up,learning rate先变大再变小。
7. momentum也是把所有g加起来，区别是加入了方向/向量，而root mean square是只考虑大小。动量的原因是跳跃。

**分类的损失函数**

1. softmax，把y变成y'。y\_i'处于0-1之间。且相乘是1。softmax就是normalize。会让大值和小值的差距更大。两个类用sigmod和多类用softmax是一样的。
2. 分类采用cross entropy的原因：交叉熵的原因是，loss的区别。cross-entropy可以随着目标轻松的梯度下降，而mse(mean square error)无法做到，因为mse在loss大的时候，难以轻松下降。

![](<.gitbook/assets/image (2).png>)

**批次标准化**

1. 梯度下降存在问题：首先从steep的参数开始下降，再去到smooth的参数下降。那我们期望w1和w2能同时平稳的下降。
2.

