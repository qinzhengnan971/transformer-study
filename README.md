# transformer-study
学习Transformer &amp; NLP
stage：预训练（接话）；微调（SFT）；RLHF；
Pre_train
（一）数据问题
1.数据从哪来：预训练数据集，清洗过后的干净jsonl数据集。自己做大模型，要取vi基百科或者网页多格式数据html，pdf，json，需要数据清洗，统一格式等操作。
（高质量数据集的稀缺性）
2.数据集处理：tokenizer将token变为inputs_id；label：加入bos和eos，pad补齐；pad=-100
（二）前向传播模型架构
 1.embedding(inputs_id->原始语义向量）以gpt为例，embedding将id通过32000（不准确）的词典转化为原义向量
 2.ROSNorm/layernorm（rosnorm：数据直接都平方相加再除以n，再开方。计算量少并且只压缩大小不压缩方向。layernorm：先算均值,方差再用数据减均值再除以标准差，正态分布那套）
  为什么要先归一化？embedding输出的数据质量不好，数值太大，梯度爆炸，数值太小，梯度消失，并且数据落差太大影响收敛很慢学习效率太低
 3.核心transform block（attention 和 fnn）
 4.k层transform结构后，由于有加权什么的，数字乱了，还需要再一次归一化，RMSNorm或者Layernorm
 5.lm_head:将【2，10，512】通过voc_dirction得到类似【2，10，6400】的logits，这个可能性分数
 6.再softmax得到每个词的概率
 7.将最大概率映射为文本
 8.残差网络：
（三）计算loss：交叉熵函数计算出loss
1.一件事情发生的概率p（x），信息量-log p（x），熵就是所有信息量和概率乘积相加。
2.x事件，真实分布P（x），模型预测Q（x）。kl散度，相对熵
3.交叉熵函数：所有x的概率p（x）*-log（q（x））相加。p（x）中voc_id是1，其余词表是0。
(四）反向传播：算梯度
通过loss，求loss从最后一层到第一层，每一层每一个参数的偏导数，优化器再根据梯度方向和学习率更新参数，使 loss 尽量下降。
（五）梯度下降（优化器）w=w0-grad*lr
这个lr不让他是固定值，这样可以调节学习步长，做到刚开始多学一点，后面学细致一点。
SFT
