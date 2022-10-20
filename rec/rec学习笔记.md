# 推荐系统入门

## 一、概述

### 推荐系统意义

个性化推荐系统通过分析用户的行为日志，得到用户当前的甚至未来可能的兴趣，给不同的用户展示不同的(个性化)的页面，来提高网站或者app的点击率、转化率、留存率等指标。

搜索和推荐都是解决互联网大数据时代信息过载的手段，但是它们也存在着许多的不同：

用户意图；个性化程度；优化目标；马太效应和长尾理论；

### 推荐系统架构

<u>架构设计的核心在于平衡和妥协</u>

思考推荐系统架构考虑的第一个问题是**确定边界**：知道推荐系统要负责哪部分问题，这就是边界内的部分。在这个基础上，架构要分为哪几个部分，每一部分需要完成的子功能是什么，每一部分依赖外界的什么。

**系统架构：**

设计思想是大数据背景下如何有效利用海量和实时数据，将推荐系统按照对数据利用情况和系统响应要求出发，将整个架构分为**离线层、近线层、在线层**三个模块。系统架构是如何权衡利弊，如何利用各种技术工具帮助我们达到想要的目的的，方便我们理解为什么推荐系统要这样设计。

离线层对于数据数量和算法复杂度限制更少，没有很强的时间要求。由于没有及时加入最新的数据，所以很容易过时。在线层能更快地响应最近的事件和用户交互，但必须实时完成，这会限制使用算法的复杂性和处理的数据量。近线层介于两种方法之间。

1. 离线层：不用实时数据，不提供实时响应；
2. 近线层：使用实时数据，不保证实时响应；
3. 在线层：使用实时数据，保证实时在线服务；

![在这里插入图片描述](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20220409204658032.png)

整个数据部分其实是一整个链路，主要是三块：

客户端及服务器实时数据处理：记录实时数据；流处理平台准实时数据处理：记录准实时数据，在推荐领域基本上只有一个类别，就是用户行为数据；大数据平台离线数据处理：所有“脏活累活”复杂的操作都是在离线完成的。

**算法架构：**

**召回、粗排、排序、重排**等算法环节角度出发的，重要的是要去理解每个环节需要完成的任务，每个环节的评价体系，以及为什么要那么设计。还有一个重要问题是每个环节涉及到的技术栈和主流算法。这种角度来看是把推荐系统从前往后串起来，**其中每一个模块既有在离线层工作的，也有在在线层工作的**。

![在这里插入图片描述](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20220409211354342.png)

召回阶段我们现在主要是在保证Item质量的基础上注重覆盖率多样性，粗排阶段主要用简单的模型来解决不同路的召回和当前用户的相关性问题，最后截断到1k个以内的候选集。

粗排的原因是有时候召回的结果还是太多，精排层速度还是跟不上，所以加入粗排。粗排可以理解为精排前的一轮过滤机制，减轻精排模块的压力。粗排介于召回和精排之间，要同时兼顾精准性和低延迟。

精排是推荐系统各层级中最纯粹的一层，他的目标比较单一且集中，一门心思的实现目标的调优即可。解决样本规模和时效性问题。

重排序阶段对精排生成的Top-N个物品的序列进行重新排序，生成一个Top-K个物品的序列，作为排序系统最后的结果，直接展现给用户。重排序的原因是因为多个物品之间往往是相互影响的，而精排序是根据PointWise得分，容易造成推荐结果同质化严重，有很多冗余信息。

**常见的有三种优化目标：Point Wise、Pair Wise 和 List Wise。**

### 推荐系统技术栈

完整的一套推荐系统体系里，不仅会涉及到推荐算法工程师、后台开发工程师、数据挖掘/分析工程师、NLP/CV工程师还有前端、客户端甚至产品、运营等支持。作为算法工程师，需要掌握的技术栈主要就是**算法和工程**。

#### 算法：

**召回**：轻量快速低延迟，不需要十分准确，但不可遗漏。**粗排**：介于召回和精排之间，要同时兼顾精准性和低延迟。一般模型也不能过于复杂。**精排**：在最大时延允许的情况下，保证精确性。精排系统构建一般需要涉及样本、特征、模型三部分。**重排**：获取精排的排序结果，基于运营策略、多样性、context上下文等，进行一个微调。重排中规则比较多，但目前也有不少基于模型来提升重排效果的方案。**混排**：多个业务线都想在Feeds流中获取曝光，则需要对它们的结果进行混排。

**画像层**：算法主要体现在如何绘制一个用户画像和商品画像。用户画像是大家比较容易理解的，比如用户年龄、爱好通常APP会通过注册界面收集这些信息内容画像各家的做法也不同，当前比较主流的都会涉及到一个**多模态信息内容理解**。一般推荐系统会加入多模态的一个内容理解。

。。。。。省略

#### 工程：

- **编程语言**：Python、Java（scala）、C++、sql、shell；
- **机器学习**：Tensorflow/Pytorch、GraphLab/GraphCHI、LGB/Xgboost、SKLearn；
- **数据分析**：Pandas、Numpy、Seaborn、Spark；
- 数据存储：mysql、redis、mangodb、hive、kafka、es、hbase；
- 相似计算：annoy、faiss、kgraph
- 流计算：Spark Streaming、Flink
- 分布式：Hadoop、Spark

## 二、推荐系统算法基础

### 经典召回模型

#### 算法评估

**召回率**：在模型召回预测的物品中，预测准确的物品占用户实际喜欢的物品的比例。

**精确率**：推荐的物品中，对用户准确推荐的物品占总物品的比例。

- 如要确保召回率高，一般是推荐更多的物品，期望推荐的物品中会涵盖用户喜爱的物品。而实际中，推荐的物品中用户实际喜爱的物品占少数，推荐的精确率就会很低。故同时要确保高召回率和精确率往往是矛盾的，所以实际中需要在二者之间进行权衡。

**覆盖率**：推荐系统能够推荐出来的物品占总物品集合的比例。

- 覆盖率表示最终的推荐列表中包含多大比例的物品。如果所有物品都被给推荐给至少一个用户， 那么覆盖率是100%。

**新颖度**：用推荐列表中物品的平均流行度度量推荐结果的新颖度。 如果推荐出的物品都很热门， 说明推荐的新颖度较低。

#### 协同过滤算法

##### 基本思想

根据用户之前的喜好以及其他兴趣相近的用户的选择来给用户推荐物品。

基于对用户历史行为数据的挖掘发现用户的喜好偏向， 并预测用户可能喜好的产品进行推荐。

一般是**仅仅基于用户的行为数据**（评价、购买、下载等）, 而不依赖于项的任何附加信息（物品自身特征）或者用户的任何附加信息（年龄， 性别等）

- 基于用户的协同过滤算法（UserCF）
- 基于物品的协同过滤算法（ItemCF）

重点是**计算相似度**     实现代码见jupyter notebook   **rec_test**

##### 相似性度量方法

1. **杰卡德（Jaccard）相似系数**     适用于隐式反馈数据（0-1）

2. **余弦相似度**     衡量了两个向量的夹角，夹角越小越相似   在度量文本相似度、用户相似度、物品相似度的时候都较为常用。            from sklearn.metrics.pairwise importcosine_similarity

3. **皮尔逊相关系数**  就是概率论中的相关系数，对协方差归一化得到，范围在 −1 到 1
   
   - 相关度量的是两个变量的变化趋势是否一致，两个随机变量是不是同增同减。
   - 不适合用作计算布尔值向量（0-1）之间相关度。
   
   from  scipy.stats  import  pearsonr

##### UserCF算法（基于用户的协同过滤算法）

计算过程：

1.计算用户之间的相似度 2.计算用户对新物品的评分预测 3.对用户进行物品推荐

具体过程和实现代码可见  jupyter notebook   **rec_test**

存在的问题：

1.数据稀疏性

- 一个大型的电子商务推荐系统一般有非常多的物品，用户可能买的其中不到1%的物品，不同用户之间买的物品重叠性较低，导致算法无法找到一个用户的邻居，即偏好相似的用户。
- 这导致UserCF不适用于那些正反馈获取较困难的应用场景(如酒店预订， 大件物品购买等低频应用)。

2.算法扩展性

- 基于用户的协同过滤需要维护用户相似度矩阵以便快速的找出 TopN*T**o**pN* 相似用户， 该矩阵的存储开销非常大，存储空间随着用户数量的增加而增加。
- 故不适合用户数据量大的情况使用

##### UserCF算法（基于用户的协同过滤算法）

ItemCF算法并不利用物品的内容属性计算物品之间的相似度， 主要通过分析用户的行为记录计算物品之间的相似度， 该算法认为， 物品 A 和物品 C 具有很大的相似度是因为喜欢物品 A 的用户极可能喜欢物品 C。

和基于内容的推荐算法(Content-Based Recommendation)进行区分！

##### 协同过滤算法的问题分析

泛化能力弱：

- 即协同过滤无法将两个物品相似的信息推广到其他物品的相似性上。
- 导致的问题是**热门物品具有很强的头部效应， 容易跟大量物品产生相似， 而尾部物品由于特征向量稀疏， 导致很少被推荐**。

![图片](http://ryluo.oss-cn-chengdu.aliyuncs.com/JavaxxhHm3BAtMfsy2AV.png!thumbnail)

- 可以看出，D 是一件热门物品，其与 A、B、C 的相似度比较大。因此，推荐系统更可能将 D推荐给用过 A、B、C的用户。
- 但是，推荐系统无法找出 A,B,C之间相似性的原因是交互数据太稀疏， 缺乏相似性计算的直接数据。

所以这就是协同过滤的天然缺陷：**推荐系统头部效应明显， 处理稀疏向量的能力弱**。

为了解决这个问题， 同时增加模型的泛化能力。2006年，**矩阵分解技术(Matrix Factorization, MF**)被提出：

- 该方法在协同过滤共现矩阵的基础上， 使用**更稠密的隐向量**表示用户和物品， 挖掘用户和物品的隐含兴趣和隐含特征。
- 在一定程度上弥补协同过滤模型处理稀疏矩阵能力不足的问题。

**1.什么时候使用UserCF，什么时候使用ItemCF？为什么？**

> （1）UserCF
> 
> - 由于是基于用户相似度进行推荐， 所以具备更强的社交特性， 这样的特点非常适于**用户少， 物品多， 时效性较强的场合**。
>   - 比如新闻推荐场景， 因为新闻本身兴趣点分散， 相比用户对不同新闻的兴趣偏好， 新闻的及时性，热点性往往更加重要， 所以正好适用于发现热点，跟踪热点的趋势。
>   - 另外还具有推荐新信息的能力， 更有可能发现惊喜, 因为看的是人与人的相似性, 推出来的结果可能更有惊喜，可以发现用户潜在但自己尚未察觉的兴趣爱好。
> 
> （2）ItemCF
> 
> - 这个更适用于兴趣变化较为稳定的应用， 更接近于个性化的推荐， 适合**物品少，用户多，用户兴趣固定持久， 物品更新速度不是太快的场合**。
> - 比如推荐艺术品， 音乐， 电影。

**2.上面介绍的相似度计算方法有什么优劣之处？**

> cosine相似度计算简单方便，一般较为常用。但是，当用户的评分数据存在 bias 时，效果往往不那么好。
> 
> - 简而言之，就是不同用户评分的偏向不同。部分用户可能乐于给予好评，而部分用户习惯给予差评或者乱评分。
> - 这个时候，根据cosine 相似度计算出来的推荐结果效果会打折扣。
> 
> 举例来说明，如下图（`X,Y,Z` 表示物品，`d,e,f`表示用户）：
> 
> ![图片](http://ryluo.oss-cn-chengdu.aliyuncs.com/JavaWKvITKBhYOkfXrzs.png!thumbnail)
> 
> - 如果使用余弦相似度进行计算，用户 d 和 e 之间较为相似。但是实际上，用户 d 和 f 之间应该更加相似。只不过由于 d 倾向于打高分，e 倾向于打低分导致二者之间的余弦相似度更高。
> - 这种情况下，可以考虑使用皮尔逊相关系数计算用户之间的相似性关系。

#### FM（factor Machine，因子分解机）

**原理介绍**

https://zhuanlan.zhihu.com/p/58160982

https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html

**实战讲解**

FM用于各阶段 https://zhuanlan.zhihu.com/p/343174108

demo（py)  https://github.com/gczr/FM

工业级demo (c/c++)  https://github.com/CastellanZhang/alphaFM

#### Item2vec

先复习Word2Vec

https://datawhalechina.github.io/fun-rec/#/ch02/ch2.1/ch2.1.2/word2vec 原理、推导、核心代码

https://zhuanlan.zhihu.com/p/89020340  Skip-Gram模型和负采样

##### Skip-gram & CBWO

##### Naive Softmax & 负采样Negative Sampling

注：注意到上图，中心词词向量为v_{c}*v**c*,而上下文词词向量为u_{o}*u**o*。也就是说每个词会对应两个词向量，**在词w做中心词时，使用v_{w}\*v\**w\*作为词向量，而在它做上下文词时，使用u_{w}\*u\**w\*作为词向量**。这样做的原因是为了求导等操作时计算上的简便。当整个模型训练完成后，我们既可以使用v_{w}*v**w*作为词w的词向量，也可以使用u_{w}*u**w*作为词w的词向量，亦或是将二者平均。

Q&A:

1. P(o|c)怎么表示？
2. 为何最小化损失函数能够得到良好表示的词向量dense word vector？

回答1：我们使用**中心词c和上下文词o的相似性**来计算P(o|c)*P*(*o*∣*c*)，更具体地，相似性由**词向量的点积**表示

使用词向量的点积表示P(o|c)的原因：1.计算简单 2.出现在一起的词向量意义相关，则希望它们相似

又P(o|c)是一个概率，所以我们在**整个语料库**上使用**softmax**将点积的值映射到概率

**Item2Vec** 的原理十分十分简单，它是基于 Skip-Gram 模型的物品向量训练方法。但又存在一些区别，如下：

- 词向量的训练是基于句子序列（sequence），但是物品向量的训练是基于物品集合（set）。
- 因此，物品向量的训练丢弃了空间、时间信息。

##### Item2vec实例——Airbnb召回

业务背景：

Airbnb 是全球最大的短租平台，包含了数百万种不同的房源。

Airbnb 平台 99% 的房源预订来自于**搜索排序**和**相似房源推荐**。

Embedding:

Airbnb 描述了两种 Embedding 的构建方法，分别为：

- 用于描述短期实时性的个性化特征 Embedding：**listing Embeddings**    (listing 表示房源)
- 用于描述长期的个性化特征 Embedding：**user-type & listing type Embeddings**
- 

Listing Embeddings 是基于用户的点击 session 学习得到的，用于表示房源的短期实时性特征。

建立多用户点击session，基于 Word2Vec 的 Skip-Gram 模型来学习不同 listing 的 Embedding 表示

改进点：

1.**正负样本集构建的改进**（使用 booked listing 作为全局上下文，负样本的选择新增了与其位于同一个 market 的 listing）

2.**Listing Embedding 的冷启动**（房主提供的房源信息，为其查找3个相似的 listing，并将它们 Embedding 的均值作为新 listing 的 Embedding表示）

##### User-type & Listing-type Embedding

除了挖掘 Listing 的短期兴趣特征表示外，还对 User 和 Listing 的长期兴趣特征表示进行了探索

长期兴趣的探索是基于 booking session（如上文，用户的历史预定序列，booked listing 表示用户在 session 中最终预定的房源）

遇到的问题：

- 预定本身就是一件低频率事件。booking sessions 数据量的大小远远小于 click sessions 

- 许多用户过去只预定了单个数量的房源，无法从长度为1的 session 中学习 Embedding

- 对于任何实体，要基于 context 学习到有意义的 Embedding，该实体至少在数据中出现5-10次。但平台上大多数 listing_ids 被预定的次数低于5-10次。

- 用户连续两次预定的时间间隔可能较长，在此期间用户的行为（如价格敏感点）偏好可能会发生改变（由于职业的变化）。

  





#### 特征工程概念补充

**feature coverage**  https://datascience.stackexchange.com/questions/17121/definition-of-feature-coverage

**feature importance**





#### 双塔模型

文献：

1.**Deep Neural Networks for YouTube Recommendations**

https://zhuanlan.zhihu.com/p/52169807

https://zhuanlan.zhihu.com/p/52504407

2.**Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations**

https://zhuanlan.zhihu.com/p/365690334

##### 模型结构

![image-20221020142810579](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020143031379.png)

![image-20221020143109055](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020143109055.png)

##### 训练方法

###### Pointwise训练

![image-20221020154155139](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020154155139.png)

###### Pairwise训练

![image-20221020154641070](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020154641070.png)

![image-20221020154649335](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020154649335.png)



###### Listwise训练

（可以类别  对比学习 NCEloss）

![image-20221020154656205](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020154656205.png)

![image-20221020154700814](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020154700814.png)

##### 正负样本选择

![image-20221020154209119](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020154209119.png)





##### 线上服务，模型更新



离线存储+线上召回

离线计算物品向量并存入向量数据库，向量数据库建索引以加速最近邻查找，在线计算用户向量

原因在于物品数量过多，计算量太大；并且用户兴趣动态变化，物品特征相对稳定



![image-20221020160521296](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020160521296.png)



全量更新&增量更新



![image-20221020160812918](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020160812918.png)

![image-20221020160757399](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020160757399.png)



















### 经典排序模型

系列文章参考：https://www.zhihu.com/people/dadada-82-81/posts



#### GBDT+LR

基本实现： https://github.com/lipengyuer/DataScience/tree/master/src/algoritm



CART算法：https://www.cnblogs.com/qiu-hua/p/14851247.html

 					https://zhuanlan.zhihu.com/p/32003259

​					 CART实例：https://www.cnblogs.com/qiu-hua/p/14851247.html



GBDT讲解  回归部分： https://mp.weixin.qq.com/s/Eh_YzmBDng5ChwSs2MUjxQ

​					分类部分： https://www.jianshu.com/p/f5e5db6b29f2

​					理论细节： https://blog.csdn.net/wuzhongqiang/article/details/108471107





#### FM

既可以用于召回也可以用于排序

显示建模二阶交叉

可以优化时空复杂度



#### FNN

https://www.jianshu.com/p/c639d52c124b

![img](https://upload-images.jianshu.io/upload_images/23551183-f79550e72c599930.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

![img](https://upload-images.jianshu.io/upload_images/23551183-afb740f6bc91e34e.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)





优点：

- 引入DNN对特征进行更高阶组合，减少特征工程，能在一定程度上增强FM的学习能力。这种尝试为后续深度推荐模型的发展提供了新的思路（相比模型效果而言，个人感觉这种融合思路意义更大）。

缺点：

- 两阶段训练模式，在应用过程中不方便，且模型能力受限于FM表征能力的上限。
- FNN专注于高阶组合特征，但是却没有将低阶特征纳入模型。

仔细分析下这种两阶段训练的方式，存在几个问题：

1）FM中进行特征组合，使用的是隐向量点积。将FM得到的隐向量移植到DNN中接入全连接层，全连接本质是将输入向量的所有元素进行加权求和，且不会对特征Field进行区分，也就是说FNN中高阶特征组合使用的是全部隐向量元素相加的方式。说到底，在理解特征组合的层面上FNN与FM是存在Gap的，而这一点也正是PNN对其进行改进的动力。

2）在神经网络的调参过程中，参数学习率是很重要的。况且FNN中底层参数是通过FM预训练而来，如果在进行反向传播更新参数的时候学习率过大，很容易将FM得到的信息抹去。个人理解，FNN至少应该采用Layer-wise learning rate，底层的学习率小一点，上层可以稍微大一点，在保留FM的二阶交叉信息的同时，在DNN上层进行更高阶的组合。



```python
import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel

class FNN(BaseModel):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(FNN, self).__init__(config)
        # 稠密和稀疏特征的数量
        self.num_dense_feature = dense_features_cols.__len__()
        self.num_sparse_feature = sparse_features_cols.__len__()

        # FNN的线性部分，对应 ∑WiXi
        self.embedding_layers_1 = nn.ModuleList([
            nn.Embedding(num_embeddings=feat_dim, embedding_dim=1)
                for feat_dim in sparse_features_cols
        ])

        # FNN的Interaction部分，对应∑∑<Vi,Vj>XiXj
        self.embedding_layers_2 = nn.ModuleList([
            nn.Embedding(num_embeddings=feat_dim, embedding_dim=config['embed_dim'])
                for feat_dim in sparse_features_cols
        ])

        # FNN的DNN部分
        self.hidden_layers = [self.num_dense_feature + self.num_sparse_feature*(config['embed_dim']+1)] + config['dnn_hidden_units']
        self.dnn_layers = nn.ModuleList([
            nn.Linear(in_features=layer[0], out_features=layer[1])\
                for layer in list(zip(self.hidden_layers[:-1], self.hidden_layers[1:]))
        ])
        self.dnn_linear = nn.Linear(self.hidden_layers[-1], 1, bias=False)

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_inputs = x[:, :self.num_dense_feature], x[:, self.num_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        # 求出线性部分
        linear_logit = [self.embedding_layers_1[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        linear_logit = torch.cat(linear_logit, axis=-1)

        # 求出稀疏特征的embedding向量
        sparse_embeds = [self.embedding_layers_2[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        dnn_input = torch.cat((dense_input, linear_logit, sparse_embeds), dim=-1)

        # DNN 层
        dnn_output = dnn_input
        for dnn in self.dnn_layers:
            dnn_output = dnn(dnn_output)
            dnn_output = torch.tanh(dnn_output)
        dnn_logit = self.dnn_linear(dnn_output)

        # Final
        y_pred = torch.sigmoid(dnn_logit)

        return y_pred
```

从上述代码中不难看出

工程实现中，是先将sparse feature通过FM预训练得到的输入层转化为 linear_logit和 sparse_embeds，然后再与dense feature拼接，送入DNN中，即dense feature不参与FM运算



#### 补充—工程注意点

FM中，对于线性层和二阶交叉层有两种不同的实现方式

1.先将sparse feature 转化为one-hot编码，然后通过nn.linear转化为低维稠密向量

2.先将sparse feature 通过LabelEncoder转化为类别特征，然后通过nn.embedding直接转化为低维稠密向量



Attention：在FM的原公式中，线性部分和二阶交叉部分都乘了特征本值

![image-20221017191709980](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221017191709980.png)

而在实现中，两者往往都是直接通过nn.embedding生成的，这是因为一般不对系数特征Xi进行one-hot编码，仅作LabelEncoder转化为类别特征。	One-hot编码后该Filed也只有一个值不为0，所以其最终结果是一致的。



**摘自源码：**

 **线性部分的计算，所有特征的Input层，然后经过一个全连接层线性计算结果logits**

 **即FM线性部分的那块计算w1x1+w2x2+...wnxn + b,只不过，连续特征和离散特征这里的线性计算还不太一样**

**连续特征由于是数值，可以直接过全连接，得到线性这边的输出。  离散特征需要先embedding得到1维embedding，然后直接把这个1维的embedding相加就得到离散这边的线性输出。**











#### PNN![image-20210308142624189](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20210308142624189.png)

https://datawhalechina.github.io/fun-rec/#/ch02/ch2.2/ch2.2.2/PNN

一共分为五层，其中除了Product Layer别的layer都是比较常规的处理方法。

模型中最重要的部分就是通过Product层对embedding特征进行交叉组合，Product层主要有线性部分和非线性部分组成，分别用l_z 和 l_p来表示。



理论推导和优化分析： https://zhuanlan.zhihu.com/p/89850560





#### Wide&Deep

![image-20200910214310877](http://ryluo.oss-cn-chengdu.aliyuncs.com/Javaimage-20200910214310877.png)

Q&A：

1. 为什么Wide部分要用L1 FTRL训练？
2. 为什么Deep部分不特别考虑稀疏性的问题？
3. 在你的应用场景中，哪些特征适合放在Wide侧，哪些特征适合放在Deep侧，为什么呢？

https://zhuanlan.zhihu.com/p/142958834

https://zhuanlan.zhihu.com/p/92279796

![img](https://pic3.zhimg.com/80/v2-0bd41080df368ff3767b42bb3bd2e882_720w.webp)

优点：

- 简单有效。结构简单易于理解，效果优异。目前仍在工业界广泛使用，也证明了该模型的有效性。
- 结构新颖。使用不同于以往的线性模型与DNN串行连接的方式，而将线性模型与DNN并行连接，同时兼顾模型的Memorization与Generalization。

缺点：

- Wide侧的特征工程仍无法避免。





#### DFM

本质上DeepFM是显式的针对特征各种组合建模：一阶特征与二阶交叉特征（FM部分）、高阶特征（DNN部分），最终将低阶到高阶的所有特征以并行的方式连接到一起。之前的模型或多或少都没有这么完备，三者至少缺其一。

![image-20210225180556628](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20210225180556628.png)





Q&A:
1、**Sparse Feature中黄色和灰色节点代表什么意思**

FM分为线性部分和二阶交叉部分，黄色代表sparse feature中的非零值，灰色代表零值

![image-20221018142022312](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221018142022312.png)

2、**图例中的绿色连线、红色连线和黑色连线代表什么意思**

图例中展示了三种颜色的线条，其中绿色的箭头表示为特征的Embedding过程，即得到特征对应的Embedding vector，通常使用 vixi 来表示，而其中的隐向量 vi 则是通过模型学习得到的参数。红色箭头表示权重为1的连接，也就是说红色箭头并不是需要学习的参数。而黑色连线则表示为正常的，需要模型学习的参数 wi 。

![image-20221018131850330](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221018131850330.png)

结合公式整体可描述为，Sparse feature非零值直接计算linear部分，embedding后得到dense embedding，计算二阶交叉部分，并送入DNN隐式计算高阶交叉，最后三部分输出相加过一个sigmoid函数得到最终结果。







#### DCN

该模型主要特点在于提出Cross network，用于特征的自动化交叉编码。传统DNN对于高阶特征的提取效率并不高，Cross Network通过调整结构层数能够构造出有限阶（bounded-degree）交叉特征，对特征进行显式交叉编码，在精简模型参数的同时有效的提高了模型的表征能力。



模型结构如下，共分为4个部分，分别为 Embedding and Stacking Layer（特征预处理输入）、Cross network（自动化特征显式交叉）、Deep network（特征隐式交叉）和Combination output Layer（输出）

![img](https://pic1.zhimg.com/80/v2-c3ca2754c02b4a2753aa6c47e7134bf8_720w.webp)

从模型结构上来看，DCN是将Wide&Deep中的Wide侧替换为Cross Network，利用该部分自动交叉特征的能力，模型无需进行额外的特征工程。

**当cross layer叠加 l 层时，交叉最高阶可以达到 l+1 阶，并且包含了所有的交叉组合，这是DCN的精妙之处。**

参考链接： https://zhuanlan.zhihu.com/p/96010464



#### 





列举几种模型进行完备性对比，结果如下所示。FNN模型与PNN模型将重心放在提取高阶特征信息，PNN中Product Layer精心构建低阶交叉特征信息（小于等于2阶），但是仅作为后续DNN的输入，并未将低阶特征与高阶特征并行连接。并且FNN需要进行参数预训练，模型构建时间开销较多。Wide&Deep模型将低阶与高阶特征同时建模，但是在Wide侧通常需要更多的特征工程工作。所以，整体对比下来DeepFM的完备性更高。

![img](https://pic3.zhimg.com/80/v2-275cc5be9cc045b19cda83a3169fa3a2_720w.webp)







#### xDeepFM

2018年由中科大、北邮、微软联合推出，该模型的主要贡献在于，基于**vector-wise**的模式提出了新的**显式交叉高阶特征**的方法，并且与DCN 一样，能够构造有限阶交叉特征。虽然xDeepFM在名称上与DeepFM 相似，但其主要对比的是**DCN模型**。

**vector-wise模式：**

与vector-wise概念相对应的是bit-wise，在最开始的FM模型当中，通过特征隐向量之间的点积来表征特征之间的交叉组合。特征交叉参与运算的最小单位为向量，且同一隐向量内的元素并不会有交叉乘积，这种方式称为vector-wise。后续FM的衍生模型，尤其是**引入DNN模块后，常见的做法是，将embedding之后的特征向量拼接到一起，然后送入后续的DNN结构模拟特征交叉的过程。这种方式与vector-wise的区别在于，各特征向量concat在一起成为一个向量，抹去了不同特征向量的概念，后续模块计算时，对于同一特征向量内的元素会有交互计算的现象出现**，这种方式称为bit-wise。将常见的bit-wise方式改为vector-wise，使模型与FM思想更贴切，这也是xDeepFM的Motivation之一。



计算过程推导与性能分析： https://zhuanlan.zhihu.com/p/101073773



**CIN** 实现参考

```python
 def build(self, input_shape):
        # input_shape  [None, field_nums, embedding_dim]
        self.field_nums = input_shape[1]
        
        # CIN 的每一层大小Hk，这里加入第0层，也就是输入层H_0
        self.field_nums = [self.field_nums] + self.cin_size
        
        # 过滤器
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape = (1, self.field_nums[0] * self.field_nums[i], self.field_nums[i+1]), # 这个大小要理解
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True
            )
            for i in range(len(self.field_nums)-1)
        }
        
        super(CIN, self).build(input_shape)
        
    def call(self, inputs):
        # inputs [None, field_num, embed_dim]
        embed_dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        
        # 从embedding的维度把张量一个个的切开,这个为了后面逐通道进行卷积，算起来好算
        # 这个结果是个list， list长度是embed_dim, 每个元素维度是[None, field_nums[0], 1]  field_nums[0]即输入的特征个数
        # 即把输入的[None, field_num, embed_dim]，切成了embed_dim个[None, field_nums[0], 1]的张量
        split_X_0 = tf.split(hidden_layers_results[0], embed_dim, 2) 
        
        for idx, size in enumerate(self.cin_size):
            # 这个操作和上面是同理的，也是为了逐通道卷积的时候更加方便，分割的是当一层的输入Xk-1
            split_X_K = tf.split(hidden_layers_results[-1], embed_dim, 2)   # embed_dim个[None, field_nums[i], 1] feild_nums[i] 当前隐藏层单元数量
            
            # 外积的运算
            out_product_res_m = tf.matmul(split_X_0, split_X_K, transpose_b=True) # [embed_dim, None, field_nums[0], field_nums[i]]
            out_product_res_o = tf.reshape(out_product_res_m, shape=[embed_dim, -1, self.field_nums[0]*self.field_nums[idx]]) # 后两维合并起来
            out_product_res = tf.transpose(out_product_res_o, perm=[1, 0, 2])  # [None, dim, field_nums[0]*field_nums[i]]
            
            # 卷积运算
            # 这个理解的时候每个样本相当于1张通道为1的照片 dim为宽度， field_nums[0]*field_nums[i]为长度
            # 这时候的卷积核大小是field_nums[0]*field_nums[i]的, 这样一个卷积核的卷积操作相当于在dim上进行滑动，每一次滑动会得到一个数
            # 这样一个卷积核之后，会得到dim个数，即得到了[None, dim, 1]的张量， 这个即当前层某个神经元的输出
            # 当前层一共有field_nums[i+1]个神经元， 也就是field_nums[i+1]个卷积核，最终的这个输出维度[None, dim, field_nums[i+1]]
            cur_layer_out = tf.nn.conv1d(input=out_product_res, filters=self.cin_W['CIN_W_'+str(idx)], stride=1, padding='VALID')
            
            cur_layer_out = tf.transpose(cur_layer_out, perm=[0, 2, 1])  # [None, field_num[i+1], dim]
            
            hidden_layers_results.append(cur_layer_out)
        
        # 最后CIN的结果，要取每个中间层的输出，这里不要第0层的了
        final_result = hidden_layers_results[1:]     # 这个的维度T个[None, field_num[i], dim]  T 是CIN的网络层数
        
        # 接下来在第一维度上拼起来  
        result = tf.concat(final_result, axis=1)  # [None, H1+H2+...HT, dim]
        # 接下来， dim维度上加和，并把第三个维度1干掉
        result = tf.reduce_sum(result, axis=-1, keepdims=False)  # [None, H1+H2+..HT]
        
        return result
```



















#### NFM

Neural Factorization Machines： https://datawhalechina.github.io/fun-rec/#/ch02/ch2.2/ch2.2.3/NFM

![image-20221017194117516](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221017194117516.png)

改进的思路就是**用一个表达能力更强的函数来替代原FM中二阶隐向量内积的部分**



![image-20221017194517248](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221017194517248.png)



这个地方不是两个隐向量的内积，而是元素积，也就是这一个交叉完了之后k个维度不求和，最后会得到一个kk维向量，而FM那里内积的话最后得到一个数， 在进行两两Embedding元素积之后，对交叉特征向量取和， 得到该层的输出向量， 很显然， 输出是一个k维的向量。

注意， 之前的FM到这里其实就完事了， 上面就是输出了，而这里很大的一点改进就是加入特征池化层之后， 把二阶交互的信息合并， 且上面接了一个DNN网络， 这样就能够增强FM的表达能力了， 因为FM只能到二阶， 而这里的DNN可以进行多阶且非线性，只要FM把二阶的学习好了， DNN这块学习来会更加容易。

如果不加DNN， NFM就退化成了FM，所以改进的关键就在于加了一个这样的层，组合了一下二阶交叉的信息，然后又给了DNN进行高阶交叉的学习，成了一种“加强版”的FM。



#### AFM

动机：

AFM的全称是Attentional Factorization Machines, 从模型的名称上来看是在FM的基础上加上了注意力机制，FM是通过特征隐向量的内积来对交叉特征进行建模，从公式中可以看出所有的交叉特征都具有相同的权重也就是1，没有考虑到不同的交叉特征的重要性程度

如何让不同的交叉特征具有不同的重要性就是AFM核心的贡献，在谈论AFM交叉特征注意力之前，对于FM交叉特征部分的改进还有FFM，其是考虑到了对于不同的其他特征，某个指定特征的隐向量应该是不同的（相比于FM对于所有的特征只有一个隐向量，FFM对于一个特征有多个不同的隐向量）



![img](https://pic3.zhimg.com/v2-a72eb9a432bab47d05a654ba64767d0e_r.jpg)



其实就是一种加性注意力机制

计算过程与性能分析： https://zhuanlan.zhihu.com/p/94009156







#### AutoInt

(Automatic Feature Interaction) https://datawhalechina.github.io/fun-rec/#/ch02/ch2.2/ch2.2.2/AutoInt

动机：

1. 浅层的模型会受到交叉阶数的限制，没法完成高阶交叉
2. 深层模型的DNN在学习高阶隐性交叉的效果并不是很好， 且不具有可解释性

引入了**Transformer**， 做成了一个特征交互层

![image-20221018163656583](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221018163656583.png)



关于建模任意的高阶交互， 我们这里拿一个transformer块看下， 对于一个transformer块， 我们发现特征之间完成了一个2阶的交互过程，得到的输出里面我们还保留着1阶的原始特征。

那么再经过一个transformer块呢？ 这里面就会有2阶和1阶的交互了， 也就是会得到3阶的交互信息。而此时的输出，会保留着第一个transformer的输出信息特征。再过一个transformer块的话，就会用4阶的信息交互信息， 其实就相当于， 第n*n*个transformer里面会建模出n+1*n*+1阶交互来， 这个与CrossNet其实有异曲同工之妙的，无法是中间交互时的方式不一样。 前者是bit-wise级别的交互，而后者是vector-wise的交互。

所以， AutoInt是可以建模任意高阶特征的交互的，并且这种交互还是显性。





其他：特别注意这里的embedding层

对于第*i*个离散特征，直接第*i*个嵌入矩阵*Vi*乘one-hot向量就取出了对应位置的embedding。 当然，如果输入的时候不是个one-hot， 而是个multi-hot的形式，那么对应的embedding输出是各个embedding求平均得到的。

而对于连续值特征， 这里不进行分桶操作，而是通过id与嵌入矩阵关联，取相应的embedding。不过，最后要乘一个连续值。           ![image-20221018164440372](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221018164440372.png)





#### FiBiNET

(Feature Importance and Bilinear feature Interaction)

背景介绍：

SFNET

Bilnear Interaction



#### DIN



工业上的CTR预测数据集大致的样子：

![img](https://img-blog.csdnimg.cn/20210118190044920.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)

类别特征经过 one-hot、multi-hot 编码后：

![image-20221020133301708](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221020133301708.png)

模型架构如下：

![img](https://img-blog.csdnimg.cn/20210118220015871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)



首先， DIN模型的输入特征大致上分为了三类： Dense(连续型), Sparse(离散型), VarlenSparse(变长离散型)，也就是指的上面的历史行为数据。而不同的类型特征也就决定了后面处理的方式会不同：

- Dense型特征：由于是数值型了，这里为每个这样的特征建立Input层接收这种输入， 然后拼接起来先放着，等离散的那边处理好之后，和离散的拼接起来进DNN
- Sparse型特征，为离散型特征建立Input层接收输入，然后需要先通过embedding层转成低维稠密向量，然后拼接起来放着，等变长离散那边处理好之后， 一块拼起来进DNN， 但是这里面要注意有个特征的embedding向量还得拿出来用，就是候选商品的embedding向量，这个还得和后面的计算相关性，对历史行为序列加权。
- VarlenSparse型特征：这个一般指的用户的历史行为特征，变长数据， 首先会进行padding操作成等长， 然后建立Input层接收输入，然后通过embedding层得到各自历史行为的embedding向量， 拿着这些向量与上面的候选商品embedding向量进入AttentionPoolingLayer去对这些历史行为特征加权合并，最后得到输出。

通过上面的三种处理， 就得到了处理好的连续特征，离散特征和变长离散特征， 接下来把这三种特征拼接，进DNN网络，得到最后的输出结果即可。



Pipeline：

![DIN_aaaa](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87DIN_aaaa.png)









































## 三、评价指标

#### 精排阶段：

ctr点击率预测：AUC与logloss

定义&如何计算：https://zhuanlan.zhihu.com/p/280797054

优缺点分析：





## 四、常用数据集：

https://www.jianshu.com/p/5c88f4bd7c71

工业上的CTR预测数据集一般都是`multi-group categorial form`的形式，就是类别型特征最为常见，这种数据集一般长这样：

![img](https://img-blog.csdnimg.cn/20210118190044920.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)

这里的亮点就是框出来的那个特征，这个包含着丰富的用户兴趣信息。

对于特征编码，作者这里举了个例子：`[weekday=Friday, gender=Female, visited_cate_ids={Bag,Book}, ad_cate_id=Book]`， 这种情况我们知道一般是通过one-hot的形式对其编码， 转成系数的二值特征的形式。但是这里我们会发现一个`visted_cate_ids`， 也就是用户的历史商品列表， 对于某个用户来讲，这个值是个多值型的特征， 而且还要知道这个特征的长度不一样长，也就是用户购买的历史商品个数不一样多，这个显然。这个特征的话，我们一般是用到multi-hot编码，也就是可能不止1个1了，有哪个商品，对应位置就是1， 所以经过编码后的数据长下面这个样子：![img](https://img-blog.csdnimg.cn/20210118185933510.png)

这个就是喂入模型的数据格式了，这里还要注意一点 就是上面的特征里面没有任何的交互组合，也就是没有做特征交叉。这个交互信息交给后面的神经网络去学习。



####  Criteo 数据集

Homepage: https://labs.criteo.com/2013/12/download-terabyte-click-logs/https://labs.criteo.com/2013/12/download-terabyte-click-logs/

不同模型效果排名: https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo
