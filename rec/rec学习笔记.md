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



##### 协同过滤算法的缺点

1.冷启动问题

2.数据稀疏性问题

3.可扩展性问题

4.静态建模，受热门物品影响，易推荐同质化物品

https://blog.csdn.net/xue208212674/article/details/107789809



#### 近似最近邻查找

（Approximate  Nearest Neighbor Search）

基本思想：

划分区域，每个区域用一个向量表示，先计算每个区域向量和query的距离、相似度



ANNOY

https://zhuanlan.zhihu.com/p/454511736



#### FM

**Factor Machine，因子分解机**

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





#### **基于图的召回**

由于用户和项目的数十亿规模，传统的方法已经不能满足于实际的需求，主要的问题体现在三个方面：

- **可扩展性**：现有的推荐方法无法扩展到在拥有十亿的用户和二十亿商品的淘宝中。
- **稀疏性**：存在大量的物品与用户的交互行为稀疏。即用户的交互到多集中于以下部分商品，存在大量商品很少被用户交互。
- **冷启动**：在淘宝中，每分钟会上传很多新的商品，由于这些商品没有用户行为的信息（点击、购买等），无法进行很好的预测。



Item2vec方法有其局限性，因为只能利用序列型数据，所以Item2vec在处理互联网场景下大量的网络化数据时往往显得捉襟见肘，这

就是 Graph Embedding技术出现的动因。

![img](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20220328133138263.png)

##### **DeepWalk**

首先基于用户行为序列构建了物品关系图。可以看出，物品 A和B之间的边产生的原因是用户U1先后购买了物品A和物品B。**如果后续产生了多条相同的有向边，则有向边的权重被加强。**在图上采用随机游走的方式，产生用户行为（物品）序列，再使用Item2vec的方法进行训练得到embedding

##### ![img](https://cdn.jsdelivr.net/gh/swallown1/blogimages@main/images/image-20220418142135912.png)



随机游走的转移概率如下图，这是针对有向有权图

即DeepWalk的跳转概率就是跳转边的权重占所有相关出边权重之和的比例

![img](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20220328144516898.png)

如果物品关系图是无向无权图，那么跳转概率将是上式个特例，即权重M_ij 将为常数1，且N+（v_i）应是节点v_i所有“边”的集

合，而不是所有“出边”的集合。





##### **Node2vec**

它通过调整随机游走权重的方法使Graph Embedding的结果更倾向于体现网络的同质性（homophily）或结构性（structural equivalence）。

具体地讲，网络的“同质性”指的是距离相近节点的Embedding应尽量近似，如下图所示，节点 *u* 与其相连的节点 *s*1、*s*2、*s*3、*s*4的 Embedding 表达应该是接近的，这就是网络“同质性”的体现。

“结构性”指的是结构上相似的节点的 Embedding 应尽量近似，图中节点 *U* 和节点 *s*6都是各自局域网络的中心节点，结构上相似，其

Embedding 的表达也应该近似，这是“结构性”的体现。

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/11.3.png)

为了使 Graph Embedding 的结果能够表达网络的“结构性”，在随机游走的过程中，需要让游⾛的过程更倾向于BFS，因为 BFS 会更多地
在当前节点的邻域中游⾛遍历，相当于对当前节点周边的网络结构进行⼀次“微观扫描”。当前节点是“局部中心节点”，还是“边缘节点”，或
是“连接性节点”，其生成的序列包含的节点数量和顺序必然是不同的，从而让最终的Embedding抓取到更多结构性信息。

另外，为了表达“同质性”，需要让随机游走的过程更倾向于DFS，因为DFS更有可能通过多次跳转，游走到远方的节点上，但无论怎样，
DFS的游走更大概率会在⼀个大的集团内部进行，这就使得⼀个集团或者社区内部的节点的Embedding更为相似，从而更多地表达网络
的“同质性”。

具体权重调整算法详见《深度学习推荐系统》4.4.2节





##### EGES

参考链接：

https://zhuanlan.zhihu.com/p/64200072

https://www.jianshu.com/p/229b686535f1



**Base Graph Embedding—>Graph Embedding with Side Information—>Enhanced Graph Embedding with Side Information**

基本思想是在DeepWalk生成的Graph Embedding基础上引入补充信息。单纯使用用户行为生成的物品相关图，固然可以生成物品的Embedding，但是如果遇到新加入的物品，或者没有过多互动信息的“长尾”物品，则推荐系统将出现严重的**冷启动问题**。为了使“冷启动”的商品获得“合理”的初始Embedding，阿里巴巴团队通过引入更多**补充信息（side information）**来丰富Embedding信息的来源，从而使没有历史行为记录的商品获得较合理的初始Embedding。



生成Graph Embedding的第⼀步是生成物品关系图，通过用户行为序列可以生成物品关系图，**也可以利⽤“相同属性”“相同类别”等信息建立物品之间的边，生成基于内容的知识图谱**。而基于知识图谱生成的物品向量可以被称为补充信息 Embedding向量。当然，根据补充信息类别的不同，可以有多个补充信息Embedding向量。

如何融合⼀个物品的多个Embedding向量，使之形成物品最后的Embedding呢？最简单的方法是在深度神经网络中加入平均池化层，将不同Embedding平均起来。为了防止简单的平均池化导致有效Embedding信息的丢失，阿里巴巴在此基础上进行了加强，对每个 Embedding 加上了权重，如图所示，对每类特征对应的 Embedding 向量，分别赋予权重*a*0，*a*1，…，a*n*。图中的隐层表达（Hidden Representation层）就是对不同Embedding进行加权平均操作的层，将加权平均后的Embedding向量输入softmax层，通过梯度反向传播，求得每个Embedding的权重a_i（i*=0…*n）。

![](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20220328154950289.png)

在实际的模型中，阿⾥巴巴采用了 exp(a_j) 而不是 a_j作为相应Embedding的权重，主要原因有⼆：⼀是避免权重为0；⼆是因为exp(a_j)在梯度下降过程中有良好的数学性质





##### GraphSAGE

GraphSAGE提出的前提是因为基于直推式(transductive)学习的图卷积网络无法适应工业界的大多数业务场景。我们知道的是，基于直推式学习的图卷积网络每次学习是针对于当前图上所有的节点。然而在实际的工业场景中，图中的结构和节点都不可能是固定的，会随着时间的变化而发生改变。在这样的场景中，直推式学习的方法就需要不断的重新训练才能够为新加入的节点学习embedding，导致在实际场景中无法投入使用。

斯坦福大学提出了一种归纳(inductive)学习的GCN方法——GraphSAGE，即**通过聚合邻居信息的方式为给定的节点学习embedding**。GraphSAGE是通过学习聚合节点邻居生成节点Embedding的函数的方式，为任意节点学习embedding，进而将GCN扩展成归纳学习任务。

![img](https://cdn.jsdelivr.net/gh/swallown1/blogimages@main/images/image-20220423094435223.png)

这个公式可以非常直观的让我们理解GraphSAGE的原理。

- h_v^0表示图上节点的初始化表示，等同于节点自身的特征。
- h_v^k表示第k层卷积后的节点表示，其来源于两个部分：
  - 第一部分来源于节点v的邻居节点集合N(v），利用邻居节点的第k-1层卷积后的特征h_u^{k-1}进行 (∑*u*∈*N*(*v*)∣*N*(*v*)∣h_u^{k-1} ）后，再进行线性变换。这里**借助图上的边将邻居节点的信息通过边关系聚合到节点表示中(简称卷积操作)**。
  - 第二部分来源于节点v的第k-1成卷积后的特征h_v^{k-1}，进行线性变换。



- 总的来说图卷积的思想是**在对自身做多次非线性变换时，同时利用边关系聚合邻居节点信息。**
- 最后一次卷积结果作为节点的最终表示Z，以用于下游任务(节点分类，链路预测或节点召回)

##### 

**邻居采样**

GraphSAGE的具体采样过程是，首先根据中心节点集合B^k，对集合中每个中心节点通过随机采样的方式对其邻居节点采样固定数量S个(如果邻居节点数量大于S，采用无放回抽样；如果小于S，则采用有放回抽样)，形成的集合表示为B^{k-1}；以此类推每次都是为前一个得到的集合的每个节点随机采样S个邻居，最终得到第k层的所有需要参与计算的节点集合B^{0}。



**聚合函数**

如何对于采样到的节点集进行聚合，介绍的4种方式：Mean 聚合、Convolutional 聚合、LSTM聚合以及Pooling聚合。由于邻居节点是无序的，所以希望构造的聚合函数具有**对称性(即输出的结果不因输入排序的不同而改变)，同时拥有较强的表达能力。**

- **M**ean 聚合：首先会对邻居节点按照**element-wise**进行均值聚合，然后将当前节点k-1层得到特征与邻居节点均值聚合后的特征，**分别送入全连接网络后相加得到结果。**
- Convolutional 聚合：这是一种基于GCN聚合方式的变种，首先对邻居节点特征和自身节点特征求均值，得到的聚合特征送入到全连接网络中。与Mean不同的是，这里**只经过一个全连接层**。
- LSTM聚合：由于LSTM可以捕捉到序列信息，因此相比于Mean聚合，这种聚合方式的表达能力更强；但由于LSTM对于输入是有序的，因此该方法**不具备对称性**。作者对于无序的节点进行随机排列以调整LSTM所需的有序性。
- Pooling聚合：对于邻居节点和中心节点进行一次非线性转化，将结果进行一次基于**element-wise**的**最大池化**操作。该种方式具有**较强的表达能力**的同时还具有**对称性**。



##### PinSAGE

PinSAGE 是在GraphSAGE的基础上进行改进以适应实际的工业场景，因此除了改进卷积操作中的邻居采样策略以及聚合函数的同时还有一些工程技巧上的改进，使得在大数据场景下能更快更好的进行模型训练。



**重要性采样**

在实际场景当中，一个item可能被数以千万的用户交互过，所以不可能聚合所有邻居节点是不可行的，只可能是采样部分邻居进行信息聚合。但是如果采用GraphSAGE中随机采样的方法，由于采样的邻居有限(这里是相对于所有节点而言)，会存在一定的偏差。因此PinSAGE 在采样中考虑了更加重要的邻居节点，即卷积时只注重部分重要的邻居节点信息，已达到高效计算的同时又可以消除偏置。

PinSAGE使用重要性采样方法，即需要为每个邻居节点计算一个重要性权重，根据权重选取top-t的邻居作为聚合时的邻居集合。其中计算重要性的过程是，以目标节点为起点，进行random-walk，采样结束之后计算所有节点访问数的L1-normalized作为重要性权重，同时这个权重也会在聚合过程中加以使用(**加权聚合**)。



**聚合函数**

在实际执行过程中通过对每一层执行一次图卷积操作以得到不同阶邻居的信息，单层图卷积过程如下三步：

1. 聚合邻居： 先将所有的邻居节点经过一次非线性转化(一层DNN)，再由聚合函数(Pooling聚合) （如元素平均，**加权和**等）将所有邻居信息聚合成目标节点的embedding。这里的加权聚合采用的是通过random-walk得到的重要性权重。
2. 更新当前节点的embedding：将目标节点当前的向量 z_u与步骤1中聚合得到的邻居向量 n_u进行拼接，在通过一次非线性转化。
3. 归一化操作：对目标节点向量 z_u** 归一化。

Convolve算法的聚合方法与GraphSAGE的Pooling聚合函数相同，主要区别在于对更新得到的向量 z_u进行归一化操作，**可以使训练更稳定，以及在近似查找最近邻的应用中更有效率。**



**基于mini-batch堆叠多层图卷积**

与GraphSAGE类似，采用的是基于mini-batch 的方式进行训练。在实际的工业场景中，由于用户交互图非常庞大，无法对于所有的节点同时学习一个embedding，因此需要从原始图上寻找与 mini-batch 节点相关的子图。具体地是说，对于mini-batch内的所有节点，会通过采样的方式逐层的寻找相关邻居节点，再通过对每一层的节点做一次图卷积操作，以从k阶邻居节点聚合信息。

![img](https://cdn.jsdelivr.net/gh/swallown1/blogimages@main/images/image-20220406204431024.png)

如上图所示：对于batch内的所有节点(图上最顶层的6个节点)，依次根据权重采样，得到batch内所有节点的一阶邻居(图上第二层的所有节点)；然后对于所有一阶邻居再次进行采样，得到所有二阶邻居(图上的最后一层)。节点采样阶段完成之后，与采样的顺序相反进行聚合操作。首先对二阶邻居进行单次图卷积，将二阶节点信息聚合已更新一阶节点的向量表示(其中小方块表示的是一层非线性转化)；其次对一阶节点再次进行图卷积操作，将一阶节点的信息聚合已更新batch内所有节点的向量表示。仅此对于一个batch内的所有的样本通过卷积操作学习到一个embedding，而每一个batch的学习过程中仅**利用与mini-batch内相关节点的子图结构。**



**训练过程**

PinSage在训练时采用的是 Margin Hinge Loss 损失函数，主要的思想是最大化正例embedding之间的相关性，同时还要保证负例之间相关性相比正例之间的相关性小于某个阈值(Margin)。具体的公式如下：
$$
J_{\mathcal{G}}\left(\mathrm{z}_{q} \mathrm{z}_{i}\right)=\mathbb{E}_{n_{k} \sim P_{n}(q)} \max \left\{0, \mathrm{z}_{q} \cdot \mathrm{z}_{n_{k}}-\mathrm{z}_{q} \cdot \mathrm{z}_{i}+\Delta\right\}
$$
其中Z_q是学习得到的目标节点embedding，Z_i是与目标节点相关item的embedding，Z_{n_k}是与目标节点不相关item的embedding，Δ为margin值，具体大小需要调参。那么对于相关节点i，以及不相关节点nk。



正负样本具体都是如何定义的，这对于召回模型的训练意义重大，让我们看看具体是如何定义的。









#### 基于序列的召回

##### MIND

背景与动机：

作者基于天猫的实际场景出发的发现，每个用户每天与数百种产品互动， 而互动的产品往往来自于很多个类别，这就说明用户的兴趣极其广泛，**用一个向量是无法表示这样广泛的兴趣的**。有没有可能用多个向量来表示用户的多种兴趣呢？以往的解决方案如下：

- 协同过滤的召回方法(itemcf和usercf)是通过历史交互过的物品或隐藏因子直接表示用户兴趣， 但会遇到稀疏或计算问题

- 基于深度学习的方法用低维的embedding向量表示用户，比如YoutubeDNN召回模型，双塔模型等，都是把用户的基本信息，或者用户交互过的历史商品信息等，过一个全连接层，最后编码成一个向量，用这个向量来表示用户兴趣，但作者认为，这是多兴趣表示的瓶颈，因为需要压缩所有与用户多兴趣相关的信息到一个表示向量，所有用户多兴趣的信息进行了混合，导致这种多兴趣并无法体现，所以往往召回回来的商品并不是很准确，除非向量维度很大，但是大维度又会带来高计算。

- DIN模型在Embedding的基础上加入了Attention机制，来选择的捕捉用户兴趣的多样性，但采用Attention机制，对于每个目标物品，都需要重新计算用户表示，这在召回阶段是行不通的(海量)，所以DIN一般是用于排序。

  

  MIND为了推断出用户的多兴趣表示，提出了一个多兴趣提取层，该层使用动态路由机制自动的能将用户的历史行为聚类，然后每个类簇中产生一个表示向量，这个向量能代表用户某种特定的兴趣，而多个类簇的多个向量合起来，就能表示用户广泛的兴趣。

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/33b251f8dcb242ad82b2ed0313f6df73.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_2,color_FFFFFF,t_70,g_se,x_16#pic_center)

  
  
  核心是多兴趣提取层(Multi-interest extractor layer)， 而这里面重点是动态路由与胶囊网络。
  
  

**胶囊网络**

胶囊网络其实可以和神经网络对比着看可能更好理解。神经网络的每一层的神经元输出的是单个的标量值，接收的输入，也是多个标量值，所以这是一种value to value的形式，而胶囊网络每一层的胶囊输出的是一个向量值，接收的输入也是多个向量，所以它是vector to vector形式的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1f698efd1f7e4b76babb061e52133e45.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_2,color_FFFFFF,t_70,g_se,x_16#pic_center)

有两点需要注意：

1. 这里的W^i参数是可学习的，和神经网络一样， 通过BP算法更新
2. 这里的c_i参数不是BP算法学习出来的，而是采用动态路由机制现场算出来的，这个非常类似于pooling层，我们知道pooling层的参数也不是学习的，而是根据前面的输入现场取最大或者平均计算得到的



**动态路由机制原理**

![在这里插入图片描述](https://img-blog.csdnimg.cn/82746b6ff8ac47fab6a89788d8d50f9e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_1,color_FFFFFF,t_70,g_se,x_16#pic_center)



下面是上述过程的展开计算过程， 这个和RNN的计算有点类似：

![在这里插入图片描述](https://img-blog.csdnimg.cn/c189e1258de64e42b576884844e718a4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_1,color_FFFFFF,t_70,g_se,x_16#pic_center)









**Multi-Interest Extractor Layer**



![在这里插入图片描述](https://img-blog.csdnimg.cn/02fd2e79c97c4345bb228b3bb2eb517c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_2,color_FFFFFF,t_70,g_se,x_16#pic_center)
$$
\begin{array}{c}
w_{i j}=\frac{\exp b_{i j}}{\sum_{k=1}^{m} \exp b_{i k}} \\
\vec{z}_{j}^{h}=\sum_{i=1}^{m} w_{i j} S_{i j} \vec{c}_{i}^{l} \\
\vec{c}_{j}^{h}=\operatorname{squash}\left(\vec{z}_{j}^{h}\right)=\frac{\left\|\vec{z}_{j}^{h}\right\|^{2}}{1+\left\|\vec{z}_{j}^{h}\right\|^{2}} \frac{\vec{z}_{j}^{h}}{\left\|\vec{z}_{j}^{h}\right\|} \\
b_{i j}=\left(\vec{c}_{j}^{h}\right)^{T} \mathrm{~S}_{i j} \vec{c}_{i}^{l}
\end{array}
$$


**B2I动态路由**

1. **共享双向映射矩阵**。在初始动态路由中，使用固定的或者说共享的双线性映射矩阵S而不是单独的双线性映射矩阵， 在原始的动态路由中，对于每个输出胶囊都会有对应的S，而这里是每个输出胶囊，都共用一个S矩阵。 原因有两个：

   1. 用户行为是可变长度的，从几十个到几百个不等，因此使用共享的双线性映射矩阵是有利于泛化。
   2. 希望兴趣胶囊在同一个向量空间中，但不同的双线性映射矩阵将兴趣胶囊映射到不同的向量空间中。因为映射矩阵的作用就是对用户的行为胶囊进行线性映射， 由于用户的行为序列都是商品，所以希望经过映射之后，到统一的商品向量空间中去。

2. **随机初始化路由对数**。

   由于利用共享双向映射矩阵S，如果再初始化路由对数为0将导致相同的初始的兴趣胶囊。
   $$
   \begin{array}{c}w_{i j}=\frac{\exp b_{i j}}{\sum_{k=1}^{m} \exp b_{i k}} \\\vec{z}_{j}^{h}=\sum_{i=1}^{m} w_{i j} S_{i j} \vec{c}_{i}^{l} \\\end{array}
   $$
   随后的迭代将陷入到一个不同兴趣胶囊在所有的时间保持相同的情景。因为每个输出胶囊的运算都一样了。为了减轻这种现象，作者通过高斯分布进行随机采样来初始化路由对数，让初始兴趣胶囊与其他每一个不同，其实就是希望在计算每个输出胶囊的时候，通过随机化的方式，希望这几个聚类中心离得远一点，这样才能表示出广泛的用户兴趣。

3. **动态的兴趣数量**，兴趣数量就是聚类中心的个数，由于不同用户的历史行为序列不同，那么相应的，其兴趣胶囊有可能也不一样多，所以这里使用了一种启发式方式自适应调整聚类中心的数量，即K值。

$$
K_{u}^{\prime}=\max \left(1, \min \left(K, \log _{2}\left(\left|\mathcal{I}_{u}\right|\right)\right)\right)
$$



**Label-aware Attention Layer**

通过多兴趣提取器层，从用户的行为embedding中生成多个兴趣胶囊。不同的兴趣胶囊代表用户兴趣的不同方面，相应的兴趣胶囊用于评估用户对特定类别的偏好。所以，在训练的期间，最后需要设置一个Label-aware的注意力层，对于当前的商品，根据相关性选择最相关的兴趣胶囊。这里其实就是一个普通的注意力机制，和DIN里面的那个注意力层基本上是一模一样，计算公式如下：
$$
\begin{aligned}
\overrightarrow{\boldsymbol{v}}_{u} &=\operatorname{Attention}\left(\overrightarrow{\boldsymbol{e}}_{i}, \mathrm{~V}_{u}, \mathrm{~V}_{u}\right) \\
&=\mathrm{V}_{u} \operatorname{softmax}\left(\operatorname{pow}\left(\mathrm{V}_{u}^{\mathrm{T}} \vec{e}_{i}, p\right)\right)
\end{aligned}
$$
训练结束后，抛开label-aware注意力层，MIND网络得到一个用户表示映射函数f_{user}。在服务期间，用户的历史序列与自身属性喂入到 f_{user}，每个用户得到多兴趣向量。然后这个表示向量通过一个近似邻近方法来检索top N物品。









##### SDM

**SDM**(Sequential Deep Matching Model) 是以会话为单位，对长序列进行切分。依据是用户在同一个Session下，其需求往往很明确，交互的商品也往往都非常类似。 但是Session与Session之间，商品类型可能骤变。 （DSIN也是这个思路）

**背景与动机**

- 动机： 召回模型需要捕获用户的动态兴趣变化，这个过程中利用好用户的长期行为和短期偏好非常关键，而以往的模型有下面几点不足：

  - 协同过滤模型： 基于用户的交互进行**静态建模**，无法感知用户的兴趣变化过程，易召回同质性的商品

  - 早期的一些序列推荐模型: 要么是对整个长序列直接建模，但这样太暴力，没法很好的学习商品之间的序列信息，有些是**把长序列分成会话**，但忽视了**一个会话中用户的多重兴趣**

  - 有些方法在考虑用户的长期行为方面，只是简单的拼接或者加权求和，而实际上**用户长期行为中只有很少一小部分对当前的预测有用**，这样暴力融合反而会适得其反，起不到效果。另外还有一些多任务或者对抗方法， 在工业场景中不适用等。

    

- 亮点:

  - SDM模型， 考虑了用户的短期行为和长期兴趣，以会话的形式进行分割，并对这两方面分别建模
  - 短期会话由于对当前决策影响比较大，那就学习的全面一点， 首先RNN学习序列关系，其次通过多头注意力机制捕捉多兴趣，然后通过一个Attention Net加权得到短期兴趣表示
  - 长期会话通过Attention Net融合，然后过DNN，得到用户的长期表示
  - 设计了一个门控机制，类似于LSTM的那种门控，能巧妙的融合这两种兴趣，得到用户最终的表示向量

![在这里插入图片描述](https://img-blog.csdnimg.cn/d297bf36d8c54b349dc666259b891927.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_2,color_FFFFFF,t_70,g_se,x_16#pic_center)









**Input Embedding with side Information**
$$
所以，假设用户的短期行为是  \mathcal{S}^{u}=\left[i_{1}^{u}, \ldots, i_{t}^{u}, \ldots, i_{m}^{u}\right] ,\\

这里面的每个商品i_{t}^{u}  其实有5个属性表示了，\\每个属性本质是ID，但转成embedding之 后，就得到了5个embedding，\\
所以这里就涉及到了融合问题。这里用  e_{i_{t}^{u}} \in \mathbb{R}^{d \times 1}  来表示每个  i_{t}^{u}  ，但这里不是embedding的pooling操作，\\而是 Concat

\boldsymbol{e}_{i_{t}^{u}}=\operatorname{concat}\left(\left\{\boldsymbol{e}_{i}^{f} \mid f \in \mathcal{F}\right\}\right)\\

其中，  \boldsymbol{e}_{i}^{f}=\boldsymbol{W}^{f} \boldsymbol{x}_{i}^{f} \in \mathbb{R}^{d_{f} \times 1}  ， \\

这个公式看着复杂，其实就是每个side info的id过embedding layer得到各自的embedding。\\
这里embedding的维 度是  d_{f}  ，等拼接起来之后，就是  d  维了。这个点要注意。\\

另外就是用户的base表示向量了，这个很简单，就是用户的基础画像，\\得到embedding，直接也是Concat，

\boldsymbol{e}_{u}=\operatorname{concat}\left(\left\{\boldsymbol{e}_{u}^{p} \mid p \in \mathcal{P}\right\}\right)\\
 e_{u}^{p}  是特征  p  的embedding。
$$
**短期用户行为建模**



先过LSTM对行为序列关系建模
$$
\begin{aligned}
\boldsymbol{i} \boldsymbol{n}_{t}^{u} &=\sigma\left(\boldsymbol{W}_{i n}^{1} \boldsymbol{e}_{i t}^{u}+\boldsymbol{W}_{i n}^{2} \boldsymbol{h}_{t-1}^{u}+b_{i n}\right) \\
f_{t}^{u} &=\sigma\left(\boldsymbol{W}_{f}^{1} \boldsymbol{e}_{i t}^{u}+\boldsymbol{W}_{f}^{2} \boldsymbol{h}_{t-1}^{u}+b_{f}\right) \\
\boldsymbol{o}_{t}^{u} &=\sigma\left(\boldsymbol{W}_{o}^{1} \boldsymbol{e}_{i}^{u}+\boldsymbol{W}_{o}^{2} \boldsymbol{h}_{t-1}^{u}+b_{o}\right) \\
\boldsymbol{c}_{t}^{u} &=\boldsymbol{f}_{t} \boldsymbol{c}_{t-1}^{u}+\boldsymbol{i} \boldsymbol{n}_{t}^{u} \tanh \left(\boldsymbol{W}_{c}^{1} \boldsymbol{e}_{i_{t}^{u}}+\boldsymbol{W}_{c}^{2} \boldsymbol{h}_{t-1}^{u}+b_{c}\right) \\
\boldsymbol{h}_{t}^{u} &=\boldsymbol{o}_{t}^{u} \tanh \left(\boldsymbol{c}_{t}^{u}\right)
\end{aligned}
$$
得到X^u，再过多头注意力机制，学习短期用户行为的多兴趣。设有h个head，则每个头中的向量维度是d/h
$$
\hat{X}^{u}=\operatorname{MultiHead}\left(X^{u}\right)=W^{O} \text { concat }\left(\text { head }_{1}^{u}, \ldots, \text { head }{ }_{h}^{u}\right)
$$
接下来再过一个User Attention， 因为作者发现，对于相似历史行为的不同用户，其兴趣偏好也不太一样。 所以加入这个用户Attention层，想挖掘更细粒度的用户个性化信息。加权求和得最终向量，维度是d。
$$
\begin{aligned}
\alpha_{k} &=\frac{\exp \left(\hat{\boldsymbol{h}}_{k}^{u T} \boldsymbol{e}_{u}\right)}{\sum_{k=1}^{t} \exp \left(\hat{\boldsymbol{h}}_{k}^{u T} \boldsymbol{e}_{u}\right)} \\
\boldsymbol{s}_{t}^{u} &=\sum_{k=1}^{t} \alpha_{k} \hat{\boldsymbol{h}}_{k}^{u}
\end{aligned}
$$
**长期用户行为建模**

长期来看，用户多种维度积累的兴趣会对当下行为产生影响，因此此部分旨在捕捉用户的长期兴趣，分为两个部分，**用户注意力层与特征拼接**

长期行为这里，是从特征的维度进行聚合，把用户的历史长序列分成了多个特征，比如用户历史点击过的商品，历史逛过的店铺，历史看过的商品的类别，品牌等，分成了多个特征子集，然后这每个特征子集里面有对应的id，比如商品有商品id, 店铺有店铺id等，对于每个子集，过user Attention layer，和用户的base向量求Attention， 相当于看看用户喜欢逛啥样的商店， 喜欢啥样的品牌，啥样的商品类别等等，得到每个子集最终的表示向量。
$$
\begin{aligned}
\alpha_{k} &=\frac{\exp \left(\boldsymbol{g}_{k}^{u T} \boldsymbol{e}_{u}\right)}{\sum_{k=1}^{\left|\mathcal{L}_{f}^{u}\right|} \exp \left(\boldsymbol{g}_{k}^{u T} \boldsymbol{e}_{u}\right)} \\
z_{f}^{u} &=\sum_{k=1}^{\left|\mathcal{L}_{f}^{u}\right|} \alpha_{k} \boldsymbol{g}_{k}^{u}
\end{aligned}
$$

$$
\begin{array}{l}
然后对这些子集表示向量进行拼接，过一个全连接层输出最终d维长期行为表示向量\\
z^{u}=\operatorname{concat}\left(\left\{z_{f}^{u} \mid f \in \mathcal{F}\right\}\right) \\
\boldsymbol{p}^{u}=\tanh \left(\boldsymbol{W}^{p} z^{u}+b\right)
\end{array}
$$





**长短期融合**

长短期兴趣融合这里，作者发现之前模型往往喜欢直接拼接起来，或者加和，注意力加权等，但作者认为这样不能很好的将两类兴趣融合起来，因为长期序列里面，其实只有很少的一部分行为和当前有关。那么这样的话，直接无脑融合是有问题的。所以这里作者用了一种较为巧妙的方式，即门控机制：
$$
\begin{array}{c}
G_{t}^{u}=\underset{o_{t}^{u}}{\operatorname{sigmoid}}\left(\boldsymbol{W}^{1} \boldsymbol{e}_{u}+\boldsymbol{W}^{2} s_{t}^{u}+\boldsymbol{W}^{3} \boldsymbol{p}^{u}+b\right) \\
\left(1-G_{t}^{u}\right) \odot p^{u}+G_{t}^{u} \odot s_{t}^{u}
\end{array}
$$
这个和LSTM的这种门控机制很像，首先门控接收的输入有用户画像e_u，用户短期兴趣s_t^u，用户长期兴趣p^u。经过sigmoid函数得到了G_{t}^{u} ，用来决定在t时刻短期和长期兴趣的贡献程度。然后根据这个贡献程度对短期和长期偏好加权进行融合。

为啥这东西就有用了呢？ 一种理解是，我们知道最终得到的短期或者长期兴趣都是d维的向量， 每一个维度可能代表着不同的兴趣偏好，比如第一维度代表品牌，第二个维度代表类别，第三个维度代表价格，第四个维度代表商店等。当然假真实的向量不可解释。

那么如果我们是直接相加或者是加权相加，其实都意味着长短期兴趣这每个维度都有很高的保留， 但其实上，万一长期兴趣和短期兴趣维度冲突了呢？ 比如短期兴趣里面可能用户喜欢这个品牌，长期用户里面用户喜欢那个品牌，那么听谁的？ 你可能说短期兴趣这个占更大权重呗，那么普通加权可是所有向量都加的相同的权重，品牌这个维度听短期兴趣的，其他维度比如价格，商店也都听短期兴趣的？

而门控机制的巧妙就在于，我会给每个维度都学习到一个权重，而这个权重非0即1(近似)， 那么接下来融合的时候，通过这个门控机制，取长期和短期兴趣向量每个维度上的其中一个。比如在品牌方面听谁的，类别方面听谁的，价格方面听谁的，只会听短期和长期兴趣的其中一个的。这样就不会有冲突发生，而至于具体听谁的，交给网络自己学习。这样就使得用户长期兴趣和短期兴趣融合的时候，每个维度上的信息保留变得有选择。使得兴趣的融合方式更加的灵活。

**这又给我们提供了一种两个向量融合的新思路，不一定非得加权or拼接or相加，还可通过门控机制让网络自己学。**





**简易实现：**

注意长期和短期会话对side info不同的处理方式

```python
def SDM(user_feature_columns, item_feature_columns, history_feature_list, num_sampled=5, units=32, rnn_layers=2,
        dropout_rate=0.2, rnn_num_res=1, num_head=4, l2_reg_embedding=1e-6, dnn_activation='tanh', seed=1024):
    """
    :param rnn_num_res: rnn的残差层个数 
    :param history_feature_list: short和long sequence field
    """
    # item_feature目前只支持doc_id， 再加别的就不行了，其实这里可以改造下
    if (len(item_feature_columns)) > 1: 
        raise ValueError("SDM only support 1 item feature like doc_id")
    
    # 获取item_feature的一些属性
    item_feature_column = item_feature_columns[0]
    item_feature_name = item_feature_column.name
    item_vocabulary_size = item_feature_column.vocabulary_size
    
    # 为用户特征创建Input层
    user_input_layer_dict = build_input_layers(user_feature_columns)
    item_input_layer_dict = build_input_layers(item_feature_columns)
    
    # 将Input层转化成列表的形式作为model的输入
    user_input_layers = list(user_input_layer_dict.values())
    item_input_layers = list(item_input_layer_dict.values())
    
    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    if len(dense_feature_columns) != 0:
        raise ValueError("SDM dont support dense feature")  # 目前不支持Dense feature
    varlen_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []
    
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(user_feature_columns+item_feature_columns)
    
    # 拿到短期会话和长期会话列 之前的命名规则在这里起作用
    sparse_varlen_feature_columns = []
    prefer_history_columns = []
    short_history_columns = []
    
    prefer_fc_names = list(map(lambda x: "prefer_" + x, history_feature_list))
    short_fc_names = list(map(lambda x: "short_" + x, history_feature_list))
    
    for fc in varlen_feature_columns:
        if fc.name in prefer_fc_names:
            prefer_history_columns.append(fc)
        elif fc.name in short_fc_names:
            short_history_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    
    # 获取用户的长期行为序列列表 L^u 
    # [<tf.Tensor 'emb_prefer_doc_id_2/Identity:0' shape=(None, 50, 32) dtype=float32>, <tf.Tensor 'emb_prefer_cat1_2/Identity:0' shape=(None, 50, 32) dtype=float32>, <tf.Tensor 'emb_prefer_cat2_2/Identity:0' shape=(None, 50, 32) dtype=float32>]
    prefer_emb_list = embedding_lookup(prefer_fc_names, user_input_layer_dict, embedding_layer_dict)
    # 获取用户的短期序列列表 S^u
    # [<tf.Tensor 'emb_short_doc_id_2/Identity:0' shape=(None, 5, 32) dtype=float32>, <tf.Tensor 'emb_short_cat1_2/Identity:0' shape=(None, 5, 32) dtype=float32>, <tf.Tensor 'emb_short_cat2_2/Identity:0' shape=(None, 5, 32) dtype=float32>]
    short_emb_list = embedding_lookup(short_fc_names, user_input_layer_dict, embedding_layer_dict)
    
    # 用户离散特征的输入层与embedding层拼接 e^u
    user_emb_list = embedding_lookup([col.name for col in sparse_feature_columns], user_input_layer_dict, embedding_layer_dict)
    user_emb = concat_func(user_emb_list)
    user_emb_output = Dense(units, activation=dnn_activation, name='user_emb_output')(user_emb)  # (None, 1, 32)
    
    # 长期序列行为编码
    # 过AttentionSequencePoolingLayer --> Concat --> DNN
    prefer_sess_length = user_input_layer_dict['prefer_sess_length']
    prefer_att_outputs = []
    # 遍历长期行为序列
    for i, prefer_emb in enumerate(prefer_emb_list):
        prefer_attention_output = AttentionSequencePoolingLayer(dropout_rate=0)([user_emb_output, prefer_emb, prefer_sess_length])
        prefer_att_outputs.append(prefer_attention_output)
    prefer_att_concat = concat_func(prefer_att_outputs)   # (None, 1, 64) <== Concat(item_embedding，cat1_embedding,cat2_embedding)
    prefer_output = Dense(units, activation=dnn_activation, name='prefer_output')(prefer_att_concat)
    # print(prefer_output.shape)   # (None, 1, 32)
    
    # 短期行为序列编码
    short_sess_length = user_input_layer_dict['short_sess_length']
    short_emb_concat = concat_func(short_emb_list)   # (None, 5, 64)   这里注意下， 对于短期序列，描述item的side info信息进行了拼接
    short_emb_input = Dense(units, activation=dnn_activation, name='short_emb_input')(short_emb_concat)  # (None, 5, 32)
    # 过rnn 这里的return_sequence=True， 每个时间步都需要输出h
    short_rnn_output = DynamicMultiRNN(num_units=units, return_sequence=True, num_layers=rnn_layers, 
                                       num_residual_layers=rnn_num_res,   # 这里竟然能用到残差
                                       dropout_rate=dropout_rate)([short_emb_input, short_sess_length])
    # print(short_rnn_output) # (None, 5, 32)
    # 过MultiHeadAttention  # (None, 5, 32)
    short_att_output = MultiHeadAttention(num_units=units, head_num=num_head, dropout_rate=dropout_rate)([short_rnn_output, short_sess_length]) # (None, 5, 64)
    # user_attention # (None, 1, 32)
    short_output = UserAttention(num_units=units, activation=dnn_activation, use_res=True, dropout_rate=dropout_rate)([user_emb_output, short_att_output, short_sess_length])
    
    # 门控融合
    gated_input = concat_func([prefer_output, short_output, user_emb_output])
    gate = Dense(units, activation='sigmoid')(gated_input)   # (None, 1, 32)
    
    # temp = tf.multiply(gate, short_output) + tf.multiply(1-gate, prefer_output)  感觉这俩一样？
    gated_output = Lambda(lambda x: tf.multiply(x[0], x[1]) + tf.multiply(1-x[0], x[2]))([gate, short_output, prefer_output])  # [None, 1,32]
    gated_output_reshape = Lambda(lambda x: tf.squeeze(x, 1))(gated_output)  # (None, 32)  这个维度必须要和docembedding层的维度一样，否则后面没法sortmax_loss
    
    # 接下来
    item_embedding_matrix = embedding_layer_dict[item_feature_name]  # 获取doc_id的embedding层
    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_input_layer_dict[item_feature_name]) # 所有doc_id的索引
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))  # 拿到所有item的embedding
    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])  # 这里依然是当可能不止item_id，或许还有brand_id, cat_id等，需要池化
    
    # 这里传入的是整个doc_id的embedding， user_embedding, 以及用户点击的doc_id，然后去进行负采样计算损失操作
    output = SampledSoftmaxLayer(num_sampled)([pooling_item_embedding_weight, gated_output_reshape, item_input_layer_dict[item_feature_name]])
    
    model = Model(inputs=user_input_layers+item_input_layers, outputs=output)
    
    # 下面是等模型训练完了之后，获取用户和item的embedding
    model.__setattr__("user_input", user_input_layers)
    model.__setattr__("user_embedding", gated_output_reshape)  # 用户embedding是取得门控融合的用户向量
    model.__setattr__("item_input", item_input_layers)
    # item_embedding取得pooling_item_embedding_weight, 这个会发现是负采样操作训练的那个embedding矩阵
    model.__setattr__("item_embedding", get_item_embedding(pooling_item_embedding_weight, item_input_layer_dict[item_feature_name]))
    return model
```

**总结：**

借鉴的地方首先是多头注意力机制也能学习到用户的多兴趣， 这样对于多兴趣，就有了胶囊网络与多头注意力机制两种思路。 而对于两个向量融合，这里又给我们提供了一种门控融合机制。





#### 基于树模型的召回

**TDM**

参考链接：https://zhuanlan.zhihu.com/p/93201318

通常大规模搜索、广告、推荐系统的召回模块包括以下几个部分：索引（用于高效检索）、评分规则（给出用户对于商品的偏好程度）、检索算法（根据评分规则，利用索引筛选出合适的商品集）。TDM的整体模型可以分为**树形索引结构**和**深层排序结构**两部分

如下图所示：

![img](https://pic4.zhimg.com/80/v2-c7a1714f03e5a0f2dbd94d78db1e54f7_1440w.webp)



其中索引的构建方式就是绿色方框中的树形结构，评分规则就是红框中的复杂DNN网络（用于输出用户对树节点的偏好程度，其中叶子节点表征每一个商品，非叶子节点是对商品的一种抽象化表征，可以不具有具体的物理意义），检索算法是**beam search**算法。

基于上述系统架构，整个召回过程可以概括为如下几个部分：

1、文中指出绿色框中树形结构的叶子节点表征的是具体的每一个商品，具有明确的物理意义（如叶子节点8代表粗跟高跟鞋，叶子节点9代表细跟高跟鞋），而非叶子节点则是对商品的进一步抽象化表征，是一种更粗粒度的表征（如节点4可能代表的是高跟鞋，当然节点4可能也不具备明确的屋里意义），总之**父节点相较于子节点来说是一种更粗粒度的表征**。



2、在这种树形索引结构的基础上如何保证检索的高效性呢，文中指出为高效的检索出Top-K的商品，该树形结构实际上是一种类似于最大堆的树结构，并且对于用户u来说，对l层非叶子节点n的偏好概率表征如下式：

![img](https://pic1.zhimg.com/80/v2-8ebbfb6037db2cedc564ee499b378548_1440w.webp)

用户-商品偏好概率（兴趣建模）：其中p(n|u)表示的就是用户u对j层节点n感兴趣的概率，α(j)是归一化因子，从上式可以得出，用户u对j层节点n感兴趣的概率，等于用户对该节点的子节点的偏好概率的最大值。所以我们如果最终需要检索出Top-K个商品，只需要**自顶向下**的在每一层检索出当前层的Top-K节点，但是当前层的检索集是上一层Top-K节点的子节点，从这些子节点中检索出当前层的Top-K节点。

具体举例来说，如图中绿色框的树形结构，最终要检索Top-2商品，那么从树的第二层开始，a）我们选取第二层的Top-K节点(2和3)；b)在第三层的时候，根据上述公式，我们可以知道该层的Top-2的一定位于节点2和节点3的子节点中，所以我们只需要从4、5、6、7节点中检索Top-2，假设检索结果是节点5和6；c)在第四层中我们检索该层的Top-2，根据公式可以知道该层的Top-2节点一定存在于节点5和6的叶子节点中，所以只需要从叶子节点10、11、12、13中检索出最终的Top-2节点，上述检索过程即为Beam Search方式。



3、在了解TDM是如何实现检索过程之后，还需要解决的问题就是如何对每层选取Top-K节点，具体做法就如上图中的红色框的部分，该部分的输入包括用户的历史行为特征以及节点的Embedding特征，在对每一层选取Top-K的时候，需要将这一层的每一个节点输入左侧模型中得到相应的预测分数，最终根据分数来取Top，文中指出TDM可以任意复杂的排序模型也得益于这种系统架构。当然**构建模型训练所需的训练样本的过程中涉及到对负样本的采样操作**，具体可以参考论文中的表述。



4、训练过程

利用采样得到训练样本之后，相应的损失函数如下所示：
$$
\begin{array}{l}
-\sum_{u} \sum_{n \in \mathcal{Y}_{u}^{+} \cup \mathcal{Y}_{u}^{-}}\\
y_{u}(n) \log P\left(\hat{y}_{u}(n)=1 \mid n u\right)+\left(1-y_{u}(n)\right) \log P\left(\hat{y}_{u}(\tilde{n})=(\log u)\right.
\end{array}
$$
在给定训练样本和损失函数的基础上，接下来需要做的就是进行模型的训练，整个系统包括树形索引结构和深层排序结构两部分，文中采用的是**联合训练**的方式，整体联合训练的方式如下：a)初始化一棵树然后基于该树训练深层模型直到其收敛；b)基于训练好的深层模型（包括叶子节点的embedding部分），利用节点的Embedding重新构建一颗新的树；c)基于新的树结构，重新训练深层模型。

具体的，在初始化树结构的时候，首先借助商品的类别信息进行排序，将相同类别的商品放到一起，然后递归的将同类别中的商品等量的分到两个子类中，直到集合中只包含一项，利用**这种自顶向下的方式构建树**。**基于该树采样生成深度模型训练所需的样本，然后进一步训练模型**，训练结束之后可以得到每个树节点对应的Embedding向量，利用节点的Embedding向量，**采用K-Means聚类方法来重新构建一颗树**，最后基于这颗新生成的树，重新训练深层网络。

![image-20210308142624189](http://ryluo.oss-cn-chengdu.aliyuncs.com/%E5%9B%BE%E7%89%87image-20220420220831318.png)



5、线上服务架构

![img](https://pic2.zhimg.com/80/v2-d7c44a76dc571a1c6125094b9cf813e9_1440w.webp)

当线上系统收到用户发送的请求时，feature server主要得到用户的相关行为特征，并将其发送给user target server，后者则利用线上部署的树形索引结构和深层网络完成用户对节点的偏好计算以及检索Top-K商品的过程。









### 经典排序模型

系列文章参考：https://www.zhihu.com/people/dadada-82-81/posts



#### 排序模型的特征

##### 有哪些特征

**1.用户画像**

用户ID（在召回、排序中做embedding)

人口统计学属性（性别、年龄）

账号信息（新老、活跃度.........）

感兴趣的类目、关键词、品牌）

**2.物品画像**

物品ID（在召回、排序中做embedding)

发布时间

GeoHash（经纬度编码）、所在城市

标题、类目、关键词、品牌.......

字数、图片数、标配数.........

**3.用户统计特征**

用户最近30天（7天、1天、1小时）的曝光数、点击数、点赞数、收藏数.......

按照类目分桶（例如最近30天，用户对美食的点击率、对科技数码的点击率）

**4.物品统计特征**

物品最近30天（7天、1天、1小时）的曝光数、点击数、点赞数、收藏数......

按照用户性别分桶、按照用户年龄分桶

作者特征：发布数、粉丝数.....

**5.场景特征（context)**

用户定位、城市

当前时刻（分段，做embedding)

是否是周末、节假日

设备信息（手机品牌、型号、操作系统）



##### **特征处理**

离散特征：做embedding。用户ID、笔记ID、作者ID、类目、关键词、城市、手机品牌。

连续特征：做分桶，变成离散特征、年龄、笔记字数、视频长度。

连续特征：其他变换、
曝光数、点击数、点赞数等数值做log（1+x）  因为都是长尾分布
转化为点击率、点赞率等值，并做平滑。



特征覆盖率

很多特征无法覆盖100%样本。
例：很多用户不填年龄，因此用户年龄特征的覆盖率远小于100%。
例：很多用户设置隐私权限，APP不能获得用户地理定位，因此场景特征有缺失。

提高特征覆盖率，可以让精排模型更准



##### 数据服务



![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/11.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/12.png)





#### 粗排

粗排：给几千篇笔记打分、单次推理代价必须小、预估的准确性不高。

精排：给几百篇笔记打分、单次推理代价很大。



前期融合：先对所有特征做concatenation，再输入神经网络。
线上推理代价大：如果有n篇候选笔记，整个大模型要做n次推理。预估准确性好

后期融合：把用户、物品特征分别输入不同的神经网络，不对用户、物品特征做融合。线上计算量小：用户塔只需要做一次线上推理，计算用户表征a。物品表征b事先储存在向量数据库中，物品塔在线上不做推理。预估准确性不如精排模型。eg:双塔模型

**因此前期融合适合做精排，后期融合适合做召回**



三塔模型，精度和速度介于两者之间

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/13.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/14.png)

有n个物品，模型上层需要做n次推理
粗排推理的大部分计算量在模型上层



从多个数据源取特征：
1个用户的画像、统计特征。
n个物品的画像、统计特征。

用户塔：只做1次推理。
物品塔：未命中缓存时需要做推理
交叉塔：必须做n次推理
上层网络做n次推理，给n个物品打分。





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





#### LHUC (PPNet)

PPNet的网络结构由基础的DNN结构和Gate NN结构组层。两种结构的融合方式是LHUC（Learning Hidden Unit Contributions）结构。



![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/5.png)



![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/10.png)







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

动机介绍：

第一是大部分模型没有考虑特征重要性，也就是交互之后，没考虑对于预测目标来讲谁更重要，一视同仁。

 第二是目前的两两特征交互，大部分依然是内积或者哈达玛积， 作者认为还不是细粒度(fine-grained way)交互。





背景介绍：

SFNET: 仍然是一种加权平均，压缩重建的过程类似于自编码器AE

![image-20210308142624189](https://img-blog.csdnimg.cn/20210703161807139.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)

Bilnear Interaction：一种特征交互方式，可视作内积与哈达玛积的结合

**双线性操作同时可以考虑交互的向量各自的各个维度上的重要性信息， 这应该是作者所说的细粒度，各个维度上的重要性**

![image-20210308142624189](https://img-blog.csdnimg.cn/20210703165031369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)

有三种类型的双线性交互方式： 

Field-All Type：所有的特征embedding共用一个*W*矩阵

Field-Each Type：每个特征embedding共用一个*W*矩阵， 那么如果有*f*个特征的话，这里的*W**i*需要*f*个。所以这里的参数个数(*f*−1)×*k*×*k*， 这里的*f*−1是因为两两组合之后，比如`[0,1,2]`， 两两组合`[0,1], [0,2], [1,2]`。 这里用到的域是0和1。

Field-Interaction Type：每组特征交互的时候，用一个*W*矩阵， 那么这里如果有*f*个特征的话，就有f*(f-1)/2个



模型结构如下：

![image-20210308142624189](https://img-blog.csdnimg.cn/20210703160140322.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)







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





#### SIM

DIN模型本质是利用注意力机制计算权重，然后做加权平均

注意力层的计算量正比于n(用户行为序列长度)，只能记录最近几百个物品，否则计算量太大，实时性不能保证

缺点：关注短期兴趣，遗忘长期兴趣



如何改进DIN？目标：保留用户长期行为序列（n很大），而且计算量不会过大。

改进DIN

- DIN 对LastN向量做加权平均，权重是相似度。
- 如果某LastN物品与候选物品差异很大，则权重接近零。
- 快速排除掉与候选物品无关的LastN物品，降低注意力层的计算量。



SIM模型

- 保留用户长期行为记录，n的大小可以是几千。
- 对于每个候选物品，在用户LastN记录中做快速查找，找到k个相似物品。
- 把LastN变成TopK，然后输入到注意力层。
- SIM模型减小计算量（从n降到k）。



第一步：查找

方法一：Hard Search

- 根据候选物品的类目，保留LastN物品中类目相同的。
- 简单，快速，无需训练。

方法二：Soft Search

- 把物品做embedding，变成向量。
- 把候选物品向量作为query，做k近邻查找，保留LastN物品中最接近的k个。
- 效果更好，编程实现更复杂。

第二步：注意力机制

- 基本与DIN没有区别

- DIN的序列短，记录用户近期行为。SIM的序列长，记录用户长期行为。

  时间越久远，重要性越低。因此加入了时间信息
  用户与某个LastN物品的交互时刻距今为t。对t做离散化，再做embedding，变成向量d。
  把两个向量做concatenation，表征一个LastN物品，向量x是物品 embedding,向量d是时间的embedding。



#### DIEN

参考链接：https://zhuanlan.zhihu.com/p/433135805



DIN的成功主要在于基于Attention机制动态刻画用户兴趣，解决了之前k维用户Embedding只能表达k个独立的兴趣。但是DIN并没有考虑用户历史行为之间的相关性，也没考虑行为之间的先后顺序。而在电商场景，特定用户的历史行为都是一个随时间排序的序列，既然是时间相关的序列，就一定存在某种依赖关系。这样的序列信息对于推荐过程无疑是有价值的。



DIEN模型和DIN模型架构相似，输入特征分别经过Embedding层、兴趣表达层、MLP层、输出层，最终得到CTR预估。

区别在于兴趣表达不同，DIEN模型的创新在于构建了兴趣进化网络。

![img](https://pic4.zhimg.com/80/v2-4f211aa4761c0e145fd7946ca8ab52df_1440w.webp)





DIEN的兴趣进化网络分为三层，分别是行为序列层（Behavior Layer）、兴趣抽取层（Interest Extractor Layer）、兴趣演化层（Interest Evolving Layer）。

每一层的作用可以总结如下。

（1）行为序列层。其主要作用是把原始的用户行为序列id转换为Embedding。

（2）兴趣抽取层。其主要作用是通过序列模型模拟用户兴趣迁移，抽取用户兴趣。

（3）兴趣演化层。其主要作用是通过在兴趣抽取层基础上加入注意力机制，模拟与目标广告相关的兴趣演化过程。兴趣演化层是DIEN最重要的模块，也是最主要的创新点。



**兴趣抽取层** 是通过序列模型GRU（Gated Recurrent Unit，门控循环单元）处理序列特征，能够刻画行为序列之间的相关性。相比传统的序列模型RNN（Recurrent Nerual Network），GRU解决了RNN梯度消失的问题。和LSTM（Long Short-Term Memory）相比，GRU的参数更少，训练收敛速度更快。



**辅助损失** 经过GRU组成的兴趣抽取层之后，用户的行为向量 被进一步抽象，形成了兴趣状态向量  。然而隐向量 只能捕捉行为之间的依赖关系，不能有效的表达用户兴趣。基于目标广告的损失函数只能有效的预测用户最终的兴趣，而无法有监督的学习历史兴趣状态  。我们知道，每一步的兴趣状态都会直接影响用户的连续行为，因此DIEN通过引入辅助损失，有监督的学习兴趣状态向量  。

首先需要明确的就是辅助损失是计算哪两个量的损失。计算的是用户每个时刻的兴趣表示（GRU每个时刻输出的隐藏状态形成的序列）与用户当前时刻实际点击的物品表示（输入的embedding序列）之间的损失，相当于是行为序列中的第t+1个物品与用户第t时刻的兴趣表示之间的损失

（为什么这里用户第t时刻的兴趣与第t+1时刻的真实点击做损失呢？我的理解是，只有知道了用户第t+1真实点击的商品，才能更好的确定用户第t时刻的兴趣）。



**兴趣演化层** 

用户的兴趣会因为外部环境或内部认知随着时间变化，特点如下：

- **兴趣是多样化的，可能发生漂移**。兴趣漂移对行为的影响是用户可能在一段时间内对各种书籍感兴趣，而在另一段时间却需要衣服。
- 虽然兴趣可能会相互影响，但是**每一种兴趣都有自己的发展过程**，例如书和衣服的发展过程几乎是独立的。**而我们只关注与target item相关的演进过程**

####  BST

使用Transformer建模用户行为序列

**用户行为序列 （UBS: User Behavior Sequence）**蕴含了可以刻画用户兴趣的丰富信息。近两年针对UBS发表了很多优质paper。大部分的UBS建模方式可归结为：

- 朴素简单的 *sum/mean pooling*，工业实践中效果其实还不错。
- *weight pooling*，关键点是weight的计算方式。例如经典模型 DIN，DIN 使用注意力机制来捕获候选item与用户点击item序列之间的相似性作为weight。
- RNN类，考虑时序信息。例如阿里随后利用 GRU 捕捉USB中的序列信息，将 DIN 升级为DIEN。

随着 Transformer 在 很多NLP任务中的表现超过RNN，相比RNN也有可并行等独特优势，利用 Transformer 替代RNN 捕捉 序列信息是一个很自然的idea，这也是本文的动机。用户行为序列 + Transformer （更准确地说，应该是Transformer中的 Multi-head Self-attention ），两者天然地很搭。

![img](https://pic2.zhimg.com/v2-93efc96287ba91d822db6fe09e574e65_r.jpg)



多头注意力机制略。这里有所不同的在于位置编码(PE）

BST通过item embedding 拼接 position embedding的形式引入时序信息

**pos(vi) = timestamp(vt) -timestamp(vi)**

其中 vi 表示用户点击序列中的第i个item，vt 表示当前候选item



**存在的问题和疑惑：**

1. 文章写的argue动机是，很多模型没有考虑行为序列的时序信息，而DIEN正是解决这个问题，文中却没有提起DIEN和其他相关模型，实验也没有对比。
2. 没有时间戳用法的相关细节介绍，例如单位以及得到之后如何使用，一个合理做法是是按照取值进行等频分桶。
3. 实验部分可以更严谨，a.没有公开数据集；b.没有介绍对比算法的参数细节，也没有很多最新算法的离线对比；c.一些重要参数的说明，例如embedding size中4～64的具体细节。

另外，从文中表述看，一个重要细节是用户**行为序列长度固定为N**，**没有进行pooling**，即在Transformer后直接拼接 N*d 喂给NN。因此一个合理推测是，截断取用户最近的N个行为，若用户少于N个行为则直接padding补零向量。如果确实如此，那么又会引出两个问题：

1. 文中参数表中说明序列长度定为20，对于淘宝非常丰富的用户行为场景来说是否显得太小，序列长度延长之后的效果差异与性能差异，文中也没有给出更多细节。
2. BST 模型采用拼接的方式，那么这里性能的提升究竟是拼接带来的，还是Transformer层带来的？实验中一个更公平的对比是，增加一个实验，WDL(+seq)， DIN 也都直接拼接最后的embdding喂给NN，而不是做pooling。





#### DISN

全称 Deep Session Interest Network(深度会话兴趣网络) ，重点在Session(会话)

参考链接：https://datawhalechina.github.io/fun-rec/#/ch02/ch2.2/ch2.2.4/DSIN

https://zhuanlan.zhihu.com/p/89700141

**动机：**在DIEN的基础上改进

一方面，DIEN使用GRU对历史行为建模，由于循环神经网络的线性计算，为了实现实时推荐，序列长度往往不能超过50，不能建模长序列。

另一方面，一大串序列的商品中，往往出现的一个规律就是在比较短的时间间隔内的商品往往会很相似，时间间隔长了之后，商品之间就会出现很大的差别，这个是很容易理解的，一个用户在半个小时之内的浏览点击的几个商品的相似度和一个用户上午点击和晚上点击的商品的相似度很可能是不一样的。

DIEN模型会直接把这一大串行为序列放入GRU让它自己去学，如果一大串序列一块让GRU学习的话，往往用户的行为快速改变和突然终止的序列会有很多噪声点，不利于模型的学习。

所以，作者这里就是从序列本身的特点出发， 把一个用户的行为序列分成了多个会话，所谓会话，其实就是按照时间间隔把序列分段，每一段的商品列表就是一个会话，那这时候，会话里面每个商品之间的相似度就比较大了，而会话与会话之间商品相似度就可能比较小。

eg:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210310144926564.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)



**模型结构**

![模型结构](https://img-blog.csdnimg.cn/20210310151619214.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)



四个核心

1. **会话划分层：**首先，在用户行为序列输入到模型之前，要按照固定的时间间隔(比如30分钟)给他分开段，每一段里面的商品序列称为一个会话Session。
2. **会话兴趣提取层：**学习商品时间的依赖关系或者序列关系，由于上面把一个整的行为序列划分成了多段，那么在这里就是每一段的商品时间的序列关系要进行学习，当然我们说可以用GRU， 不过这里作者用了**多头注意力机制**，相比GRU来讲，没有梯度消失，并且可以并行计算。
3. **会话交互层：**上面研究了会话内各个商品之间的关联关系，接下来就是研究会话与会话之间的关系了，虽然我们说各个会话之间的关联性貌似不太大，但是可别忘了会话可是能够表示一段时间内用户兴趣的， 所以研究会话与会话的关系其实就是在学习用户兴趣的演化规律，这里用了**双向的LSTM**，不仅看从现在到未来的兴趣演化，还能学习未来到现在的变化规律。
4. **会话兴趣局部激活层**既然会话内各个商品之间的关系和会话与会话之间的关系学到了。然后当然是针对性的模拟与目标广告相关的兴趣进化路径了。这里仍然使用注意力机制， 每次关注与当前商品更相关的兴趣。



数据流动：

会话划分层：Q = K x T x dim_model

K是会话个数，T是会话中行为序列的长度，dim_model是序列经过embedding层后的嵌入向量维度

——————————————————————————————————————————————————

会话兴趣提取层： K个会话分别过多头注意力机制，数据维度不变，然后对会话内沿着T维度作avg-pooling，得到

K个 维度为dim_model的向量I1，I2，.......Ik

——————————————————————————————————————————————————

会话交互层，通过双向LSTM建模

——————————————————————————————————————————————————

会话兴趣局部激活层：用户的会话兴趣与目标物品越相近，那么应该赋予更大的权重，这里依然使用注意力机制来刻画这种相关性，根据结构图也能看出，这里是用了两波注意力计算，最终分别得到一个[1，dim_model]的向量

![image-20221026173207468](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221026173207468.png)

——————————————————————————————————————————————————

Output Layer：这个就很简单了，上面的用户行为特征， 物品行为特征以及求出的会话兴趣特征进行拼接，然后过一个DNN网络，就可以得到输出了。



实现细节

1.多头注意力机制中，采用了特殊的位置编码方式。这是因为还需要考虑各个会话之间的位置信息，毕竟这里是多个会话，并且各个会话之间也是有位置顺序的呀，所以除了会话内行为序列的位置编码，还需要对每个会话添加一个Positional Encoding， 在DSIN中，这种对位置的处理，称为Bias Encoding。

![image-20221026173708745](C:\Users\ys\AppData\Roaming\Typora\typora-user-images\image-20221026173708745.png)

2.要维护两个mask矩阵，一个用于会话内部的行为序列，一个用于会话之间。





### 推荐系统多样性（重排）

#### 概述

##### 物品相似度

如果曝光给用户的物品两两不相似，就说明推荐系统具有多样性

如何衡量物品的相似度

1.基于物品属性标签——类目、品牌、关键词

物品属性标签：类目、品牌、关键词
根据一级类目、二级类目、品牌计算相似度

eg

物品i：美妆、彩妆、香奈儿。
物品j：美妆、香水、香奈儿。
相似度：sim1（i，j）= 1，sim2（i，j）= 0 , sim3（i，j）= 1。



2.基于物品向量表征。

- 用召回的双塔模型学到的物品向量（不好，原因是推荐系统中头部现象很严重，热门物品干扰大，双塔模型学不好这一点，不利于与对长尾物品的推荐）
- 基于内容的向量表征（好，使用CV/NLP模型提取特征并融合）

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/18.png)

但这样存在训练困难，传统方法需要人工标注数据，例如使用图片分类任务

**CLIP是当前公认最有效的预训练方法。**
**思想：对于图片一文本二元组，预测图文是否匹配。**
**优势：无需人工标注。小红书的笔记天然包含图片+文字，大部分笔记图文相关**。



##### 提升多样性的方法

在粗排和精排后添加后处理，**其中精排后处理也叫重排**，而粗排后处理不被关注，但也很有用

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/16.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/15.png)







#### MMR多样性算法

（Maximal Margnal Relevance）是从搜索排序中引申来的

回顾： 精排给n个候选物品打分，融合之后的分数为reward_i，reward_n
把第i和j个物品的相似度记作sim（i，j）
从n个物品中选出k个，既要有高精排分数，也要有多样性

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/19.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/20.png)



滑动窗口优化

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/21.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/22.png)

对滑动窗口直观的解释，距离近的物品希望不相似，距离远的则无所谓

例如曝光30个物品，希望第1，2，3个物品不相似，第30个物品与第一个是否相似则无关紧要



#### 重排的规则

规则优先级高于多样性算法

**规则1：最多连续出现k篇某种笔记**
小红书推荐系统的物品分为图文笔记、视频笔记
最多连续出现k=5篇图文笔记，最多连续出现k=5篇视频笔记
如果排i到i+4的全都是图文笔记，那么排在i+5的必须是视频笔记

**规则2：每k篇笔记最多出现1篇某种笔记**
运营推广笔记的精排分会乘以大于1的系数（boost）帮助笔记获得更多曝光
为了防止boost影响体验，限制每k=9篇笔记最多出现1篇运营推广笔记
如果排第i位的是运营推广笔记，那么排i+1到i+8的不能是运营推广笔记

**规则3：前t篇笔记最多出现k篇某种笔记**
排名前t篇笔记最容易被看到，对用户体验最重要（小红书的top4为首屏）
小红书推荐系统有带电商卡片的笔记，过多可能会影响体验
前t=1篇笔记最多出现k=0篇带电商卡片的笔记
前t=4篇笔记最多出现k=1篇带电商卡片的笔记



MMR和重排规则可以轻易结合

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/23.png)



#### DPP多样性算法

详见notes：https://github.com/wangshusen/RecommenderSystem/blob/main/Notes/06_Rerank.pdf

**数学基础**

**矩阵的特征值之积等于矩阵的行列式**

**矩阵的特征值之和等于矩阵的迹**

超平行体



















### 多任务学习

#### 概述

##### 引言

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/1.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/2.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/3.png)



![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/4.png)







##### 1.为什么要用多任务学习

（1）如果要用一个词来概括所有各式各样的推荐系统的终极目标，那就是“用户满意度”，但我们无法找到一个显示的指标量化用户满意度。虽然没有显式的用户满意度评价指标，但是现在的app都存在类似上述视频号推荐场景的丰富具体的隐式反馈。但这些独立的隐式反馈也存在一些挑战：

- 目标偏差：点赞、分享表达的满意度可能比播放要高
- 物品偏差：不同视频的播放时长体现的满意度不一样，有的视频可能哄骗用户看到尾部（如标题党）
- 用户偏差：有的用户表达满意喜欢用点赞，有的用户可能喜欢用收藏

因此我们需要使用多任务学习模型针对多个目标进行预测，并在线上融合多目标的预测结果进行排序。多任务学习也不能直接表达用户满意度，但是可以最大限度利用能得到的用户反馈信息进行充分的表征学习，并且可建模业务之间的关系，从而高效协同学习具体任务。

（2）工程便利，不用针对不同的任务训练不同的模型。一般推荐系统中排序模块延时需求在40ms左右，如果分别对每个任务单独训练一个模型，难以满足需求。出于控制成本的目的，需要将部分模型进行合并。合并之后，能更高效的利用训练资源和进行模型的迭代升级。



##### **2.为什么多任务学习有效**

多任务学习的优势在于通过部分参数共享，联合训练，能在保证“还不错”的前提下，实现多目标共同提升。原因有以下几种：

- 任务互助：对于某个任务难学到的特征，可通过其他任务学习
- 隐式数据增强：不同任务有不同的噪声，一起学习可抵消部分噪声
- 学到通用表达，提高泛化能力：模型学到的是对所有任务都偏好的权重，有助于推广到未来的新任务
- 正则化：对于一个任务而言，其他任务的学习对该任务有正则化效果

> 多任务学习有效的原因是引入了归纳偏置，两个效果：
>
> - 互相促进： 可以把多任务模型之间的关系看作是互相先验知识，也称为归纳迁移，有了对模型的先验假设，可以更好提升模型的效果。解决数据稀疏性其实本身也是迁移学习的一个特性，多任务学习中也同样会体现
> - 泛化作用：不同模型学到的表征不同，可能A模型学到的是B模型所没有学好的，B模型也有其自身的特点，而这一点很可能A学不好，这样一来模型健壮性更强



##### 3.多任务学习研究的问题

- 网络结构设计：主要研究哪些参数共享、在什么位置共享、如何共享。这一方向我们认为可以分为两大类，第一类是在设计网络结构时，考虑目标间的显式关系（例如淘宝中，点击之后才有购买行为发生），以阿里提出的ESMM为代表；另一类是目标间没有显示关系（例如短视频中的收藏与分享），在设计模型时不考虑label之间的量化关系，以谷歌提出的MMOE为代表。
- 多loss的优化策略：主要解决loss数值有大有小、学习速度有快有慢、更新方向时而相反的问题。最经典的两个工作有UWL（Uncertainty Weight）：通过自动学习任务的uncertainty，给uncertainty大的任务小权重，uncertainty小的任务大权重；GradNorm：结合任务梯度的二范数和loss下降梯度，引入带权重的损失函数Gradient Loss，并通过梯度下降更新该权重。





##### loss加权融合

一种最简单的实现多任务学习的方式是对不同任务的loss进行加权

eg1：使用加权的交叉熵损失函数，以视频播放时长作为正样本权值，以1作为负样本权值。作者认为按点击率排序会倾向于把诱惑用户点击（用户未必真感兴趣)的视频排前面，而观看时长能更好地反映出用户对视频的兴趣，通过重新设计loss使得该模型在保证主目标点击的同时，将视频观看时长转化为样本的权重，达到优化平均观看时长的效果。

eg2：人工手动调整权值，例如 0.3 x L(点击)+0.7 x L(视频完播)



这种loss加权的方式优点如下：

- 模型简单，仅在训练时通过梯度乘以样本权重实现对其它目标的加权
- 模型上线简单，和base完全相同，不需要额外开销

缺点：

- 本质上并不是多目标建模，而是将不同的目标转化为同一个目标。样本的加权权重需要根据AB测试才能确定。



##### 预估分数融合



![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/6.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/7.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/8.png)

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/9.png)









##### Shared-Bottom

底层共享结构：通过共享底层模块，学习任务间通用的特征表征，再往上针对每一个任务设置一个Tower网络，每个Tower网络的参数由自身对应的任务目标进行学习。Shared Bottom可以根据自身数据特点，使用MLP、DeepFM、DCN、DIN等，Tower网络一般使用简单的MLP。

代码如下，共享特征embedding，共享底层DNN网络，任务输出层独立，loss直接使用多个任务的loss值之和。

```python
def Shared_Bottom(dnn_feature_columns, num_tasks=None, task_types=None, task_names=None,
                  bottom_dnn_units=[128, 128], tower_dnn_units_lists=[[64,32], [64,32]],
                  l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024,dnn_dropout=0,
                  dnn_activation='relu', dnn_use_bn=False):

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed)
    #共享输入特征
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    #共享底层网络
    shared_bottom_output = DNN(bottom_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    #任务输出层
    tasks_output = []
    for task_type, task_name, tower_dnn in zip(task_types, task_names, tower_dnn_units_lists):
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(shared_bottom_output)

        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        tasks_output.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=tasks_output)
    return modelCopy to clipboardErrorCopied
```

优点：

- 浅层参数共享，互相补充学习，任务相关性越高，模型loss优化效果越明显，也可以加速训练。

缺点：

- 任务不相关甚至优化目标相反时（例如新闻的点击与阅读时长），可能会带来负收益，多个任务性能一起下降。

一般把Shared-Bottom的结构称作**“参数硬共享”**，多任务学习网络结构设计的发展方向便是如何设计更灵活的共享机制，从而实现**“参数软共享”**。



#### ESMM

**背景与动机**

传统的CVR预估问题存在着两个主要的问题：**样本选择偏差**和**稀疏数据**。下图的白色背景是曝光数据，灰色背景是点击行为数据，黑色背景是购买行为数据。传统CVR预估使用的训练样本仅为灰色和黑色的数据。

![img](https://pic4.zhimg.com/80/v2-2f0df0f6933dd8405c478fcce91f7b6f_1440w.jpg)

这会导致两个问题：

- 样本选择偏差（sample selection bias，SSB）：如图所示，CVR模型的正负样本集合={点击后未转化的负样本+点击后转化的正样本}，但是线上预测的时候是样本一旦曝光，就需要预测出CVR和CTR以排序，样本集合={曝光的样本}。构建的训练样本集相当于是从一个与真实分布不一致的分布中采样得到的，这一定程度上违背了机器学习中训练数据和测试数据独立同分布的假设。
- 训练数据稀疏（data sparsity，DS）：点击样本只占整个曝光样本的很小一部分，而转化样本又只占点击样本的很小一部分。如果只用点击后的数据训练CVR模型，可用的样本将极其稀疏。



ESMM借鉴多任务学习的思路，引入两个辅助任务CTR、CTCVR(已点击然后转化)，同时消除以上两个问题。

三个预测任务如下：

- **pCTR**：p(click=1 | impression)；
- **pCVR**: p(conversion=1 | click=1,impression)；
- **pCTCVR**: p(conversion=1, click=1 | impression) = p(click=1 | impression) * p(conversion=1 | click=1, impression)；

> 注意：其中只有CTR和CVR的label都同时为1时，CTCVR的label才是正样本1。如果出现CTR=0，CVR=1的样本，则为不合法样本，需删除。 pCTCVR是指，当用户已经点击的前提下，用户会购买的概率；pCVR是指如果用户点击了，会购买的概率。



**模型结构**

![img](https://pic1.zhimg.com/80/v2-6d8189bfe378dc4bf6f0db2ba0255eac_1440w.jpg)

主任务和辅助任务共享特征，不同任务输出层使用不同的网络，将cvr的预测值*ctr的预测值作为ctcvr任务的预测值，利用ctcvr和ctr的label构造损失函数：

![img](https://pic3.zhimg.com/80/v2-0098ab4556a8c67a1c12322ea3f89606_1440w.jpg)





该架构具有两大特点，分别给出上述两个问题的解决方案：

- 帮助CVR模型在完整样本空间建模（即曝光空间X)。从公式中可以看出，pCVR 可以由pCTR 和pCTCVR推导出。从原理上来说，相当于分别单独训练两个模型拟合出pCTR 和pCTCVR，再通过pCTCVR 除以pCTR 得到最终的拟合目标pCVR 。在训练过程中，模型只需要预测pCTCVR和pCTR，利用两种相加组成的联合loss更新参数。pCVR 只是一个中间变量。而pCTCVR和pCTR的数据是在完整样本空间中提取的，从而相当于pCVR也是在整个曝光样本空间中建模。

-  特征表达的迁移学习（embedding层共享）。CVR和CTR任务的两个子网络共享embedding层，网络的embedding层把大规模稀疏的输入数据映射到低维的表示向量，该层的参数占了整个网络参数的绝大部分，需要大量的训练样本才能充分学习得到。由于CTR任务的训练样本量要大大超过CVR任务的训练样本量，ESMM模型中特征表示共享的机制能够使得CVR子任务也能够从只有展现没有点击的样本中学习，从而能够极大地有利于缓解训练数据稀疏性问题。

  ![img](https://pic1.zhimg.com/80/v2-0b0c6dc7d4c38fa422a2876b7c4cc638_1440w.jpg)



**代码实践**

与Shared-Bottom同样的共享底层机制，之后两个独立的Tower网络，分别输出CVR和CTR，计算loss时只利用CTR与CTCVR的loss。CVR Tower完成自身网络更新，CTR Tower同时完成自身网络和Embedding参数更新。在评估模型性能时，重点是评估主任务CVR的auc。

```python
def ESSM(dnn_feature_columns, task_type='binary', task_names=['ctr', 'ctcvr'],
         tower_dnn_units_lists=[[128, 128],[128, 128]], l2_reg_embedding=0.00001, l2_reg_dnn=0,
         seed=1024, dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False):

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    ctr_output = DNN(tower_dnn_units_lists[0], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    cvr_output = DNN(tower_dnn_units_lists[1], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    ctr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(ctr_output)
    cvr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cvr_output)

    ctr_pred = PredictionLayer(task_type, name=task_names[0])(ctr_logit)
    cvr_pred = PredictionLayer(task_type)(cvr_logit)

    ctcvr_pred = tf.keras.layers.Multiply(name=task_names[1])([ctr_pred, cvr_pred])#CTCVR = CTR * CVR

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[ctr_pred, cvr_pred, ctcvr_pred])
    return model
```



**思考存在的问题**

1. 能不能将乘法换成除法？ 即分别训练CTR和CTCVR模型，两者相除得到pCVR。论文提供了消融实验的结果，表中的DIVISION模型，比起BASE模型直接建模CTCVRR和CVR，有显著提高，但低于ESMM。原因是pCTR 通常很小，除以一个很小的浮点数容易引起数值不稳定问题。

   ![img](https://pic3.zhimg.com/80/v2-c0b2c860bd63a680d27c911c2e1ba8a2_1440w.jpg)

2. 网络结构优化，Tower模型更换？两个塔不一致？ 原论文中的子任务独立的Tower网络是纯MLP模型，事实上业界在使用过程中一般会采用更为先进的模型（例如DeepFM、DIN等），两个塔也完全可以根据自身特点设置不一样的模型。这也是ESMM框架的优势，子网络可以任意替换，非常容易与其他学习模型集成。

3. 比loss直接相加更好的方式？ 原论文是将两个loss直接相加，还可以引入动态加权的学习机制。

4. 更长的序列依赖建模？ 有些业务的依赖关系不止有曝光-点击-转化三层，后续的改进模型提出了更深层次的任务依赖关系建模。

   阿里的ESMM2: 在点击到购买之前，用户还有可能产生加入购物车（Cart）、加入心愿单（Wish）等行为。





#### MMOE

2018年谷歌提出的，全称是Multi-gate Mixture-of-Experts

模型中，如果采用一个网络同时完成多个任务，就可以把这样的网络模型称为多任务模型， 这种模型能在不同任务之间学习共性以及差异性，能够提高建模的质量以及效率。 常见的多任务模型的设计范式大致可以分为三大类：

- hard parameter sharing 方法： 这是非常经典的一种方式，底层是共享的隐藏层，学习各个任务的共同模式，上层用一些特定的全连接层学习特定任务模式。

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/ed10df1df313413daf2a6a6174ef4f8c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_1,color_FFFFFF,t_70,g_se,x_16#pic_center)

  这种方法最大的优势是Task越多， 单任务更加不可能过拟合，即可以减少任务之间过拟合的风险。 但是劣势也非常明显，就是底层强制的shared layers难以学习到适用于所有任务的有效表达。 **尤其是任务之间存在冲突的时候**。当两个任务相关性没那么好(比如排序中的点击率与互动，点击与停留时长)，此时这种结果会遭受训练困境，毕竟所有任务底层用的是同一组参数。

- soft parameter sharing: 硬的不行，那就来软的，这个范式对应的结果从`MOE->MMOE->PLE`等。 即底层不是使用共享的一个shared bottom，而是有多个tower， 称为多个专家，然后往往再有一个gating networks在多任务学习时，给不同的tower分配不同的权重，那么这样对于不同的任务，可以允许使用底层不同的专家组合去进行预测，相较于上面所有任务共享底层，这个方式显得更加灵活

- 任务序列依赖关系建模：这种适合于不同任务之间有一定的序列依赖关系。比如电商场景里面的ctr和cvr，其中cvr这个行为只有在点击之后才会发生。所以这种依赖关系如果能加以利用，可以解决任务预估中的样本选择偏差(SSB)和数据稀疏性(DS)问题

  

![在这里插入图片描述](https://img-blog.csdnimg.cn/29c5624f2c8a46c097f097af7dbf4b45.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA57-75rua55qE5bCPQOW8ug==,size_2,color_FFFFFF,t_70,g_se,x_16#pic_center)



**混合专家模型**

我们知道共享的这种模型结构，会遭受任务之间冲突而导致可能无法很好的收敛，从而无法学习到任务之间的共同模式。这个结构也可以看成是多个任务共用了一个专家。

先抛开任务关系， 我们发现一个专家在多任务学习上的表达能力很有限，于是乎，尝试引入多个专家，这就慢慢的演化出了混合专家模型。 

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/10.31.2.png)

模型集成思想： 这个东西很像bagging的思路，即训练多个模型进行决策，这个决策的有效性显然要比单独一个模型来的靠谱一点，不管是从泛化能力，表达能力，学习能力上，应该都强于一个模型

1. 注意力思想: 为了增加灵活性， 为不同的模型还学习了重要性权重，这可能考虑到了在学习任务的共性模式上， 不同的模型学习的模式不同，那么聚合的时候，显然不能按照相同的重要度聚合，所以为各个专家学习权重，默认了不同专家的决策地位不一样。这个思想目前不过也非常普遍了。
2. multi-head机制: 从另一个角度看， 多个专家其实代表了多个不同head, 而不同的head代表了不同的非线性空间，之所以说表达能力增强了，是因为把输入特征映射到了不同的空间中去学习任务之间的共性模式。可以理解成从多个角度去捕捉任务之间的共性特征模式。

MOE使用了多个混合专家增加了各种表达能力，但是， 一个门控并不是很灵活，因为这所有的任务，最终只能选定一组专家组合，即这个专家组合是在多个任务上综合衡量的结果，并没有针对性了。 如果这些任务都比较相似，那就相当于用这一组专家组合确实可以应对这多个任务，学习到多个相似任务的共性。 但如果任务之间差的很大，这种单门控控制的方式就不行了，因为此时底层的多个专家学习到的特征模式相差可能会很大，毕竟任务不同，而单门控机制选择专家组合的时候，肯定是选择出那些有利于大多数任务的专家， 而对于某些特殊任务，可能学习的一塌糊涂。

所以，这种方式的缺口很明显，这样，也更能理解为啥提出多门控控制的专家混合模型了。



**MMOE结构**

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/10.31.1.png)

这个改造看似很简单，只是在OMOE上额外多加了几个门控网络，但是却起到了杠杆般的效果，我这里分享下我的理解。

- 首先，就刚才分析的OMOE的问题，在专家组合选取上单门控会产生限制，此时如果多个任务产生了冲突，这种结构就无法进行很好的权衡。 而MMOE就不一样了。MMOE是针对每个任务都单独有个门控选择专家组合，那么即使任务冲突了，也能根据不同的门控进行调整，选择出对当前任务有帮助的专家组合。所以，我觉得单门控做到了**针对所有任务在专家选择上的解耦**，而多门控做到了**针对各个任务在专家组合选择上的解耦**。
- 多门控机制能够建模任务之间的关系了。如果各个任务都冲突， 那么此时有多门控的帮助， 此时让每个任务独享一个专家，如果任务之间能聚成几个相似的类，那么这几类之间应该对应的不同的专家组合，那么门控机制也可以选择出来。如果所有任务都相似，那这几个门控网络学习到的权重也会相似，所以这种机制把任务的无关，部分相关和全相关进行了一种统一。
- 灵活的参数共享， 这个我们可以和hard模式或者是针对每个任务单独建模的模型对比，对于hard模式，所有任务共享底层参数，而每个任务单独建模，是所有任务单独有一套参数，算是共享和不共享的两个极端，对于都共享的极端，害怕任务冲突，而对于一点都不共享的极端，无法利用迁移学习的优势，模型之间没法互享信息，互为补充，容易遭受过拟合的困境，另外还会增加计算量和参数量。 而MMOE处于两者的中间，既兼顾了如果有相似任务，那就参数共享，模式共享，互为补充，如果没有相似任务，那就独立学习，互不影响。 又把这两种极端给进行了统一。
- 训练时能快速收敛，这是因为相似的任务对于特定的专家组合训练都会产生贡献，这样进行一轮epoch，相当于单独任务训练时的多轮epoch。



**极化问题**

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/11.6.png)



解决方案：

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/11.6.2.png)







**MMOE网络的搭建逻辑**

首先是传入封装好的dnn_features_columns

就是数据集先根据特征类别分成离散型特征和连续型特征，然后通过sparseFeat或者DenseFeat进行封装起来，组成的一个列表。

```python
dnn_features_columns = [SparseFeat(feat, feature_max_idx[feat], embedding_dim=4) for feat in sparse_features] \
                         + [DenseFeat(feat, 1) for feat in dense_features]Copy to clipboardErrorCopied
```



传入之后， 首先为这所有的特征列建立Input层，然后选择出离散特征和连续特征来，连续特征直接拼接即可， 而离散特征需要过embedding层得到连续型输入。把这个输入与连续特征拼接起来，就得到了送入专家的输入。

接下来，建立MMOE的多个专家， 这里的专家直接就是DNN，当然这个可以替换，比如MOSE里面就用了LSTM，这样的搭建模型方式非常灵活，替换起来非常简单。 把输入过多个专家得到的专家的输出，这里放到了列表里面。

接下来，建立多个门控网络，由于MMOE里面是每个任务会有一个单独的门控进行控制，所以这里的门控网络个数和任务数相同，门控网络也是DNN，接收输入，得到专家个输出作为每个专家的权重，把每个专家的输出加权组合得到门控网络最终的输出，放到列表中，这里的列表长度和task_num对应。

接下来， 为每个任务建立tower，学习特定的feature信息。同样也是DNN，接收的输入是上面列表的输出，每个任务的门控输出输入到各自的tower里面，得到最终的输出即可。 最终的输出也是个列表，对应的每个任务最终的网络输出值。



```python
def MMOE(dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
        gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu',
        dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    
    num_tasks = len(task_names)
    
    # 构建Input层并将Input层转成列表作为模型的输入
    input_layer_dict = build_input_layers(dnn_feature_columns)
    input_layers = list(input_layer_dict.values())
    
    # 筛选出特征中的sparse和Dense特征， 后面要单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns))
    
    # 获取Dense Input
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])
    
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(dnn_feature_columns)
    # 离散的这些特特征embedding之后，然后拼接，然后直接作为全连接层Dense的输入，所以需要进行Flatten
    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict, flatten=False)
    
    # 把连续特征和离散特征合并起来
    dnn_input = combined_dnn_input(dnn_sparse_embed_input, dnn_dense_input)
    
    # 建立专家层
    expert_outputs = []
    for i in range(num_experts):
        expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='expert_'+str(i))(dnn_input)
        expert_outputs.append(expert_network)
    
    expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outputs)
    
    # 建立多门控机制层
    mmoe_outputs = []
    for i in range(num_tasks):  # num_tasks=num_gates
        # 建立门控层
        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='gate_'+task_names[i])(dnn_input)
        gate_out = Dense(num_experts, use_bias=False, activation='softmax', name='gate_softmax_'+task_names[i])(gate_input)
        gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)
        
        # gate multiply the expert
        gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False), name='gate_mul_expert_'+task_names[i])([expert_concat, gate_out])
        
        mmoe_outputs.append(gate_mul_expert)
    
    # 每个任务独立的tower
    task_outputs = []
    for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outputs):
        # 建立tower
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='tower_'+task_name)(mmoe_out)
        logit = Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outputs.append(output)
    
    model = Model(inputs=input_layers, outputs=task_outputs)
    return modelCopy to clipboardErrorCopied
```



**存在的问题**

- MMOE中所有的Expert是被所有任务所共享，这可能无法捕捉到任务之间更复杂的关系，从而给部分任务带来一定的噪声。
- 在复杂任务机制下，MMOE不同专家在不同任务的权重学的差不多
- 不同的Expert之间没有交互，联合优化的效果有所折扣





#### PLE

PLE(Progressive Layered Extraction)模型由腾讯PCG团队在2020年提出，主要为了解决跷跷板问题

**背景与动机**

文章首先提出多任务学习中不可避免的两个缺点：

- 负迁移（Negative Transfer）：针对相关性较差的任务，使用shared-bottom这种硬参数共享的机制会出现负迁移现象，不同任务之间存在冲突时，会导致模型无法有效进行参数的学习，不如对多个任务单独训练。
- 跷跷板现象（Seesaw Phenomenon）：针对相关性较为复杂的场景，通常不可避免出现跷跷板现象。多任务学习模式下，往往能够提升一部分任务的效果，但同时需要牺牲其他任务的效果。即使通过MMOE这种方式减轻负迁移现象，跷跷板问题仍然广泛存在。

在腾讯视频推荐场景下，有两个核心建模任务：

- VCR(View Completion Ratio)：播放完成率，播放时间占视频时长的比例，回归任务
- VTR(View Through Rate) ：有效播放率，播放时间是否超过某个阈值，分类任务

这两个任务之间的关系是复杂的，在应用以往的多任务模型中发现，要想提升VTR准确率，则VCR准确率会下降，反之亦然。





，PLE在网络结构设计上提出两大改进：

**一、CGC**(Customized Gate Control) 定制门控

PLE将共享的部分和每个任务特定的部分**显式的分开**，强化任务自身独立特性。把MMOE中提出的Expert分成两种，任务特定task-specific和任务共享task-shared。保证expert“各有所得”，更好的降低了弱相关性任务之间参数共享带来的问题。

网络结构如图所示，同样的特征输入分别送往三类不同的专家模型（任务A专家、任务B专家、任务共享专家），再通过门控机制加权聚合之后输入各自的Tower网络。门控网络，把原始数据和expert网络输出共同作为输入，通过单层全连接网络+softmax激活函数，得到分配给expert的加权权重，与attention机制类型。

![img](https://pic3.zhimg.com/80/v2-c92975f7c21cc568a13cd9447adc757a_1440w.jpg)

任务A有 ![[公式]](https://www.zhihu.com/equation?tex=m_A) 个expert，任务B有 ![[公式]](https://www.zhihu.com/equation?tex=m_B) 个expert，另外还有 ![[公式]](https://www.zhihu.com/equation?tex=m_S) 个任务A、B共享的Expert。这样对Expert做一个**显式的分割**，可以让task-specific expert只受自己任务梯度的影响，不会受到其他任务的干扰（每个任务保底有一个独立的网络模型)，而只有task-shared expert才受多个任务的混合梯度影响。

MMOE则是将所有Expert一视同仁，都加权输入到每一个任务的Tower，其中任务之间的关系完全交由gate自身进行学习。虽然MMOE提出的门控机制理论上可以捕捉到任务之间的关系，比如任务A可能与任务B确实无关，则MMOE中gate可以学到，让个别专家对于任务A的权重趋近于0，近似得到PLE中提出的task-specific expert。**如果说MMOE是希望让expert网络可以对不同的任务各有所得，则PLE是保证让expert网络各有所得。**



二、**PLE** (progressive layered extraction) 分层萃取

PLE就是上述CGC网络的多层纵向叠加，以获得更加丰富的表征能力。在分层的机制下，Gate设计成两种类型，使得不同类型Expert信息融合交互。task-share gate融合所有Expert信息，task-specific gate只融合specific expert和share expert。模型结构如图：

![img](https://pic2.zhimg.com/80/v2-ff3b4aff3511e6e56a3b509f244c5ab1_1440w.jpg)

将任务A、任务B和shared expert的输出输入到下一层，下一层的gate是以这三个上一层输出的结果作为门控的输入，而不是用原始input特征作为输入。这使得gate同时融合task-shares expert和task-specific expert的信息，论文实验中证明这种不同类型expert信息的交叉，可以带来更好的效果。





## 四、冷启动问题

### 物品冷启动

UGC（user )：用户上传的，例如小红书上用户新发布的笔记、B站上用户新上传的视频、今日头条上作者新发布的文章.........

PGC：(platform) 平台采购的，例如腾讯视频中的视频



#### **为什么要特殊对待新笔记？**

新笔记缺少与用户的交互，导致推荐的难度大、效果差
扶持新发布、低曝光的笔记，可以增强作者发布意愿

扶持新笔记的目的：
目的1：促进发布，增大内容池
新笔记获得的曝光越多，作者创作积极性越高
反映在发布渗透率、人均发布量

目的2：挖掘优质笔记
做探索，让每篇新笔记都能获得足够曝光
挖掘的能力反映在高热笔记占比



#### **优化冷启的目标**

1.精准推荐：克服冷启的困难，把新笔记推荐给合适的用户，不引起用户反感。
2.激励发布：流量向低曝光新笔记倾斜，激励作者发布
3.挖掘高潜：通过初期小流量的试探，找到高质量的笔记，给与流量倾斜



#### **评价指标**

1.作者侧指标：
发布渗透率=当日发布人数/日活人数  （发布一篇或以上，就算一个发布人数）

例：
当日发布人数=100万
日活人数=2000万
发布渗透率= 100/2000 = 5%



人均发布量=当日发布笔记数/日活人数

例：
每日发布笔记数=200万
日活人数=2000万
人均发布量= 200/2000 = 0.1

发布渗透率、人均发布量反映出作者的发布积极性
冷启的重要优化目标是促进发布，增大内容池。
新笔记获得的曝光越多，首次曝光和交互出现得越早，作者发布积极性越高



2.用户侧指标：
新笔记的消费指标：新笔记的点击率、交互率

新笔记的点击率、交互率。问题：曝光的基尼系数很大
少数头部新笔记占据了大部分的曝光。
分别考察高曝光、低曝光新笔记
高曝光：比如>1000次曝光。低曝光：比如<1000次曝光

大盘消费指标：大盘的消费时长、日活、月活

大力扶持低曝光新笔记会发生什么？
作者侧发布指标变好。
用户侧大盘消费指标变差

因此在做冷启动时要保证大盘指标基本持平



3.内容侧指标：
高热笔记占比
高热笔记：前30天获得1000+次点击
高热笔记占比越高，说明冷启阶段挖掘优质笔记的能力越强



**普通笔记只看用户侧指标，不看其他两个指标**



#### 冷启动的优化点

优化全链路（包括召回和排序，让新笔记尽量快的走完推荐全链路，尽可能的被曝光，同时做到个性化推荐，不让用户反感）
流量调控（流量怎么在新物品、老物品中分配）

**扶持新笔记的两个抓手：单独的召回通道、在排序阶段提权**



##### 召回

召回的依据
自带图片、文字、地点；算法或人工标注的标签。没有用户点击、点赞等信息；没有笔记ID embedding



冷启召回的困难
缺少用户交互，还没学好笔记ID embedding，导致双塔模型效果不好
缺少用户交互，导致ItemCF不适用



召回通道
ItemCF召回（不适用）
双塔模型（改造后适用）
类目、关键词召回（适用）
聚类召回（适用）
Look-Alike召回（适用）



###### 改进双塔模型

ID Embedding改进方案1：新笔记使用default embedding

所有新笔记共享一个学习得到的default  embedding，到下次模型训练的时候，新笔记才有自己的ID embedding 向量。

ID Embedding改进方案2：利用相似笔记embedding向量。

使用多模态预训练模型得到基于图文内容的笔记特征向，并查找topk内容最相似的高曝笔记（高曝笔记IDembedding学的比较好）

把k个高曝笔记的embedding向量取平均，作为新笔记的embedding



多个向量召回池
多个召回池，让新笔记有更多曝光机会1小时新笔记
6小时新笔记，
24小时新笔记，
30天笔记


共享同一个双塔模型，那么多个召回池不增加训练的代价。





###### 类目、关键词召回

**基于类目的召回**

系统本身会建立每个用户的用户画像，包括感兴趣的类目、关键词，可以基于此做新物品的召回



系统维护类目索引：类目 —> 笔记列表（按时间倒排）
用类目索引做召回：用户画像—>类目—> 笔记列表

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/24.png)

取回笔记列表上前k篇笔记（即最新的k篇）



基于关键词的召回与之类似，不再赘述



缺点1：只对刚刚发布的新笔记有效取回某类目/关键词下最新的k篇笔记。发布几小时之后，就再没有机会被召回。
缺点2：弱个性化，不够精准





###### 聚类召回

基本思想
如果用户喜欢一篇笔记，那么他会喜欢内容相似的笔记（与ItemCF的区别在于，由于新物品缺少用户交互记录，所以基于图文内容衡量物品相似度）
事先训练一个神经网络，基于笔记的类目和图容，把笔记映射到向量（CNN&Bert  CLIP训练）
对笔记向量做聚类，划分为1000 cluster，记录cluster 的中心方向。（k-means聚类，用余弦度）

具体操作如下：

1.聚类索引
一篇新笔记发布之后，用神经网络把它映射到一个特征向量
从1000个向量（对应1000 个cluster）中找到最相似的向量，作为新笔记的cluster
索引：cluster →笔记ID列表（按时间倒排）

2.线上召回
给定用户ID，找到他的last-n交互的笔记列表，把这些笔记作为种子笔记。
把每篇种子笔记映射到向量，寻找最相似的cluster。（知道了用户对哪些cluster感兴趣）
从每个cluster的笔记列表中，取回最新的m篇笔记。最多取回mn篇新笔记。

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/25.png)



###### Look-like人群扩散

最初起源于互联网广告，基于种子用户，扩散推荐给其他潜在用户

eg：Tesla model3的用户特征是25-30岁，本科以上，喜欢苹果电子产品。这样的潜在用户可能只有几万，因为大部分用户的信息填报是缺失的，可能不填年龄、学历。可以基于Look-like扩散到几百万用户。

如何计算两个用户的相似度？
UserCF：两个用户有共同的兴趣点
Embedding：两个用户向量的cosine较大



look-like用于新物品召回

点击、点赞、收藏、转发--用户对笔记可能感兴趣
把有交互的用户作为新笔记的种子用户用look-alike在相似用户中扩散



通过将与新笔记有交互的用户的用户向量作平均作为该新笔记的表征

这个特征向量是有交互的用户的向量的平均
采用近线更新此特征向量，每当有用户交互该物品，更新笔记的特征向量



![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/26.png)



最后，当用户向服务器发送推荐请求，将该用户向量与新物品的特征向量做近似最近邻查找。

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/27.png)





##### 流量监控

工业界的做法
假设推荐系统只分发年龄<30天的笔记
假设采用自然分发，新笔记（年龄<24小时）的曝光占比为1/30
扶持新笔记，让新笔记的曝光占比远大于1/30



流量调控技术的发展
1.在推荐结果中强插新笔记
2.对新笔记的排序分数做提权

3.通过提权，对新笔记做保量（例如设定每篇新笔记需要在发布后24h至少获得100次曝光）
4.差异化保量（在新笔记发布时衡量它的质量，对质量高的笔记设置更高的保量值）



###### 新物品提权（boost)

在粗排和重排这两个漏斗处人工干预新老物品的流量分配

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/28.png)优点：容易实现，投入产出比好

缺点：曝光量对提权系数很敏感，很难精确控制曝光量，容易过度曝光和不充分曝光



###### 新物品保量

**也是通过更复杂的提权算法实现**

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/29.png)

动态提权保量

![](https://raw.githubusercontent.com/jackaihfia2334/hexo_image_save/master/30.png)





保量的难点

保量成功率远低于100%，很多笔记在24小时达不到100次曝光

- 召回、排序存在不足
- 提权系数调得不好

线上环境变化会导致保量失败

- 线上环境变化：新增召回通道、升级排序模型
- 改变重排打散规则
- 线上环境变化后，需要调整提权系数



思考题：

给所有新笔记一个很大的提权系数（比如4倍），直到达成100次曝光为止。这样的保量成功率很高。

为什么不用这种方法呢？给新笔记分数boost越多，对新笔记越有利？

好处：分数提升越多，曝光次数越多
坏处：把笔记推荐给不太合适的受众

- 点击率、点赞率等指标会偏低
- 长期会受推荐系统打压，难以成长为热门笔记



###### 差异化保量

保量：不论新笔记质量高低，都做扶持，在前24小时给100次曝光
差异化保量：不同笔记有不同保量目标，普通笔记保100次曝光，内容优质的笔记保100~500次曝光

基础保量：24小时100次曝光。
内容质量：用多模态模型根据图文内容评价质量高低，给予额外保量目标，上限是加200次曝光
作者质量：根据作者历史上的笔记质量，给予额外保量目标，上限是加200次曝光。
一篇笔记最少有100次保量，最多有500次保量。



#### 冷启动的A/B测试

推荐系统常用的AB测试只考察用户侧消费指标，而冷启动的AB测试还需要额外考察作者侧发布指标

新笔记冷启的AB测试：

作者侧指标：发布渗透率、人均发布量。
用户侧指标：对新笔记的点击率、交互率。
大盘指标：消费时长、日活、月活。
**标准的AB测试只针对大盘指标和用户侧指标**













总结：
**冷启的AB测试需要观测作者发布指标和用户消费指标**
**各种AB测试的方案都有缺点**





**设计方案的时候，问自己几个问题：**
实验组、对照组新笔记会不会抢流量？
新笔记、老笔记怎么抢流量？
同时隔离笔记、用户，会不会让内容池变小？
如果对新笔记做保量，会发生什么？











## 五、工业界实用技术



### 工程实现

#### 离线训练之参数服务器





## 六、评价指标

#### 精排阶段

CTR(点击率)预测：AUC与logloss

定义&如何计算：https://zhuanlan.zhihu.com/p/280797054

优缺点分析：





CVR (Click Value Rate): 转化率

















## 七、常用数据集：

https://www.jianshu.com/p/5c88f4bd7c71

工业上的CTR预测数据集一般都是`multi-group categorial form`的形式，就是类别型特征最为常见，这种数据集一般长这样：

![img](https://img-blog.csdnimg.cn/20210118190044920.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_1,color_FFFFFF,t_70#pic_center)

这里的亮点就是框出来的那个特征，这个包含着丰富的用户兴趣信息。

对于特征编码，作者这里举了个例子：`[weekday=Friday, gender=Female, visited_cate_ids={Bag,Book}, ad_cate_id=Book]`， 这种情况我们知道一般是通过one-hot的形式对其编码， 转成系数的二值特征的形式。但是这里我们会发现一个`visted_cate_ids`， 也就是用户的历史商品列表， 对于某个用户来讲，这个值是个多值型的特征， 而且还要知道这个特征的长度不一样长，也就是用户购买的历史商品个数不一样多，这个显然。这个特征的话，我们一般是用到multi-hot编码，也就是可能不止1个1了，有哪个商品，对应位置就是1， 所以经过编码后的数据长下面这个样子：![img](https://img-blog.csdnimg.cn/20210118185933510.png)

这个就是喂入模型的数据格式了，这里还要注意一点 就是上面的特征里面没有任何的交互组合，也就是没有做特征交叉。这个交互信息交给后面的神经网络去学习。



####  Criteo 数据集

Homepage: https://labs.criteo.com/2013/12/download-terabyte-click-logs/https://labs.criteo.com/2013/12/download-terabyte-click-logs/

##### 不同模型效果排名: https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo







