
# 为什么做这个项目？

## 以下不是我做的

- 训练模型基于 Andrej Karpathy 的代码 https://github.com/karpathy/char-rnn ，以及 Samy Bengio 的论文 http://arxiv.org/abs/1506.03099 。
- Zhang Zibin 为上述代码添加了中文支持 http://github.com/zhangzibin/char-rnn-chinese 。

## 我所做的部分
- JSON RPC Web Service，基于 [xavante](http://keplerproject.github.io/xavante/) 和 [json4lua](https://github.com/craigmj/json4lua)。
- 在 `char-rnn-chinese` 的基础上修改了网页布局，并完成此“对话”。
- 对采样程序处理未收录字符的部分略作了修改。

**新增** http://innovors.info:9987/?subdir=guwen 古文（根据全唐文数据训练）生成
**新增** 福柯（根据《疯癫与文明》、《规训与惩罚》）；各家杂合（“overall”）。


## 为什么做这个项目

某天，几位小伙伴聊天时说起，现在那些搞艺术的人（包括艺术家和评论者）很喜欢和哲学家们混在一起，写一些让人半懂不懂的东西，据说是在描述、评论或阐释作品。细读之后不免发现，这些半通不通的文本并没有真正理解他们所引用的哲学家的意思，而只是在模仿这些先哲说话的方式。因此我就想到，既然 `char-rnn` 具有从字符的粒度上把握文本“风格”的能力（可是它又不像 `neuralart` 那样，还有一个相对独立的“内容”网络之类），那么何不让程序模仿那些半吊子的“爱好者”，学着那些哲学大家的方式，写一些和那些艺术工作者们一样半通不通却貌似深刻的文本呢？于是就有了这个项目。

具体实现上，使用海德格尔、萨特以及康德的一些作品作为训练文本，分别训练了三个“风格模型”。每个模型都可以接受一段文字来为模型的内部状态赋初值。第一轮中采用的模型是人为选定的，让它产生一段文字之后，再用这段文字去初始化其他的各个模型产生多个备选文本，并计算各个模型的“自信程度”（也就是说，所产生出的文字出自各训练文本的可能性），最高的那个会被选中输出。此后的每一轮都以上一轮的优胜模型和输出为基础，如此类推。因为每个模型只“看到”了同一个作者的文本，它们产生的文本也比较鲜明地带有原作者的风格印记，因此将它们拼凑在一起，看起来会像是一场对话一样。我们知道，这样的“对话”或拼接可以一直进行下去，所以我称这个项目为“无尽的对话”。而它产生的东西是没有意义的，这就符合了那些艺术工作者们所理解的“后现代”风格，并且也是他们实际在做的事情。

给了几个小伙伴看了程序产生的文本之后，他们意外地觉得，有些内容还是比较有趣的。于是我把它放到了网上。这就有了这个项目。

## 所以它算什么

什么都不是。——也许算计算机艺术吧：）

## 如何参与

- 访问 http://innovors.info:9987 ，分享你觉得有意思的结果；
- 对本项目提出建议、意见和设想，请到 [Issues](https://github.com/zhuth/char-rnn-chinese/issues) 版块提出，但更希望你直接：
- Fork，修改代码，把它放上线！

### Install Requisites

Please install [Torch](http://torch.ch) first, clone this repository, and install extra requisites:

	luarocks install --server=http://rocks.moonscript.org/manifests/amrhassan --local json4Lua
	luarocks install xavante
	luarocks install wsapi-xavante

## 授权

MIT License

## 致谢

感谢 @[taineleau](https://www.douban.com/people/ylen/) 对 RNN-LSTM 的介绍与提供的硬件支持；感谢 @[围巾喵](https://www.douban.com/people/Viking_mew_two/)、@[瑞尔曼](https://www.douban.com/people/45233999/)、 @[genuine](https://www.douban.com/people/60566956/) 的相关讨论。
