
# 为什么做这个项目？

## 以下不是我做的

- 训练模型基于 Andrej Karpathy 的代码 https://github.com/karpathy/char-rnn ，以及 Samy Bengio 的论文 http://arxiv.org/abs/1506.03099 。
- Zhang Zibin 为上述代码添加了中文支持 http://github.com/zhangzibin/char-rnn-chinese 。

## 我所做的部分
- JSON RPC Web Service，基于 [xavante](http://keplerproject.github.io/xavante/) 和 [json4lua](https://github.com/craigmj/json4lua)。
- 在 `char-rnn-chinese` 的基础上修改了网页布局，并完成此“对话”。
- 对采样程序处理未收录字符的部分略作了修改。

## 为什么做这个项目

本项目的一个上线版本 http://tianhua.me:9987 使用了包括康德、海德格尔、萨特在内的哲学家作品的中译文作为训练输入。个人来说，这些哲学家当然是值得尊敬的。但时下，尤其是在当代艺术相关的讨论中，一个时髦的做法是将这些哲学家的文本不由分说地糅合进艺术家和评论者的论述之中，甚至以这种杂合的方式作为“作品”。这种据说是“后现代”的文本受到一部分人的追捧。由于材料限制未能收录更为典型的法国当代诸“哲学家”作品，以及特别是目前在当代艺术界的贩卖“[二手烟](http://www.bundpic.com/posts/post/555aeb40f032a0e68c981d1a)”的“哲学活动家”，这不得不说是一个遗憾。不过，正是在对先前哲学家进行不断“借用”的意义上，这个项目——“无尽的对话”——与那些“哲学活动家”和一心仰慕哲学的艺术工作者们相呼应。他们通过堆砌让人半懂不懂的词句，它构成了一个貌似严肃、高深的文本，但它只有高深的形式；而这个项目将他们的做法推到了极端，由无意识的机器来“代劳”，以构成对他们的讽刺。

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

感谢 @(taineleau)[https://www.douban.com/people/ylen/] 对 RNN-LSTM 的介绍与提供的硬件支持；感谢 @(围巾喵)[https://www.douban.com/people/Viking_mew_two/]、@(瑞尔曼)[https://www.douban.com/people/45233999/]、 @(genuine)[https://www.douban.com/people/60566956/] 的相关讨论。
