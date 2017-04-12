# gotalk

这是我在 [QCon 北京2017演讲](http://2017.qconbeijing.com/presentation/872) 的配套代码。

基于 Go 实现了一个深度学习看图说话服务，即机器学习的 serving（inference） 部分，Tensorflow 的 Python 训练代码在 [这里](https://github.com/tensorflow/models/tree/master/im2txt)。

代码了实现了 tensorflow 模型导入、输入输出和 LSTM beam search 功能，并实现了 web 服务。

我打包的 Docker 镜像中已经包含了所有模型文件，可以直接运行，见 [docker hub](https://hub.docker.com/r/unmerged/gotalk/)。
