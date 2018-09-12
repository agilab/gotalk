# gotalk

模型基于 Google 的论文[《Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge》](https://arxiv.org/pdf/1609.06647.pdf)

这个项目基于 Go 实现了一个深度学习看图说话服务，即机器学习的 serving（inference） 部分，Tensorflow 的 Python 训练代码在 [这里](https://github.com/tensorflow/models/tree/master/im2txt)。

代码了实现了 tensorflow 模型导入、输入输出和 LSTM beam search 功能，并实现了 web 服务。

我打包的 Docker 镜像中已经包含了所有模型文件，可以直接运行，见 [docker hub](https://hub.docker.com/r/unmerged/gotalk/)。	

模型文件由 [free_tf_model 工具](https://github.com/agilab/freeze_tf_model) 从 tf.train.Saver 保存的 checkpoint 生成，对于图说这个项目，训练代码保存的 checkpoint 无法直接用，需要通过 [这个代码](https://github.com/agilab/im2txt) 中的 save_model.py 工具转化成 inference 模型后再调用 free_tf_model。
