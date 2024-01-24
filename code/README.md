## README

single_rnn.py为单任务RNN文件，其中数据集需要先放置在文件同一目录下

- 默认使用LSTM模型，如果需要使用RNN或者GRU模型，在line35-line37中选择所需要的模型结果即可

改文件为comp单任务的RNN模型训练，最终会输出模型在comp任务上的测试准确率



lwf.py文件为使用了CNN/RNN结合LWF的实现方式，其中数据集需要先放置在文件同一目录下

- 默认是使用LSTM模型，如果需要使用CNN模型，将代码文件line57-line74取消注释，将line32-line55注释，直接运行即可
- 如果需要使用RNN或者GRU模型，在line35-line37中选择所需要的模型结果即可

训练过程中会输出当前任务及之前任务的测试集准确率，最终会输出模型在4个任务上的测试准确率



+ single_cnn.py为单任务CNN文件

  

+ ewc.py文件为使用了CNN结合EWC的实现方式，卷积层和全连接层的超参数分别为lambda1和lambda2，可以根据需要调节超参数的大小。