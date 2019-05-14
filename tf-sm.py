from tensorflow.examples.tutorials.mnist import input_data  

import tensorflow as tf  
#mnist数据集被分成两部分：60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）
#这样的切分很重要，在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，
#从而更加容易把设计的模型推广到其他数据集上（泛化）
mnist = input_data.read_data_sets("datasets/mnist_data/", one_hot=True)
sess = tf.InteractiveSession()
 
#输入图片x是一个2维的浮点数张量。这里，分配给它的shape为[None, 784]，其中784是一张展平的MNIST图片的维度。
#None表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小，意即x的数量不定。
x = tf.placeholder("float", shape=[None, 784])
 
#输出类别值y_也是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。
#虽然placeholder的shape参数是可选的，有了它，TensorFlow能够自动捕捉因数据维度不一致导致的错误。
y_ = tf.placeholder("float", shape=[None, 10])
 
#权重，初始化为0的tensor
W = tf.Variable(tf.zeros([784,10]))
#偏置值，初始化为0的tensor
b = tf.Variable(tf.zeros([10]))
 
#初始化
#变量需要通过seesion初始化后，才能在session中使用。
#这一初始化步骤为，为初始值指定具体值（本例当中是全为零），并将其分配给每个变量,可以一次性为所有变量完成此操作。
sess.run(tf.global_variables_initializer())
 
#y为模型的预测值
y = tf.nn.softmax(tf.matmul(x,W) + b)
 
#误差 交叉熵
#tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了。我们计算的交叉熵是指整个minibatch的。
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
 
#用最速下降法让交叉熵下降，步长为0.01
#返回的train_step操作对象，在运行时会使用梯度下降来更新参数。
#因此，整个模型的训练可以通过反复地运行train_step来完成。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
 
#每一步迭代，都会加载50个训练样本，然后执行一次train_step，
#并通过feed_dict将x 和 y_张量占位符用训练数据替代。
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1]})
 
#tf.argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
#由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
#而 tf.argmax(y_,1) 代表正确的标签，可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
#本方法返回一个布尔数组。为了计算分类的准确率，将布尔值转换为浮点数来代表对、错，然后取平均值。
#例如：[True, False, True, True]变为[1,0,1,1]，计算出平均值为0.75。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 
#计算平均数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
#打印准确度
print(accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#http://fengjian0106.github.io/

#https://blog.csdn.net/kevin_cc98/article/details/79582906

#https://blog.csdn.net/u014679795/article/details/53467264

#https://github.com/fengjian0106/hed-tutorial-for-document-scanning


#http://fengjian0106.github.io/2018/06/02/Document-Scanning-With-TensorFlow-And-OpenCV-Part-Two/