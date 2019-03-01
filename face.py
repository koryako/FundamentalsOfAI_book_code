# coding:utf-8

import sys

reload(sys)

sys.setdefaultencoding('utf8')



import cv2

# 待检测的图片路径

imagepath = r'./heat.jpg'

# 获取训练好的人脸的参数数据，这里直接从 GitHub 上使用默认值

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
#https://github.com/opencv/opencv/tree/master/data/haarcascades 特征模型
# 读取图片

image = cv2.imread(imagepath)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 探测图片中的人脸

faces = face_cascade.detectMultiScale(

    gray,

    scaleFactor = 1.15,

    minNeighbors = 5,

    minSize = (5,5),

    flags = cv2.cv.CV_HAAR_SCALE_IMAGE

)

print "发现 {0} 个人脸!".format(len(faces))

for(x,y,w,h) in faces:

    # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

    cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)
    

cv2.imshow("Find Faces!",image)

cv2.waitKey(0)


model = Sequential()

model.add(BatchNormalization(input_shape=(96, 96, 1)))

model.add(Convolution2D(24, 5, 5, border_mode=”same”,

init=’he_normal’, input_shape=(96, 96, 1),
dim_ordering=”tf”))
model.add(Activation(“relu”))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
border_mode=”valid”))

model.add(Convolution2D(36, 5, 5))

model.add(Activation(“relu”))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
border_mode=”valid”))

model.add(Convolution2D(48, 5, 5))
model.add(Activation(“relu”))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
border_mode=”valid”))

model.add(Convolution2D(64, 3, 3))
model.add(Activation(“relu”))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
border_mode=”valid”))

model.add(Convolution2D(64, 3, 3))
model.add(Activation(“relu”))

model.add(GlobalAveragePooling2D());

model.add(Dense(500, activation=”relu”))
model.add(Dense(90, activation=”relu”))
model.add(Dense(30))

model.compile(optimizer=’rmsprop’, loss=’mse’, metrics=

[‘accuracy’])

checkpointer = ModelCheckpoint(filepath=’face_model.h5',
verbose=1, save_best_only=True)

epochs = 30

hist = model.fit(X_train, y_train, validation_split=0.2,
shuffle=True, epochs=epochs, batch_size=20, callbacks=
[checkpointer], verbose=1)

features = model.predict(region, batch_size=1)
"""
如果上述的操作还不能满足你的需求，你还可以进行如下步骤：

实验如何在保持精度和提高推理速度的同时减少卷积层和滤波器的数量；

使用迁移学习来替代卷积的部分（Xception是我的最爱）

使用一个更详细的数据库

做一些高级的图像增强来提高鲁棒性

你可能依然觉得太简单了，那么推荐你学习去做一些3D的处理，你可以参考Facebook和NVIDIA是怎么进行人脸识别和追踪的。

另外，你可以用已经学到的这些进行一些新奇的事情（你可能一直想做但不知道怎么实现）：

在视频聊天时，把一些好玩的图片放置在人脸面部上，比如：墨镜，搞笑的帽子和胡子等；

交换面孔，包括你和朋友的脸，动物和物体等；

在自拍实时视频中用一些新发型、珠宝和化妆进行产品测试；

检测你的员工是因为喝酒无法胜任一些任务；

从人们的反馈表情中提取当下流行的表情；

使用对抗网络（GANs）来进行实时的人脸-卡通变换，并使用网络实现实时人脸和卡通动画表情的同步。

好了~你现在已经学会了怎么制作你自己的视频聊天滤镜了，快去制作一个有趣的吧
"""


#include "opencv2/objdetect.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>
 
using namespace cv;
using namespace std;
 
int resize_save(Mat& faceIn, char *path, int FaceSeq);
int get_face(char *path);
 
int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		printf("usage: %s <path>\n", argv[0]);
		return -1;
	}
	
	get_face(argv[1]);
 
    return 0;
}
 
int get_face(char *path)
{
	CascadeClassifier face_cascade;  
	VideoCapture camera;
	char key = 0;
	Mat	frame;
	int ret = 0;
	int faceNum = 1;
	vector<Rect> faces;  
	Mat img_gray;  
	Mat faceImg;
 
	camera.open(0);		// 打开摄像头
	if(!camera.isOpened())
	{
	  cout << "open camera failed. " << endl;
	  return -1;
	}
	cout << "open camera succeed. " << endl;
 
	// 加载人脸分类器
	ret = face_cascade.load("/root/library/opencv/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt2.xml");
	if( !ret )
	{
		printf("load xml failed.\n");
		return -1;
	}
	cout << "load xml succeed. " << endl;
 
	while (1)  
	{
		camera >> frame;  
		if(frame.empty())
		{
			continue;
		}
		
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);  
		equalizeHist(img_gray, img_gray);  
		
		// 检测目标
		face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, 0, Size(50, 50)); 
 
		for(size_t i =0; i<faces.size(); i++)  
		{
			 /* 画矩形框出目标 */
			rectangle(frame, Point(faces[0].x, faces[0].y), Point(faces[0].x + faces[0].width, faces[0].y + faces[0].height),	
							Scalar(0, 255, 0), 1, 8);	 
		}
		imshow("camera", frame);  // 显示
		key = waitKey(1);  	// 显示后要添加延时
		
		switch (key)  
		{
			case 'p':	// 按 P 一键拍脸
				// 只限定检测一个人脸
				if(faces.size() == 1)
				{
					faceImg = frame(faces[0]);
					ret = resize_save(faceImg, path, faceNum);	// 调整大小及保存
					if(ret == 0)
					{
						printf("resize_save success.\n");
						faceNum ++;
					}
				}
				break;	
 
			case 27:	// 按 Esc 键退出
				cout << "Esc..." << endl;
				return 0;
				
			default:  
				break;	
		}  
	}  
}
 
int resize_save(Mat& faceIn, char *path, int FaceSeq)
{
	string strName;
	Mat image;
	Mat faceOut;  
	int ret;
 
	if(faceIn.empty())
	{  
    	printf("faceIn is empty.\n");
      	return -1;  
	}  
 
	if (faceIn.cols > 100)  
	{  
		resize(faceIn, faceOut, Size(92, 112));		// 调整大小，这里选择与官方人脸库图片大小兼容
		strName = format("%s/%d.jpg", path, FaceSeq);	// 先要创建文件夹
		ret = imwrite(strName, faceOut);  // 文件名后缀要正确 .jpg .bmp ...
		if(ret == false)	// 出现错误，请检测文件名后缀、文件路径是否存在
		{
			printf("imwrite failed!\n");
			printf("please check filename[%s] is legal ?!\n", strName.c_str());
			return -1;
		}
		imshow(strName, faceOut);  
	}  
	waitKey(20);  
 
    return 0;
}

--------------------- 
作者：曾哥哥_zeng 
来源：CSDN 
原文：https://blog.csdn.net/qq_30155503/article/details/79776485 
版权声明：本文为博主原创文章，转载请附上博文链接！