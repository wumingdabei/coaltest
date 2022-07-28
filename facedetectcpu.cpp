// Copyright 2022 lb
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
 
int main()
{
	Mat image, image_gray;
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cerr << "Can not open video source";
		return -1;
	}
	CascadeClassifier face_cascade;    								
		if (!face_cascade.load("haarcascade_frontalface_alt2.xml"))
		{
			cout << "Load haarcascade_frontalface_alt failed!" << endl;
			return 0;
		}
	while (1)                 
	{
		if (!capture.read(image)) {
			cerr << "Can not read frame from webcam";
			return -1;
		}
 
		cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
		equalizeHist(image_gray, image_gray);
		
		
 
		//vector 是个类模板 需要提供明确的模板实参 vector<Rect>则是个确定的类 模板的实例化
		vector<Rect> faceRect;
	
		int64 start = cv::getTickCount();
 
		//检测关于脸部位置
		face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 2, 0, Size(30, 30));
 
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
		cout << "FPS : " << fps << endl;
 
		for (size_t i = 0; i < faceRect.size(); i++)
		{
			rectangle(image, faceRect[i], Scalar(0, 0, 255));
		}
 
		imshow("人脸识别图", image);
		if (waitKey(30) == 'q') {
			break;
		}
	}
 
	return 0;
}