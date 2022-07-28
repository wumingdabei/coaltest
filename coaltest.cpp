//author zhenglibo

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
using namespace cv;
using namespace std;

int main(){
     VideoCapture cap;
     cap.open("12.mp4"); 
 
     if(!cap.isOpened())
         return -1;
 
     int width = cap.get(CAP_PROP_FRAME_WIDTH);
     int height = cap.get(CAP_PROP_FRAME_HEIGHT);
     int frameRate = cap.get(CAP_PROP_FPS);
     int totalFrames = cap.get(CAP_PROP_FRAME_COUNT);
 
     cout << "视频的宽度：" << width << endl;
     cout << "视频的高度：" << height << endl;
     cout << "视频的总帧数：" << totalFrames << endl;
     cout << "帧率：" << frameRate << endl;
 
     Mat frame;
     while(1)
     {
         cap>>frame;
         if(frame.empty())
             break;

         namedWindow("test2",0);

         resizeWindow("test2", 500, 500);

         imshow("test2",frame);
        
         waitKey(20);
     }
     cap.release();
 
     return 0;
 }