#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Can not open video source";
        return -1;
    }
	std::vector<cv::Rect> h_found;
    cv::Ptr<cv::cuda::CascadeClassifier> cascade = cv::cuda::CascadeClassifier::create("haarcascade_frontalface_alt2.xml");
    cv::cuda::GpuMat d_frame, d_gray, d_found;
    while(1)
    {
        Mat frame;
        if ( !cap.read(frame) ) {
            cerr << "Can not read frame from webcam";
            return -1;
        }
        d_frame.upload(frame);

        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);     
        cout<< "5555555"<<endl;   
        int64 start = cv::getTickCount();
        cascade->detectMultiScale(d_gray, d_found);
        cascade->convert(d_found, h_found);
        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        cout << "FPS : " << fps << endl;
        
		for(int i = 0; i < h_found.size(); ++i)
		{
              rectangle(frame, h_found[i], Scalar(0,255,255), 5);
		}

        imshow("Result", frame);
        if (waitKey(1) == 'q') {
            break;
        }

		// d_frame.upload(frame);
		// cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);

 
		// for (int i = 0; i < h_found.size(); ++i)
		// {
		// 	rectangle(frame, h_found[i], Scalar(0, 255, 255), 3);
		// }
 
		// imshow("Result", frame);
		// if (waitKey(30) == 'q') {
		// 	break;
		// }


    }

    return 0;
}
