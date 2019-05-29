#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main() 
{
    ifstream fin("calibdata.txt");  //标定图像路径文件
    ofstream fout("caliberation_result.txt");  // 保存标定结果的文件  

    int image_count = 0;  
    Size image_size;  
    Size board_size = Size(6, 4);  //change in here          
    vector<Point2f> image_points_buf;        
    vector<vector<Point2f>> image_points_seq; 
    string filename; 
    vector<string> filenames;

    while (getline(fin, filename))
    {
        ++image_count;
		Mat imageInput = imread("1.bmp", 1);
        filenames.push_back(filename);
        if(image_count == 1){
            image_size.width = imageInput.cols;
            image_size.height = imageInput.rows;
        }
        if (findChessboardCorners(imageInput, board_size, image_points_buf) == 0){           
            cout << "can not find chessboard corners!\n";  
            exit(1);
        } 
        else {
            Mat view_gray;
            cvtColor(imageInput, view_gray, COLOR_RGB2GRAY);
            cornerSubPix(view_gray, image_points_buf, Size(5,5), Size(-1,-1), TermCriteria(2 + 1, 30, 0.1));
            image_points_seq.push_back(image_points_buf); 
            drawChessboardCorners(view_gray, board_size, image_points_buf, false); 
            //imshow("Camera Calibration", view_gray);     
            waitKey(500);       
        }
    }
    int CornerNum = board_size.width * board_size.height;

    //deal camare in time
    Size square_size = Size(10, 10);  
    vector<vector<Point3f>> object_points;
    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); 
    vector<int> point_counts;  
    Mat distCoeffs=Mat(1, 5, CV_32FC1,Scalar::all(0));      
    vector<Mat> tvecsMat;    
    vector<Mat> rvecsMat;    

    int i, j, t;
    for (t=0; t<image_count; t++) {
        vector<Point3f> tempPointSet;
        for (i=0; i<board_size.height; i++) 
        {
            for (j=0; j<board_size.width; j++) 
            {
                Point3f realPoint;
                realPoint.x = i * square_size.width;
                realPoint.y = j * square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        object_points.push_back(tempPointSet);
    }
    for (i=0; i<image_count; i++){
        point_counts.push_back(board_size.width * board_size.height);
    }   
    //start
    calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
    //end

    Mat mapx = Mat(image_size, CV_32FC1);
    Mat mapy = Mat(image_size, CV_32FC1);
    Mat R = Mat::eye(3, 3, CV_32F);
    std::stringstream StrStm;
    fin.close();
    fout.close();
    //deal with vidio
    VideoCapture cap("chi.mp4");
    if (!cap.isOpened())
        return 0;
    int resultImg_cols, resultImg_rows;
	Mat frame;
	VideoWriter video("dis.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));//can change
    while(1){
        cap>>frame;
        if (frame.empty())
            break;
		//one way
        /*initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
        Mat showImg = frame.clone();
        remap(frame, showImg, mapx, mapy, INTER_LINEAR);*/

		//another way
		Mat showImg = frame.clone();
		undistort(frame, showImg, cameraMatrix, distCoeffs);
		resize(frame, frame, Size(640, 480)); //can change
		video << frame;
        /*imshow("frame", frame);
        imshow("result", showImg);
        if (27 == waitKey(1))
            break;*/
    }
    destroyAllWindows();

    waitKey(0);
    return 0;
}
