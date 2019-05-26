#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


int main() 
{
    ifstream fin("calibdata.txt");  //标定图像路径文件
    ofstream fout("caliberation_result.txt");  // 保存标定结果的文件  

    int image_count = 0;  
    Size image_size;  
    Size board_size = Size(9, 6);            
    vector<Point2f> image_points_buf;        
    vector<vector<Point2f>> image_points_seq; 
    string filename; 
    vector<string> filenames;

    while (getline(fin, filename))
    {
        ++image_count;
        Mat imageInput = imread(filename);
        filenames.push_back(filename);
        if(image_count == 1){
            image_size.width = imageInput.cols;
            image_size.height = imageInput.rows;
        }
        if (0 == findChessboardCorners(imageInput, board_size, image_points_buf)){           
            cout << "can not find chessboard corners!\n";  // 找不到角点
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

    //以下是摄像机标定
    Size square_size = Size(10, 10);         // 实测标定板棋盘格大小
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
    //开始
    calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
    //完成

    double total_err = 0.0;  
    double err = 0.0;          
    vector<Point2f> image_points2;  

    for (i=0;i<image_count;i++)
    {
        vector<Point3f> tempPointSet = object_points[i];
        projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
        //计算误差
        vector<Point2f> tempImagePoint = image_points_seq[i];
        Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
        Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);

        for (int j = 0 ; j < tempImagePoint.size(); j++){
            image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);
            tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        total_err += err/= point_counts[i]; 
        fout << "误差：" << endl;   
        fout << i+1 << err<< endl;   
    }      
    fout << "total：" << total_err/image_count<<endl;   

    //保存定标结果 
    Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); 
    fout << "内参矩阵：" << endl;   
    fout << cameraMatrix << endl << endl;   
    fout << "畸变系数：\n";   
    fout << distCoeffs << endl << endl << endl;   
    for (int i=0; i<image_count; i++) 
    { 
        fout << i+1 << "的旋转向量：" << endl;   
        fout << tvecsMat[i] << endl; 
        Rodrigues(tvecsMat[i], rotation_matrix);   
        fout << i+1 << "的旋转矩阵：" << endl;   
        fout << rotation_matrix << endl;   
        fout << i+1 << "的平移向量：" << endl;   
        fout << rvecsMat[i] << endl << endl;   
    }   
    fout<<endl;

    Mat mapx = Mat(image_size, CV_32FC1);
    Mat mapy = Mat(image_size, CV_32FC1);
    Mat R = Mat::eye(3, 3, CV_32F);
    string imageFileName;
    std::stringstream StrStm;
    for (int i = 0 ; i != image_count ; i++)
    {
        initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
        Mat imageSource = imread(filenames[i]);
        Mat newimage = imageSource.clone();
        remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);     
        StrStm.clear();
        imageFileName.clear();
        StrStm << i+1;
        StrStm >> imageFileName;
        imageFileName += "_d.jpg";
        imwrite(imageFileName, newimage);
    }
    fin.close();
    fout.close();
    return 0;
}