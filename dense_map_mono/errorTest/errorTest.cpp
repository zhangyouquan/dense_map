#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <string>
using namespace cv;
using namespace std;

int main()
{
    fstream fin("/home/zyq/桌面/dense_map/dense_map_mono/data.txt");
    if(!fin)cout << "读文件出错！" << endl;

    double x,y;
    vector<pair<double,double>> data;
    while(!fin.eof())
    {
        fin >> x >> y;
        cout << "深度均值：" << x << " 深度方差：" << y << endl;
        data.push_back(make_pair(x,y));
    }

    double count = 0;
    vector<Point2d> pts_mu;
    vector<Point2d> pts_squared;
    Mat picture(300,610,CV_8UC3, Scalar (255,255,255));
    for(vector<pair<double,double>>::iterator it = data.begin(); it != data.end(); it++)
    {
        pts_mu.push_back(Point2d(count,300 - it->first * (-100)));
        pts_squared.push_back(Point2d(count,300 - it->second * 100));
        count += 3.0;//刻度
    }

    for(int i = 0; i < pts_mu.size() - 1; i++)
    {
        line(picture,pts_mu[i],pts_mu[i+1],
             Scalar(0,255,0),4);
        line(picture,pts_squared[i],pts_squared[i+1],
             Scalar(0,0,255),4);
    }
    string words1 = "red: depth_mu";
    string words2 = "green: depth_squared";
    putText(picture,words1,Point2d(440,20),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(255,0,0));
    putText(picture,words2,Point2d(440,40),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(255,0,0));
    imshow("深度误差和方差：",picture);

    waitKey(0);

    return 0;
}