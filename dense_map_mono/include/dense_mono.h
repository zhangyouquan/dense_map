#pragma once //防止头文件重复添加，不过不通用，只有微软支持。如果要跨平台，最好还是用 #ifndef #define #endif
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <ostream>
using namespace std;
#include <sophus/se3.hpp>
using Sophus::SE3d;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

#include <opencv2/core/core.hpp>//hpp文件说明这是模板类
#include <opencv2/highgui/highgui.hpp>//包括图形界面和视频图像处理的头文件。
#include <opencv2/imgproc/imgproc.hpp>//图像处理模块,比如，图像转灰度，图像滤波等
using namespace cv;

/**
 *    单目相机在已知轨迹下的稠密深度估计
 *    使用极线搜索 + NCC 匹配的方式
 */


// ------------------------------------------------------------------
//一些参数
const int boarder = 20;         // 边缘宽度
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度

const double fx = 481.2f;       // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;

vector<pair<double,double>> error_val;
const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差
// ------------------------------------------------------------------

// 重要的函数
/// 从 REMODE 数据集读取数据
bool readDatasetFiles(const string &path,vector<string> &color_image_files,
                      vector<SE3d> &poses,cv::Mat &ref_depth);

/**
 * 根据新的图像更新深度估计
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         参考图像到当前图像的位姿
 * @param depth         深度
 * @param depth_cov     深度方差
 * @return              是否成功
 */
bool update(
        const Mat &ref,
        const Mat &curr,
        const SE3d &T_C_R,
        Mat &depth,
        Mat &depth_cov2
);

/**
 * 极线搜索
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         位姿
 * @param pt_ref        参考图像中点的位置
 * @param depth_mu      深度均值
 * @param depth_cov     深度方差
 * @param pt_curr       当前点
 * @param epipolar_direction  极线方向
 * @return              是否成功
 */
bool epipolarSearch(
        const Mat &ref,
        const Mat &curr,
        const SE3d &T_C_R,
        const Vector2d &pt_ref,
        const double &depth_mu,
        const double &depth_cov,
        Vector2d &pt_curr,
        Vector2d &epipolar_direction
);

/**
 * 更新深度滤波器
 * @param pt_ref    参考图像点
 * @param pt_curr   当前图像点
 * @param T_C_R     位姿
 * @param epipolar_direction 极线方向
 * @param depth     深度均值
 * @param depth_cov2    深度方向
 * @return          是否成功
 */
bool updateDepthFilter(
        const Vector2d &pt_ref,
        const Vector2d &pt_curr,
        const SE3d &T_C_R,
        const Vector2d &epipolar_direction,
        Mat &depth,
        Mat &depth_cov2
);


/**
 * 计算 NCC 评分
 * @param ref       参考图像
 * @param curr      当前图像
 * @param pt_ref    参考点
 * @param pt_curr   当前点
 * @return          NCC评分
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 双线性灰度插值
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt)
{
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}


// 显示估计的深度图
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 像素到相机坐标系
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
            (px(0, 0) - cx) / fx,
            (px(1, 0) - cy) / fy,
            1
    );
}

// 相机坐标系到像素
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
            p_cam(0, 0) * fx / p_cam(2, 0) + cx,
            p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 检测一个点是否在图像边框内
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) < width - boarder && pt(1, 0) <= height - boarder;
}


/// 评测深度估计
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// -----------