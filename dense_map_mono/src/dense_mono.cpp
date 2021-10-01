#include <iostream>
#include "dense_mono.h"

int main(int argc, char** argv)
{
    if(argc !=2 )
    {
        cout << "两个参数：可执行程序，数据集文件" << endl;
        return -1;
    }

    //从数据集里面读取数据
    vector<string> color_image_files;
    vector<SE3d> pose_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1],color_image_files,pose_TWC,ref_depth);
    if(!ret)
    {
        cout << "读数据集出错！" << endl;
        return -1;
    }

    cout << "一共读取的图片数目：" << color_image_files.size() << endl;

    //第一张图
    Mat ref = imread(color_image_files[0],0);
    SE3d pose_ref_TWC = pose_TWC[0];

    //后面要用到高斯分布的深度滤波器，就是设像素点深度值符合高斯分布P(d) = N(init_depth,init_cov2)
    double init_depth = 3.0;    // 深度初始值
    double init_cov2 = 3.0;     // 方差初始值

    Mat depth(height,width,CV_64F,init_depth); //深度图
    Mat depth_cov2(height,width,CV_64F,init_cov2); //深度图方差

    for(int index = 1; index < color_image_files.size(); index++)
    {
        cout << "***稠密重建到第 " << index << " 张图了。" <<  endl;
        Mat curr = imread(color_image_files[index],0);
        if(curr.empty())continue;
        SE3d curr_pose = pose_TWC[index];
        SE3d pose_T_C_R = curr_pose.inverse() * pose_ref_TWC;// Tck+1ck
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth);//评估一下深度估计的怎么样
        plotDepth(ref_depth, depth);
        imshow("当前帧2D图像：", curr);
        waitKey(1);
    }

    cout << "估计完毕，保存深度图 ..." << endl;
    imwrite("depth.png", depth);


    ofstream ost;
    ost.open("/home/zyq/桌面/dense_map/dense_map_mono/data.txt");
    if(!ost.is_open())
    {
        cout << "没有这个文件。。。" << endl;
    }
    for(vector<pair<double,double>>::iterator it = error_val.begin(); it != error_val.end(); it++)
    {
        ost << it->first << " " << it->second << endl;
    }

    ost.close();

    cout << "保存完成." << endl;

    return 0;
}
//从数据集中读取参考位姿和参考深度值
bool readDatasetFiles(const string &path,vector<string> &color_image_files,
                      vector<SE3d> &poses,cv::Mat &ref_depth)
{
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if(!fin)return false;

    while (!fin.eof())
    {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image;
        fin >> image;

        double data[7];
        for(auto &d : data)
        {
            fin >> d;
        }
        color_image_files.push_back(path + string("/images/")+image);
        poses.push_back(SE3d(Quaterniond(data[6],data[3],data[4],data[5]),
                             Vector3d(data[0],data[1],data[2])));
        if(!fin.good())break;
    }

    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height,width,CV_64F);
    if(!fin.is_open())return false;

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }
    }

    return true;
}

bool update(
        const Mat &ref,
        const Mat &curr,
        const SE3d &T_C_R,
        Mat &depth,
        Mat &depth_cov2
)
{
    for(int x = boarder; x < width - boarder; x++)
    {
        for(int y = boarder; y < height - boarder; y++)
        {
            if(depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov)continue;
            //若深度还未收敛，极限搜索
            Vector2d pt_curr;//当前帧上的坐标点
            Vector2d epipolar_direction;//极线方向
            bool ret = epipolarSearch(ref,curr,T_C_R,Vector2d(x,y),
                                      depth.ptr<double>(y)[x],
                                      sqrt(depth_cov2.ptr<double>(y)[x]),
                                      pt_curr,epipolar_direction);

            if(ret == false)continue;//匹配失败

            //匹配成功，更新深度图
            updateDepthFilter(Vector2d(x,y),pt_curr,
                              T_C_R,epipolar_direction,depth,depth_cov2);

        }
    }
}

//对极约束最终的目的是根据像素匹配点求出E或F然后求出R和t,确定相机位姿。
bool epipolarSearch(
        const Mat &ref,
        const Mat &curr,
        const SE3d &T_C_R,
        const Vector2d &pt_ref,//参考点坐标
        const double &depth_mu,//深度
        const double &depth_cov,//深度图标准差
        Vector2d &pt_curr,
        Vector2d &epipolar_direction
)
{
    Vector3d f_ref = px2cam(pt_ref);///转到归一化相机坐标系
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;///参考帧的P向量,相当于OP向量(初始深度)。

    ///1、深度范围自定义
    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); /// 按深度初始值投影的像素（这里认为depth_mu就是平均深度了）
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;//这个可能的深度范围为什么这样求呢?
    if(d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min)); ///按最小深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max)); ///按最大深度投影的像素

    //2、确定极线（范围）
    Vector2d epipolar_line = px_max_curr - px_min_curr; /// 极线（线段形式）也就是活限制了极线搜索的范围。
    epipolar_direction = epipolar_line; /// 极线方向
    epipolar_direction.normalize();

    double half_length = 0.5 * epipolar_line.norm();    /// 极线线段的半长度
    if (half_length > 100) half_length = 100;   /// 我们不要搜索太多东西

    /// 3、开始极限搜索匹配.以深度均值点为中心，左右各取半长度
    /// 归一化互相关，0表示不相似，1代表相似
    double best_ncc = -1.0;
    Vector2d best_px_curr;///找到的最优的匹配点
    for(double l = -half_length; l <= half_length; l += 0.7) //l+=sqrt(2)
    {
        Vector2d px_curr = px_mean_curr + l * epipolar_direction; //待匹配点
        if(!inside(px_curr))continue;///若此点不在规定边界内
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC(ref,curr,pt_ref,px_curr);
        if(ncc > best_ncc)
        {
            best_ncc = ncc;///统计出最大的ncc,最大是1.
            best_px_curr = px_curr;
        }
    }

    if(best_ncc < 0.85f)  /// 只相信 NCC 很高的匹配
    {
        return false;
    }
    pt_curr = best_px_curr;
    return true;

}

bool updateDepthFilter(
        const Vector2d &pt_ref,
        const Vector2d &pt_curr,
        const SE3d &T_C_R,
        const Vector2d &epipolar_direction,
        Mat &depth,
        Mat &depth_cov2
)
{
    //三角化计算深度
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();//实现规范化，让一个向量保持相同的方向，normalize()其实就是把自身的各元素除以它的2范数
    // 但它的长度为1.0，如果这个向量太小而不能被规范化，一个零向量将会被返回。
    // 而normalized()与normalize()类似，只不过normalize()是对自身上做修改，
    // 而normalized()返回的是一个新的Vector/Matrix，并不改变原有的矩阵。


    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    //三角化计算深度
    ///下面这公式就是十四讲三角测量那一节的内容，利用对极约束。s2 * x2 = s1 * R * x1 + t
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 即：d_ref * f_ref = d_cur * f2 + t_RC
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T  f_ref, -f_ref^T  f2 ] [d_ref]   [f_ref^T  t]
    //    [ f2^T  f_ref, -f2^T     f2 ] [d_cur] = [f2^T     t]

    //以下部分参考 https://blog.csdn.net/michaelhan3/article/details/89483148 解二元一次方程
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    //点乘运算(内积)，就是对这两个向量对应位一一相乘之后求和的操作。点乘矩阵或向量可以交换顺序
    Vector2d b = Vector2d(t.dot(f_ref),t.dot(f2));

    //构建方程组
    Matrix2d A;
    A(0,0)= f_ref.dot(f_ref);
    A(0,1)= -f_ref.dot(f2);
    A(1,0)= -A(0,1);//
    A(1,1)= -f2.dot(f2);

    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;           // ref 侧的结果
    Vector3d xn = t + ans[1] * f2;          // cur 结果
    Vector3d p_esti = (xm + xn) / 2.0;      // P的位置，取两者的平均
    double depth_estimation = p_esti.norm();   // 深度值
    // 对于Vector，norm返回的是向量的二范数,
    // 对于Matrix，norm返回的是矩阵的弗罗贝尼乌斯范数（也是所有元素平方和再开方）

    /// 计算不确定性（以一个像素为误差）
    Vector3d p = f_ref * depth_estimation; //在参考帧下的世界坐标
    Vector3d a = p - t;
    double t_norm = t.norm();//2范数,相当于模长。
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);//在极线方向上减去一个像素，前面已经归一化
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    //高斯融合
    //对于Mat的ptr函数，返回的是<>中的模板类型指针，指向的是()中的第row行的起点 []是访问row行的某列元素
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

double NCC(const Mat &ref, const Mat &curr,
           const Vector2d &pt_ref,
           const Vector2d &pt_curr)
{
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    for(int x = -ncc_window_size; x <= ncc_window_size; x++)
    {
        for(int y = -ncc_window_size; y <= ncc_window_size; y++)
        {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1,0)))[int(x + pt_ref(0,0))]) / 255.0;
            mean_ref += value_ref;
            values_ref.push_back(value_ref);

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;
            values_curr.push_back(value_curr);
        }
    }
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 计算 Zero mean NCC
    double numerator1 = 0, numerator2 = 0, demoniator1 = 0, demoniator2 = 0;
    for(int i = 0; i < values_ref.size(); i++ )
    {
        double mul = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr); //去均值
        numerator1 += mul;//分子

        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }

    numerator2 = sqrt(demoniator1 * demoniator2 + 1e-10); //分母,1e-10防止坟墓出现零
    return numerator1 / numerator2;
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate)
{
    cout << "以深度图的平均误差和方差来评估一下估计的深度是否准确……" << endl;
    double depth_mu = 0, depth_squared = 0;
    int depth_nums = 0;
    for(int y = boarder; y < depth_truth.rows - boarder; y++)
    {
        for(int x = boarder; x < depth_truth.cols - boarder; x++)
        {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            depth_mu += error;
            depth_squared += error * error;
            depth_nums++;
        }
    }

    depth_mu /= depth_nums;
    depth_squared /= depth_nums;

    cout << "计算得到平均误差为： " << depth_mu << " 方差为： " << depth_squared << endl;

    //同时为了以后显示方案曲线，保存均值和方差值
    error_val.push_back(make_pair(depth_mu,depth_squared));
}

void plotDepth(const Mat &depth_truth, const Mat &depth_estimate)
{
    imshow("深度图真值：",depth_truth * 0.4);//深度值乘以0.4以后的结果表示--也就是纯白点（1）的深度约2.5米，颜色越深代表距离越近。也就是量程就是2.5米呗。
    imshow("估计的深度图：",depth_estimate * 0.4);
    imshow("深度误差图：",depth_truth - depth_estimate);
    waitKey(1);
}