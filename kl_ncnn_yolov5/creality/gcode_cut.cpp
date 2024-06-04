#include "gcode_cut.h"
class Gcode {
public:
    float minx, miny, minz, maxx, maxy, maxz;

    Gcode(float x1, float y1, float z1, float x2, float y2, float z2)
            : minx(x1), miny(y1), minz(z1), maxx(x2), maxy(y2), maxz(z2) {}
};

std::vector<float> parse_gcode(const std::string& file_path) {
    std::vector<float> g_range(6, 0.0);
    std::ifstream file(file_path);
    std::string line;
    while (getline(file, line)) {
        if (line.find(";MINX:") != std::string::npos)
            g_range[0] = stof(line.substr(6));
        else if (line.find(";MINY:") != std::string::npos)
            g_range[1] = stof(line.substr(6));
        else if (line.find(";MINZ:") != std::string::npos)
            g_range[2] = stof(line.substr(6));
        else if (line.find(";MAXX:") != std::string::npos)
            g_range[3] = stof(line.substr(6));
        else if (line.find(";MAXY:") != std::string::npos)
            g_range[4] = stof(line.substr(6));
        else if (line.find(";MAXZ:") != std::string::npos) {
            g_range[5] = stof(line.substr(6));
            break;
        }
    }
    return g_range;
}

void coefficients_read(const std::string& path_to_coefficients, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, cv::Mat& rvecs_pnp, cv::Mat& tvecs_pnp) {
    // cv::Mat cameraMatrix;
    // cv::Mat distCoeffs;
    // cv::Mat rvec;
    // cv::Mat tvec;
    // cv::Mat rvecs_pnp;
    // cv::Mat tvecs_pnp;
    int image_target_height;
    int image_target_width;
    std::vector<cv::Point2f> point_image_caliboard;

		std::vector<std::vector<cv::Point2f>> whold_point_cloud_ptr;
    cv::FileStorage fs(path_to_coefficients, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "File cannot be opened: " << path_to_coefficients << std::endl;
        return;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    // fs["extrinsic_parameters_r_0"] >> rvec1;
    // fs["extrinsic_parameters_t_0"] >> tvec1;
    fs["image_height"] >> image_target_height;
    fs["image_width"] >> image_target_width;
    cv::Mat first_image_point;
    fs["first_image_point"] >> first_image_point;
    for (int v = 0; v < first_image_point.rows; v++)
    {
        point_image_caliboard.emplace_back(cv::Point2f(first_image_point.at<double>(v, 0), first_image_point.at<double>(v, 1)));
    }
    fs.release();
    std::vector<cv::Point3f> point_cloud_caliboard;

    float length = 15;
    float offsetx = 25;
    float offsety = 47.5;

    for (int v = 0; v < 8; v++)
    {
        for (int u = 0; u < 11; u++)
        {
            point_cloud_caliboard.emplace_back(cv::Point3f(u * 15 + offsetx, v * 15 + offsety, 0));
        }
    }

    solvePnP(point_cloud_caliboard, point_image_caliboard, cameraMatrix, distCoeffs, rvecs_pnp, tvecs_pnp);
    // std::cout<<"rt:"<<rvecs_pnp<<" "<<tvecs_pnp<<std::endl;

}



cv::Point2f project_points(const std::vector<cv::Point3f>& points, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, projected_points);
    return projected_points[0];
}

// 获取分割矩形坐标
void getRect_by_gcode(const std::string& gcode_path,const std::string& camera_param_path,
            const float gcode_z,cv::Mat& img,cv::Point& top_left,cv::Point& bottom_right){
    // 解析Gcode文件
    std::vector<float> g_range = parse_gcode(gcode_path);
    Gcode g_p(g_range[0], g_range[1], g_range[2], g_range[3], g_range[4], g_range[5]);
    // std::cout<<"g_range:"<<g_range[0]<<" "<<g_range[1]<<" "<<g_range[2]<<" "<<g_range[3]<<" "<<g_range[4]<<" "<< g_range[5]<<" "<<std::endl;
    // 读取相机参数
    cv::Mat cameraMatrix, distCoeffs, rvec, tvec;
    coefficients_read(camera_param_path, cameraMatrix, distCoeffs, rvec, tvec);
    // std::cout<<"rt:"<<rvec<<" "<<tvec<<std::endl;
    tvec.at<double>(1, 0) = -tvec.at<double>(1, 0); // 对 tvec 进行修正

    // std::cout<<"g_range:"<<g_p.minx<<" "<<g_p.miny<<" "<<g_range[2]<<" "<<g_p.maxx<<" "<<g_p.maxy<<" "<< g_range[5]<<" "<<std::endl;
    // 投影点云到图像上
    // 获取图像尺寸
    int image_width = img.cols;
    int image_height = img.rows;
    cv::Point pt1,pt2;
    // 没有找到打印区域，默认将原图输入
    if (g_p.minx==0&&g_p.maxx==0 &&g_p.miny==0&&g_p.maxy==0){
        pt1=cv::Point(0, 0);
        pt2=cv::Point(image_width-1, image_height-1);
    }else{
        pt1 = project_points({cv::Point3i(g_p.minx, g_p.miny, gcode_z)}, rvec, tvec, cameraMatrix, distCoeffs);
        pt2 = project_points({cv::Point3i(g_p.maxx, g_p.maxy, gcode_z)}, rvec, tvec, cameraMatrix, distCoeffs);
    }

    int img_w = 640;
    int img_h = 320;
    int w_x = pt2.x - pt1.x;
    if (w_x < img_w){
        pt1.x = pt1.x - (img_w - w_x)*0.5;
        pt2.x = pt2.x + (img_w - w_x)*0.5;
    }
    int h_y = pt2.y;
    if (h_y < img_h){
        pt2.y= pt2.y + (img_h - h_y);
    }
    // 定义顶点坐标和右下角坐标
    top_left.x = pt1.x;  // 顶点坐标
    top_left.y = 0;  // 顶点坐标
    bottom_right = pt2;  // 右下角坐标

    
    // 确保点不超出图像尺寸
    top_left.x = std::max(0, std::min(top_left.x, image_width - 1));
    top_left.y = std::max(0, std::min(top_left.y, image_height - 1));
    bottom_right.x = std::max(0, std::min(bottom_right.x, image_width - 1));
    bottom_right.y = std::max(0, std::min(bottom_right.y, image_height - 1));
    // std::cout<<"pt1:"<<top_left<<" "<<bottom_right<<std::endl;
    // 裁剪图像
    cv::Rect roi(top_left, bottom_right);
    img = img(roi);
    // cv::imwrite("./img_roi.jpg",img);
}
//warp
// void getRect_by_gcode_warp(const std::string& gcode_path,const std::string& camera_param_path,
//             const float gcode_z,cv::Mat& img,cv::Point& top_left,cv::Point& bottom_right){
//     // 解析Gcode文件
//     std::vector<float> g_range = parse_gcode(gcode_path);
//     Gcode g_p(g_range[0], g_range[1], g_range[2], g_range[3], g_range[4], g_range[5]);
//     // std::cout<<"g_range:"<<g_range[0]<<" "<<g_range[1]<<" "<<g_range[2]<<" "<<g_range[3]<<" "<<g_range[4]<<" "<< g_range[5]<<" "<<std::endl;
//     // 读取相机参数
//     cv::Mat cameraMatrix, distCoeffs, rvec, tvec;
//     coefficients_read(camera_param_path, cameraMatrix, distCoeffs, rvec, tvec);
//     // std::cout<<"rt:"<<rvec<<" "<<tvec<<std::endl;
//     tvec.at<double>(1, 0) = -tvec.at<double>(1, 0); // 对 tvec 进行修正

//     // std::cout<<"g_range:"<<g_p.minx<<" "<<g_p.miny<<" "<<g_range[2]<<" "<<g_p.maxx<<" "<<g_p.maxy<<" "<< g_range[5]<<" "<<std::endl;
//     // 投影点云到图像上
//     // 获取图像尺寸
//     int image_width = img.cols;
//     int image_height = img.rows;
//     cv::Point pt1,pt2;
//     // 没有找到打印区域，默认将原图输入
//     if (g_p.minx==0&&g_p.maxx==0 &&g_p.miny==0&&g_p.maxy==0){
//         pt1=cv::Point(0, 0);
//         pt2=cv::Point(image_width-1, image_height-1);
//     }else{
//         pt1 = project_points({cv::Point3i(g_p.minx, g_p.miny, gcode_z)}, rvec, tvec, cameraMatrix, distCoeffs);
//         pt2 = project_points({cv::Point3i(g_p.maxx, g_p.maxy, gcode_z)}, rvec, tvec, cameraMatrix, distCoeffs);
//     }

//     int img_w = 640;
//     int img_h = 320;
//     int w_x = pt2.x - pt1.x;
//     if (w_x < img_w){
//         pt1.x = pt1.x - (img_w - w_x)*0.7;
//         pt2.x = pt2.x + (img_w - w_x)*0.7;
//     }
//     int h_y = pt2.y;
//     if (h_y < img_h){
//         pt1.y= pt1.y + (img_h - h_y)*0.2;
//         pt2.y= pt2.y + (img_h - h_y)*0.7;

//     }
//     // 定义顶点坐标和右下角坐标
//     top_left.x = pt1.x;  // 顶点坐标
//     top_left.y = 0;  // 顶点坐标
//     bottom_right = pt2;  // 右下角坐标

    
//     // 确保点不超出图像尺寸
//     top_left.x = std::max(0, std::min(top_left.x, image_width - 1));
//     top_left.y = std::max(0, std::min(top_left.y, image_height - 1));
//     bottom_right.x = std::max(0, std::min(bottom_right.x, image_width - 1));
//     bottom_right.y = std::max(0, std::min(bottom_right.y, image_height - 1));
//     // std::cout<<"pt1:"<<top_left<<" "<<bottom_right<<std::endl;
//     // 裁剪图像
//     cv::Rect roi(top_left, bottom_right);
//     img = img(roi);
//     // cv::imwrite("./img_roi.jpg",img);
// }

