/*** 
 * @Author: chen hong 1634280408@qq.com
 * @Date: 2023-11-21 16:30:29
 * @LastEditTime: 2023-12-25 12:01:24
 * @LastEditors: chen hong 1634280408@qq.com
 * @Description: 
 * @FilePath: /ncnn_yolov5/creality/detection.cpp
 * @Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
 */
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "yolov5.h"
#include "gcode_cut.h"
#include "AI_property.h"

using namespace cv;

int main(int argc, char** argv)
{
    
    // if (argc != 4)
    if(argc != 4 and argc != 6 )
    {
        fprintf(stderr, "Usage: %s [imagepath] [savepath] [mode] \n", argv[0]);
        fprintf(stderr, "or Usage: %s [imagepath] [savepath] [mode] [gcode_file] [z_height]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* savepath = argv[2];
    const char* mode = argv[3];

    
    cv::Mat m = cv::imread(imagepath, 1);
    cv::Mat m_src = m.clone();
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

     // 包裹检测,先对图像进行裁减
    if(std::string(mode).find("blob") != std::string::npos &&  argc == 6){
        
        const char* gcode_path = argv[4];
        const char* z_height = argv[5];
        const float z_h = atof(z_height);
#ifndef CROSS_COMPILE
        const char* cam_path = "../blob_opt/parameter.xml";
#else
        const char* cam_path = "/usr/lib/yolov5/parameter.xml";
#endif
        cv::Point top_left,bottom_right;
        // 创建图像
        // std::cout<<"blob cut images!!"<<std::endl;
        cv::Mat img_c = m.clone();
       
        //                gcode路径   腔体相机参数  当前大概层高   裁减图像   左上角     右下角
        getRect_by_gcode(gcode_path,  cam_path,    z_h,        img_c, top_left, bottom_right);
        // getRect_by_gcode("../blob_opt/gcode/202401306.gcode","../blob_opt/parameter.xml",     0.2,         img_c  , top_left, bottom_right );
        // std::cout<<"top_p:"<<top_left<< " " <<bottom_right<<std::endl;
        if (!img_c.empty())
        {
            m = img_c.clone();
        }
    }
    //warp
//     else if(std::string(mode).find("warp") != std::string::npos &&  argc == 6){
        
//         const char* gcode_path = argv[4];
//         const char* z_height = argv[5];
//         const float z_h = atof(z_height);
// #ifndef CROSS_COMPILE
//         const char* cam_path = "../blob_opt/parameter.xml";
// #else
//         const char* cam_path = "/usr/lib/yolov5/parameter.xml";
// #endif
//         cv::Point top_left,bottom_right;
//         // 创建图像
//         // std::cout<<"blob cut images!!"<<std::endl;
//         cv::Mat img_c = m.clone();
       
//         //                      gcode路径   腔体相机参数  当前大概层高   裁减图像   左上角     右下角
//         getRect_by_gcode_warp(gcode_path,  cam_path,    z_h,        img_c, top_left, bottom_right);
//         // getRect_by_gcode("../blob_opt/gcode/202401306.gcode","../blob_opt/parameter.xml",     0.2,         img_c  , top_left, bottom_right );
//         // std::cout<<"top_p:"<<top_left<< " " <<bottom_right<<std::endl;
//         if (!img_c.empty())
//         {
//             m = img_c.clone();
//         }
//     }

    std::vector<Object> objects;
    detect_yolov5(m, objects, mode);
    draw_objects(m, objects, savepath, mode);

     // 设备参数
    std::string device = "K1_MAX";// 此处修改device参数      
    std::string device_vision = "1.0.1";
    ai_inference_result_decide(device, device_vision,mode,objects,m_src);//AI结果判断

   

    return 0;
}
