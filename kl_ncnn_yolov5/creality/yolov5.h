/*** 
 * @Author: chen hong 1634280408@qq.com
 * @Date: 2023-11-21 16:30:29
 * @LastEditTime: 2023-11-29 11:59:40
 * @LastEditors: chen hong 1634280408@qq.com
 * @Description: 
 * @FilePath: /ncnn_yolov5/creality/yolov5.h
 * @Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
 */
#ifndef _YOLOV5_H_
#define _YOLOV5_H_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#ifdef _cplusplus
extern "C"{
#endif

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects, const char* mode);
void draw_objects(const cv::Mat & bgr, const std::vector<Object>& objects, const char* savepath, const char* mode);
bool no_extrusion_obj_detect(const std::vector<Object>& objects,const std::vector<const char*> class_names);


struct Results {
    int re_label;
    float re_prob;
    float re_obj_rect_x;
    float re_obj_rect_y;
    float re_obj_rect_width;
    float re_obj_rect_height;
  };

void get_results (const cv::Mat & bgr,const std::vector<Object>& objects, struct Results res[], int* cnt, const char* mode);

#ifdef _cplusplus
  }
#endif

#endif /* _YOLOV5_H_ */
