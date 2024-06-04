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
    float scores;
};

int detect_yolov5_dynamic(const cv::Mat& bgr, std::vector<Object>& objects);
void draw_objects(const cv::Mat & bgr, const std::vector<Object>& objects, const char* savepath);


struct Results {
    int re_label;
    float re_prob;
    float score;
    float re_obj_rect_x;
    float re_obj_rect_y;
    float re_obj_rect_width;
    float re_obj_rect_height;
  };

void get_results (const cv::Mat & bgr,const std::vector<Object>& objects, struct Results res[], int* cnt);

#ifdef _cplusplus
  }
#endif

#endif /* _YOLOV5_H_ */
