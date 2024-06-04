#ifndef _GCODE_CUT_H_
#define _GCODE_CUT_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>


#ifdef _cplusplus
extern "C"{
#endif


void getRect_by_gcode(const std::string& gcode_path,
        const std::string& camera_param_path,const float gcode_z,
        cv::Mat& img,cv::Point& top_left,cv::Point& bottom_right);
void getRect_by_gcode_warp(const std::string& gcode_path,
        const std::string& camera_param_path,const float gcode_z,
        cv::Mat& img,cv::Point& top_left,cv::Point& bottom_right);

#ifdef _cplusplus
  }
#endif

#endif /* _GCODE_CUT_H_ */
