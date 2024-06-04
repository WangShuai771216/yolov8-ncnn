#ifndef _AI_PROPERTY_H_
#define _AI_PROPERTY_H_

#include <iostream>
#include <fstream>
#include <json-c/json.h>
#include <vector>
#include <cstring>
#include <sstream> 
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include "yolov5.h"
#include <chrono>
// 自定义结构体表示标签
struct Label {
    std::string cls_name;
    float confidence[2];

};

// 自定义结构体表示模型
struct Model {
    std::string name;
    std::vector<Label> labels;
};

// 自定义结构体表示设备配置
struct DeviceConfig {
    std::string device;
    std::string min_version;
    std::string max_version;    
    std::vector<Model> models;
    std::string json_path;
};


// bool AI_property_parse(std::string json_path);
// // 递归函数，解析JSON对象
// void parseJsonObject(json_object *jobj, DeviceConfig& deviceConfig);
// void createAndWriteFile(const std::string& filePath, const std::string& content);
// bool createDirectories(const std::string& path);
void ai_inference_result_decide(const std::string device,const std::string device_vision,
                                const char* mode,std::vector<Object> objects,cv::Mat m);

#endif /* _AI_PROPERTY_H_ */