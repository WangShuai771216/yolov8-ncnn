// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include <map>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <map>
//#include <filesystem>
#include <numeric>
#include <iterator>
//#include <Eigen/Dense>
using namespace std;
using namespace cv;

#define MAX_STRIDE 64

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    float scores;
};

struct Results
{
    int re_label;
    float score;
    float re_obj_rect_x;
    float re_obj_rect_y;
    float re_obj_rect_width;
    float re_obj_rect_height;
};

struct Image
{
    std::string name;
    int ind; // 图片索引
    double x;
    float score;
    int index; // 流量角的索引

    Image(const std::string& n, int i, double xPos, float s, int indx)
        : name(n), ind(i), x(xPos), score(s), index(indx)
    {
    }
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);

            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    ;
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5 - 1;

    const int feat_offset = num_class + 5;
    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                // combined score = box score * class score
                // apply sigmoid first to get normed 0~1 value
                float confidence = sigmoid(box_score) * sigmoid(class_score);

                // filter candidate boxes with combined score >= prob_threshold
                if (confidence < prob_threshold)
                    continue;

                float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                float cx = (dx * 2.f - 0.5f + j) * stride;
                float cy = (dy * 2.f - 0.5f + i) * stride;

                float bw = pow(dw * 2.f, 2) * anchor_w;
                float bh = pow(dh * 2.f, 2) * anchor_h;

                // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
                float x0 = cx - bw * 0.5f;
                float y0 = cy - bh * 0.5f;
                float x1 = cx + bw * 0.5f;
                float y1 = cy + bh * 0.5f;

                float sc = sigmoid(feat_blob.channel(3 * feat_offset + q).row(i)[j]);

                // collect candidates
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = confidence;
                obj.scores = sc;
                objects.push_back(obj);
            }
        }
    }
}

static int detect_yolov5_dynamic(const cv::Mat& bgr, std::vector<Object>& objects,const char* type)
{
    ncnn::Net yolov5;
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // windows无法使用绑定大小核
    // ncnn::set_cpu_powersave(1); // bind to little cores
    // ncnn::set_cpu_powersave(2); // bind to big cores
    // ncnn::set_cpu_powersave(0); // 0:全部 1:小核 2:大核
    yolov5.opt.lightmode = true;                 // 启用时，中间的blob将被回收。默认情况下启用
    yolov5.opt.num_threads = 1;                  //  线程数  默认值是由get_cpu_count()返回的值。openmp关闭之后,无法法用
    yolov5.opt.use_packing_layout = false;       // improve all operator performance on all arm devices, will consume more memory
    yolov5.opt.openmp_blocktime = 0;             // openmp线程在进入睡眠状态前忙着等待更多工作的时间 默认值为20ms，以保持内核的正常工作。而不至于在之后有太多的额外功耗，openmp关闭之后,无法法用
    yolov5.opt.use_winograd_convolution = false; // improve convolution 3x3 stride1 performance, may consume more memory
    yolov5.opt.use_sgemm_convolution = false;    // improve convolution 1x1 stride1 performance, may consume more memory
    yolov5.opt.use_int8_inference = true;        // 启用量化的int8推理 对量化模型使用低精度的int8路径 默认情况下启用
    yolov5.opt.use_vulkan_compute = false;       // 使用gpu进行推理计算
    yolov5.opt.use_bf16_storage = false;         // improve most operator performance on all arm devices, may consume more memory
    yolov5.opt.use_packing_layout = false;       //  enable simd-friendly packed memory layout,  improve all operator performance on all arm devices, will consume more memory, enabled by default
    //----------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef CROSS_COMPILE
    if (strcmp(type,"k1")==0){
        yolov5.load_param("./best_ll.param");
        yolov5.load_model("./best_ll.bin");
    }else if (strcmp(type,"F008")==0){
        yolov5.load_param("./best-ll.param");
        yolov5.load_model("./best-ll.bin");
    }else{
        std::cerr << "type错误,程序中断！" << std::endl;
        std::abort();
        return 0;
    }
#else
     if (strcmp(type, "k1") == 0) {
        yolov5.load_param("/usr/lib/yolov5/best_ll.param");
        yolov5.load_model("/usr/lib/yolov5/best_ll.bin");
    } else if (strcmp(type, "F008") == 0) {
        yolov5.load_param("/usr/lib/yolov5/best-ll.param");
        yolov5.load_model("/usr/lib/yolov5/best-ll.bin");
    }else{
        std::cerr << "type错误,程序中断！" << std::endl;
        std::abort();
        return 0;
    }
#endif

    const int target_size = 320;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    // load image, resize and letterbox pad to multiple of max_stride
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    const int max_stride = 64;

    // solve resize scale
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    const int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    const int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;

    ex.extract("out0", out0);
    ex.extract("out1", out1);
    ex.extract("out2", out2);
    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out0, prob_threshold, objects8);
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out1, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out2, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // collect final result after nms
    const int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    fprintf(stdout, "Detect over\n");
    return 0;
};

bool compareByX(const Object& obj1, const Object& obj2)
{
    return obj1.rect.x < obj2.rect.x;
};

static void draw_objects(const cv::Mat& bgr, std::vector<Object>& objects, const char* savepath,const char* type)
{
    static const char* class_names[] = {
        "rect"};

    // 保存绘制图片名称
    std::string name = savepath;
    name = name.insert(name.size() - 4, "_draw");

    // 替换jpg为txt
    std::string txt_name = savepath;
    std::string replaceStr = ".txt";
    size_t pos = txt_name.find(".jpg");
    if (pos != std::string::npos)
    {
        txt_name.replace(pos, 4, replaceStr);
    }
    // 查找倒数第一个下划线的位置
    std::string insertStr = "labels/";
    size_t pos1 = txt_name.rfind("/");
    if (pos1 != std::string::npos)
    {
        // 在下划线位置之后插入指定字符串
        txt_name.insert(pos1 + 1, insertStr);
    }

    cout << txt_name << endl;
    time_t nowtime = time(NULL);
    struct tm* p;
    p = localtime(&nowtime);
    char tmp[192];

    cv::Mat image = bgr.clone();
    int f = 0;

    // 使用自定义的比较函数按照rect的x值进行排序r45_2141698819131.5742743
    std::sort(objects.begin(), objects.end(), compareByX);

    ////根据宽、间隙，去除冗余框
    int rect_width=150;
    int gap=70;
    // if (strcmp(type,"k1")==0){
    //     rect_width = 150;
    //     gap = 50;
    // }else{
    //     rect_width = 70;
    //     gap = 29;
    // }

    // 删除满足条件的元素
    objects.erase(std::remove_if(objects.begin(), objects.end(), [rect_width](const Object& obj) { return obj.rect.width > rect_width; }),
                  objects.end());

    // 处理相邻元素差值小于30的情况
    for (size_t i = 1; i < objects.size(); ++i)
    {
        if ((objects[i].rect.x - objects[i - 1].rect.x) < gap)
        {
            if (objects[i].prob > objects[i-1].prob){
                objects.erase(objects.begin() + i-1);
                --i;
            }else{
                objects.erase(objects.begin() + i);
                --i;
            }
        }
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d =%.5f %.6f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.scores, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        char tmp123[192];
        std::ofstream location_out;
        //          sprintf(tmp123, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d-%d_raw.txt", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        sprintf(tmp123, "%s", txt_name.c_str());
        location_out.open(tmp123, std::ios::out | std::ios::app); // 以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
        // location_out.open(tmp123, std::ios::out); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
        location_out << obj.label << " " << (obj.rect.x + obj.rect.width * 0.5) / 1600 << " " << (obj.rect.y + obj.rect.height * 0.5) / 1200 << " " << obj.rect.width / 1600 << " " << obj.rect.height / 1200 << " " << obj.scores << "\n";
        location_out.close();

        cv::rectangle(image, obj.rect, cv::Scalar(0, 0, 255));

        char text[256];
        sprintf(text, "%s %.6f %.1f%%", class_names[obj.label], obj.scores, obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(0, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    };

    // cv::imshow("image", image);
    // cv::waitKey(0);
    cv::imwrite(name, image);
}

std::vector<Results> parseDetectionResults(const std::string& filePath)
{
    std::vector<Results> result;

    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return result;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        Results res;
        iss >> res.re_label >> res.re_obj_rect_x >> res.re_obj_rect_y >> res.re_obj_rect_width >> res.re_obj_rect_height >> res.score;
        // cout<<res.re_label <<" "<< res.re_obj_rect_x << " "<<res.re_obj_rect_y << " "<<res.re_obj_rect_width <<" "<<res.re_obj_rect_height<< " "<<res.score<<endl;
        result.push_back(res);
    }

    file.close();
    return result;
}

void saveMaxValues(const std::vector<Results>& objects, float& maxLabel0, float& maxLabel1, float& maxX0, float& maxY0, float& maxX1, float& maxY1, int& index0, int& index1)
{
    maxX0 = 0.0f;
    maxY0 = 0.0f;
    maxX1 = 0.0f;
    maxY1 = 0.0f;
    index0 = 0;
    index1 = 0;
    for (const Results& obj : objects)
    {
        if (obj.re_label == 0 && obj.score > maxLabel0)
        {
            maxLabel0 = obj.score;
            maxX0 = obj.re_obj_rect_x;
            maxY0 = obj.re_obj_rect_y;
            index0 = &obj - &objects[0];
        }
        else if (obj.re_label == 1 && obj.score > maxLabel1)
        {
            maxLabel1 = obj.score;
            maxX1 = obj.re_obj_rect_x;
            maxY1 = obj.re_obj_rect_y;
            index1 = &obj - &objects[0];
        }
    }
}

Image findImageWithMaxScore(const std::vector<Image>& images)
{
    // if (images.empty()) {
    //     throw std::runtime_error("Empty vector of images.");
    // }
    Image maxImage = images[0];
    for (const auto& image : images)
    {
        if (image.score > maxImage.score)
        {
            maxImage = image;
        }
    }

    return maxImage;
}

// void get_results(const char *devicepath)
// {
//     static const char *class_names[] = {
//         "actue", "rect"};

//     cv::Mat image = bgr.clone();
//     *cnt = objects.size();

//     if (*cnt > 20)
//     {
//         cout << "picture count is too many: " << *cnt << endl;
//         *cnt = 20;
//     }

//     for (size_t i = 0; i < *cnt; i++)
//     {
//         const Object &obj = objects[i];

//         res[i].re_label = obj.label;
//         res[i].re_prob = obj.prob;
//         res[i].score = obj.scores;
//         res[i].re_obj_rect_x = obj.rect.x;
//         res[i].re_obj_rect_y = obj.rect.y;
//         res[i].re_obj_rect_width = obj.rect.width;
//         res[i].re_obj_rect_height = obj.rect.height;
//     }
// }

std::map<std::string, int> my_dict = {{"38", 0}, {"63", 1}, {"88", 2}, {"113", 3}, {"138", 4}};
std::map<std::string, int> my_dict2 = {{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}};

/// Define Gaussian function
double gaussian(double x, double a, double b, double c)
{
    return a * exp(-(x - b) * (x - b) / (2 * c * c));
}

int GaussianFitFunSolve(std::vector<double>& x_data, std::vector<double>& y_data, double& midpoint,const char* type)
{
    if (y_data.size() < 3)
    {
        return -1;
    }
    if (strcmp(type,"k1")==0){
        std::reverse(y_data.begin(), y_data.end());
    }else if(strcmp(type,"F008")==0){
        std::sort(y_data.begin(), y_data.end());
    }

    // std::cout << "x data :" << std::endl;
    for (int i = 0; i < y_data.size(); i++)
    {
        x_data.push_back(i);
        // std::cout << i << ' ';
    }
    // std::cout << std::endl;
    // std::cout << "------------------------" << std::endl;

    // ��ӡ��ȡ�������ݼ���Ƿ���ȷ
    // std::cout << "y data :" << std::endl;
    // for (const auto& num : y_data)
    // {
    //     std::cout << num << ' ';
    // }
    // std::cout << std::endl;

    // // ������ƾ���A��Ŀ������b
    // int n = x_data.size();
    // Eigen::MatrixXd A(n, 2);
    // Eigen::VectorXd b(n);
    // for (int i = 0; i < n; i++)
    // {
    //     A(i, 0) = 1.0;
    //     A(i, 1) = x_data[i];
    //     // A(i, 2) = x_data[i] * x_data[i];
    //     b(i) = ln(y_data[i]) + 0.02 * x_data[i] * x_data[i];
    // }

    int n = x_data.size();
    double sigma = 3;
    double s = 2 * sigma * sigma;
    cv::Mat A = cv::Mat::zeros(n, 2, CV_64FC1);
    cv::Mat B = cv::Mat::zeros(n, 1, CV_64FC1);
    for (int i = 0; i < n; i++)
    {
        A.at<double>(i, 0) = 1.0;
        A.at<double>(i, 1) = x_data[i];
        B.at<double>(i, 0) = log(y_data[i]) + x_data[i] * x_data[i] / s;
    }
    cv::Mat C = cv::Mat::zeros(2, 1, CV_64FC1);

    // std::cout << "A" << std::endl
    //           << A << std::endl;
    // std::cout << "B" << std::endl
    //           << B << std::endl;

    cv::solve(A, B, C, cv::DECOMP_QR);
    double mu = C.at<double>(1) * s / 2.0;
    // std::cout << "midpoint orignin = " << mu << std::endl;

    if (mu < 0)
        mu = 0;
    if (mu >= n)
        mu = n - 1;

    // �����˹���ߵĲ���
    // Eigen::VectorXd x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    // double mu = -x(1) / (2 * x(2));
    // double mu = x(1) * 25.0;
    // double sigma = sqrt(-1 / (2 * x(2)));
    // double amplitude = x(0);

    //// �����Ͻ��
    // cout << "mu = " << mu << ", sigma = " << sigma << ", amplitude = " << amplitude << endl;

    // �����е�
    midpoint = mu;

    // ����е�
    // std::cout << "midpoint = " << midpoint << std::endl;
    return 0;
}

int GaussianFitFun(std::vector<double>& x_data, std::vector<double>& y_data, double& midpoint)
{
    double sum_x = 0;
    double sum_value = 0;
    // std::cout << "x data :" << std::endl;

    std::reverse(y_data.begin(), y_data.end());

    for (int i = 0; i < y_data.size(); i++)
    {
        x_data.push_back(i);
        sum_x += double(i) * y_data[i] * y_data[i];
        sum_value += y_data[i] * y_data[i];
        // std::cout << i << '-' << y_data[i] << ' ';
    }
    // std::cout << std::endl;
    // std::cout << "------------------------" << std::endl;
    if (sum_x > 0)
    {
        midpoint = sum_x / sum_value;
    }
    else
    {
        midpoint = 2;
    }

    // std::cout << "sum_x = " << sum_x << std::endl;
    // std::cout << "sum_value = " << sum_value << std::endl;
    // std::cout << "midpoint = " << midpoint << std::endl;
    return 0;
}

std::string get_substring(std::string s)
{
    size_t start = s.find("FlowIrImage") + 11; // Find the position of the first '_' and add 1 to get the position of the next character
    // std::cout << "start :" << start << std::endl;
    size_t end = s.find(".txt", start); // Find the position of the second '_' starting from the 'start' position
    // std::cout << "end :" << end << std::endl;

    if (end == std::string::npos)
    {
        return "-1";
    }
    return s.substr(start, end - start); // Extract the substring between the two '_'
}

// std::string get_substring(std::string s)
// {
//     size_t start = s.find('_') + 1;  // Find the position of the first '_' and add 1 to get the position of the next character
//     size_t end = s.find('_', start); // Find the position of the second '_' starting from the 'start' position
//     if (end == std::string::npos)
//     {
//         return "-1";
//     }
//     return s.substr(start, end - start); // Extract the substring between the two '_'
// }

void get_GaussianFitFun_results(const char* devicepath,const char* type)
{
    std::string folder_path = devicepath;
    folder_path = folder_path + "labels/";
    // std::cout <<folder_path<<std::endl;
    // Get all txt files in the folder
    std::vector<std::string> txt_files;
    DIR* dir = opendir(folder_path.c_str());
    if (dir == nullptr)
    {
        std::cerr << "无法打开目录：" << folder_path << std::endl;
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL)
    {
        // 跳过'.'和'..'两个目录
        if (entry->d_name[0] == '.')
            continue;
        // 获取文件后缀名
        char* dot = strrchr(entry->d_name, '.');
        if (dot && strcmp(dot, ".txt") == 0)
        {
            // 如果是 .txt 文件，则打印文件名
            //std::cout << entry->d_name << std::endl;
            txt_files.push_back(folder_path + entry->d_name);
        }
    }
    closedir(dir);

    std::vector<std::string> sorted_txt_files(txt_files.size());
    for (const auto& txt_file : txt_files)
    {
        // std::cout << "txt_file :" << txt_file << std::endl;
        std::string substring = get_substring(txt_file);
        // std::cout << "substring :" << substring << std::endl;
        if (substring != "-1")
        {
            // std::cout << "my_dict2 index :" << my_dict2[substring] << std::endl;
            // int position = my_dict2[substring];
            int position = my_dict2[substring];
            sorted_txt_files[position] = txt_file;
        }
    }

    std::vector<std::vector<double> > total_data;
    std::vector<double> y_data;
    for (const auto& txt_file : sorted_txt_files)
    {
        std::vector<double> arrays_w;

        std::ifstream file(txt_file);
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::vector<double> values(std::istream_iterator<double>{iss}, std::istream_iterator<double>());
            arrays_w.push_back(values[5]);
        }
        file.close();

        total_data.push_back(arrays_w);
        double mean = 0;
        if (!arrays_w.empty())
        {
            mean = std::accumulate(arrays_w.begin(), arrays_w.end(), 0.0) / arrays_w.size();
        }
        // std::cout << txt_file << ": array = ";
        // for (const auto &value : arrays_w)
        // {
        //     std::cout << value << " ";
        // }
        // std::cout << "---- mean = " << mean << std::endl;

        y_data.push_back(mean);
    }

    auto max_iter = std::max_element(y_data.begin(), y_data.end());
    if (*max_iter <= 0)
    {
        std::cout << " ERROR INPUT !!" << std::endl;
        return;
    }

    if (*max_iter < 0.3)
    {
        std::cout << " ERROR Max value !!" << std::endl;
        return;
    }

    double mean = std::distance(y_data.begin(), max_iter);
    std::cout << "mean_index = " << mean << std::endl;

    if (mean < 0 || mean >= 5)
    {
        std::cout << " ERROR MEAN list !!" << std::endl;
        return;
    }

    // std::vector<double> data;
    // double mean;
    // GaussianFitFun(data, y_data, mean);

    // int mean_int = std::floor(mean + 0.5);
    // std::cout << "mean_int = " << mean_int << std::endl;

    std::vector<double> data2;
    double mean2;
    if (GaussianFitFunSolve(data2, total_data[mean], mean2,type) < 0)
    {
        std::cout << "End Position = -1" << std::endl;
        return;
    }
    std::cout << "mean2 = " << mean2 << std::endl;
    // GaussianFitFun(data2, total_data[mean], mean2);

    double end_mean = (static_cast<double>(mean) * 4 * 0.004) + 0.02 + std::round(mean2) * 0.004;

    std::cout << "End Position = " << end_mean << std::endl;

    //std::cout << "Done!" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    cv::Mat frame;
    const char* devicepath = argv[1];
    const char* savepath = argv[2];
    const char* type = argv[3];
    std::vector<Object> objects;

    // local img ----------------------------------------------------------------------------------------
    if (strstr(devicepath, ".jpg") && !strstr(devicepath, "http"))
    {
        frame = cv::imread(devicepath, 1);
        detect_yolov5_dynamic(frame, objects,type);
        draw_objects(frame, objects, savepath,type);
    }

    // local img-dir ----------------------------------------------------------------------------------------
    if (strstr(devicepath, "img"))
    {
        // std::vector<std::string> subdirectories = getSubdirectories(devicepath);
        // for (const auto& subdirectory : subdirectories) {
        std::vector<cv::String> fn;
        cv::glob(devicepath, fn, false);
        int count = fn.size(); // number of png files in images folder
        for (int i = 0; i < count; i++)
        {
            cout << fn[i].c_str() << endl;
            frame = cv::imread(fn[i], 1);
            // int width = frame.cols;ss
            // // 定义裁剪区域
            // cv::Rect roi(200, 0, width - 400, frame.rows);

            // // 裁剪图像
            // cv::Mat frame1 = frame(roi);
            detect_yolov5_dynamic(frame, objects,type);
            draw_objects(frame, objects, fn[i].c_str(),type);
            fprintf(stderr, "%d / %d \n", i, count);
        }
        // get_results(devicepath);
        get_GaussianFitFun_results(devicepath,type);
    }
    return 0;
}
