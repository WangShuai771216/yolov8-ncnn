// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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


#include <thread>
#include "cpu.h"  //绑定大小核

#include <ctime>  // 获取当前时间


#include <fstream>
#include <iostream>
#include <string>


#include <unistd.h>  // time.sleep


#include <algorithm>    // std::find

#include "layer.h"
#include "net.h"

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




#define NCNN_PROFILING

#define YOLOV5_V60 1 //YOLOv5 v6.0



#if YOLOV5_V60
#define MAX_STRIDE 64
#else
#define MAX_STRIDE 32
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)



#endif //YOLOV5_V60



#ifdef NCNN_PROFILING
#include "benchmark.h"
#endif


double fps;


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
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
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
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
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);
                    if (confidence >= prob_threshold)
                    {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects, 
            ncnn::Net* yolov5, const char* threshold, const char* input_size)
{

//    const float prob_threshold = 0.25f;
    const float prob_threshold = atof(threshold);
//    const float prob_threshold = 0.62f;
    const float nms_threshold = 0.45f;
    int target_size = atof(input_size);
//    int target_size = 640;
//    int target_size = 320;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
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

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2GRAY, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5->create_extractor();
    //-----------------------------------------------------------------------------------------------
//    ex.set_light_mode(true);
//    ex.set_num_threads(1);  //openmp关闭之后就是否使用线程控制
    //-----------------------------------------------------------------------------------------------



    ex.input("images", in_pad);


    //-----------------------------------------------------------------------------
//    int delay_time = 1000;
//    int delay_time = 500000;
////    float delay_time = 0.4;
////    float delay_time = 0.3;
////    float delay_time = 0.2;
////    float delay_time = 0.1;
//
//    ncnn::Mat conv1;
//
//    //    ex.extract("conv0", conv1);
//    ex.extract("122", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv3", conv1);
//    ex.extract("125", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv12", conv1);
//    ex.extract("134", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv23", conv1);
//    ex.extract("145", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv32", conv1);
//    ex.extract("154", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv39", conv1);
//    ex.extract("161", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv50", conv1);
//    ex.extract("172", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv59", conv1);
//    ex.extract("181", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv66", conv1);
//    ex.extract("188", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv73", conv1);
//    ex.extract("195", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv84", conv1);
//    ex.extract("206", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv93", conv1);
//    ex.extract("215", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv129", conv1);
//    ex.extract("251", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv154", conv1);
//    ex.extract("276", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv164", conv1);
//    ex.extract("286", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv174", conv1);
//    ex.extract("296", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv184", conv1);
//    ex.extract("306", conv1);
//    usleep(delay_time);
//    //    ex.extract("conv194", conv1);
//    ex.extract("316", conv1);
//    usleep(delay_time);
    //--------------------------------------------------
//
//




    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
        // std::cout<<"object8:"<<objects8.size()<<std::endl;

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        // std::cout<<"anchors 8"<< std::endl;
    }

    // stride 16
    {
        ncnn::Mat out;

        ex.extract("353", out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        // std::cout<<"objects16:"<<objects16.size()<<std::endl;
    }

    // stride 32
    {
        ncnn::Mat out;

        ex.extract("367", out);

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        // std::cout<<"anchors 32"<< std::endl;
    }


    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

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

    return 0;
}
//extern const char* class_names[10];

static int draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, int is_streaming,
                        const char* ifsave, const char* ifshow, const char* mode, int inum, const char* model_param_path,
                        const char* filestr=nullptr)
{

    
    std::string model_param_path_str(model_param_path);
    
    
    std::vector<const char*> class_names; // 使用 std::vector 替代 C 数组
    if (std::string(model_param_path_str).find("ywjc") != std::string::npos) {
        class_names = {"foreign_object"}; // 初始化数组
    }
    else if (std::string(model_param_path_str).find("-xkjc") != std::string::npos) {
        class_names = {"no-extrusion"}; // 初始化数组   err_gap no-extrusion
    }
    else if (std::string(model_param_path_str).find("-part") != std::string::npos) {
        class_names = {"part"}; // 初始化数组   err_gap no-extrusion
    }
    else if (std::string(model_param_path_str).find("-nozzle") != std::string::npos) {
        class_names = {"nozzle"}; // 初始化数组
    }
    else{
        class_names = {"nozzle-blob", "spaghetti", "stringing"}; // 初始化数组
    }



//    time_t nowtime = time(NULL);
//    struct tm *p;
//    p = gmtime(&nowtime);
//    char tmp[192];

    time_t nowtime = time(NULL);
    struct tm *p;
    p = localtime(&nowtime);
    char tmp[192];
//    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", p);
//    printf("%s\n", tmp);


//    std::string s_file_save;
    cv::Mat image = bgr.clone();
//    std::string cls_label;

    
    if (strcmp(ifsave, "save") == 0){
        for (size_t i = 0; i < objects.size(); i++){
            const Object& obj = objects[i];
            float x1 = obj.rect.x;
            float y1 = obj.rect.y;
            float x2 = obj.rect.x + obj.rect.width;
            float y2 = obj.rect.y + obj.rect.height;
            float center_x = ((x1+x2)*0.5) / image.cols;
            float center_y = ((y1+y2)*0.5) / image.rows;
            float center_w = (abs((x2-x1))*1.0) / image.cols;
            float center_h = (abs((y2-y1))*1.0) / image.rows;
            std::string cls_label = class_names[obj.label];
            char tmp123[192];
            std::ofstream location_out;
//            sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.txt",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d_raw.txt", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
//            location_out.open(tmp123, std::ios::out | std::ios::app);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
            location_out.open(tmp123, std::ios::out);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
            location_out <<cls_label<<" "<<center_x<<" "<<center_y<<" "<<center_w<<" "<<center_h<<"\n";
            location_out.close();

            if(obj.label==0 || obj.label==1 || obj.label==2){
//                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);}

        }
    }

    if (strcmp(ifsave, "saveall") == 0){
        for (size_t i = 0; i < objects.size(); i++){
            const Object& obj = objects[i];
            float x1 = obj.rect.x;
            float y1 = obj.rect.y;
            float x2 = obj.rect.x + obj.rect.width;
            float y2 = obj.rect.y + obj.rect.height;
            float center_x = ((x1+x2)*0.5) / image.cols;
            float center_y = ((y1+y2)*0.5) / image.rows;
            float center_w = (abs((x2-x1))*1.0) / image.cols;
            float center_h = (abs((y2-y1))*1.0) / image.rows;
            std::string cls_label = class_names[obj.label];
            char tmp123[192];
            std::ofstream location_out;
//            sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.txt",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d_raw.txt", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
//            location_out.open(tmp123, std::ios::out | std::ios::app);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
            location_out.open(tmp123, std::ios::out);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
            location_out <<cls_label<<" "<<center_x<<" "<<center_y<<" "<<center_w<<" "<<center_h<<"\n";
            location_out.close();
        }
//        sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        cv::imwrite(tmp, image);
    }

    else if (strcmp(ifsave, "savetest") == 0){
        for (size_t i = 0; i < objects.size(); i++){
            const Object& obj = objects[i];
            float x1 = obj.rect.x;
            float y1 = obj.rect.y;
            float x2 = obj.rect.x + obj.rect.width;
            float y2 = obj.rect.y + obj.rect.height;
            float center_x = ((x1+x2)*0.5) / image.cols;
            float center_y = ((y1+y2)*0.5) / image.rows;
            float center_w = (abs((x2-x1))*1.0) / image.cols;
            float center_h = (abs((y2-y1))*1.0) / image.rows;
            std::string cls_label = class_names[obj.label];
            char tmp123[192];
            std::ofstream location_out;
//            sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d-%d_draw.txt",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d_draw.txt",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            location_out.open(tmp123, std::ios::out | std::ios::app);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
            location_out <<cls_label<<" "<<center_x<<" "<<center_y<<" "<<center_w<<" "<<center_h<<"\n";
            location_out.close();
        }
//        sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
//        cv::imwrite(tmp, image);
    }

    else if (strcmp(ifsave, "saveselect") == 0)
    {
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];
            float x1 = obj.rect.x;
            float y1 = obj.rect.y;
            float x2 = obj.rect.x + obj.rect.width;
            float y2 = obj.rect.y + obj.rect.height;
            float center_x = ((x1 + x2) * 0.5) / image.cols;
            float center_y = ((y1 + y2) * 0.5) / image.rows;
            float center_w = (abs((x2 - x1)) * 1.0) / image.cols;
            float center_h = (abs((y2 - y1)) * 1.0) / image.rows;
            std::string cls_label = class_names[obj.label];
            //            char tmp123[192];
            //            std::ofstream location_out;
            //            sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.txt",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            //            location_out.open(tmp123, std::ios::out | std::ios::app);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
            //            location_out <<cls_label<<" "<<center_x<<" "<<center_y<<" "<<center_w<<" "<<center_h<<"\n";
            //            location_out.close();

//            if (obj.prob < 0.62)
//            {
//                char tmp123[192];
//                std::ofstream location_out;
//                sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.txt", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
//                location_out.open(tmp123, std::ios::out | std::ios::app); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
//                location_out << cls_label << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
//                location_out.close();
//
//                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
//                cv::imwrite(tmp, image);
//            }



            if (obj.prob >= 0.62)
            {
                char tmp123[192];
                std::ofstream location_out;
//                sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.txt", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp123, "ai_result/%d-%d-%d-%d-%d-%d_raw.txt",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                location_out.open(tmp123, std::ios::out | std::ios::app); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                location_out << cls_label << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                location_out.close();

//                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }



        }
    }

    else if (strcmp(ifsave, "saveseg") == 0)
    {
//        if (NULL==objects.size()){
        if (objects.empty()){
//            sprintf(tmp, "ai_result_null/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            sprintf(tmp, "ai_result_null/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            cv::imwrite(tmp, image);
        }



        else{
            for (size_t i = 0; i < objects.size(); i++){
                const Object& obj = objects[i];
                float x1 = obj.rect.x;
                float y1 = obj.rect.y;
                float x2 = obj.rect.x + obj.rect.width;
                float y2 = obj.rect.y + obj.rect.height;
                float center_x = ((x1 + x2) * 0.5) / image.cols;
                float center_y = ((y1 + y2) * 0.5) / image.rows;
                float center_w = (abs((x2 - x1)) * 1.0) / image.cols;
                float center_h = (abs((y2 - y1)) * 1.0) / image.rows;
                std::string cls_label = class_names[obj.label];

                if (obj.prob >= 0.62)
                {
                    char tmp123[192];
                    std::ofstream location_out;
//                    sprintf(tmp123, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d-%d_raw.txt", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp123, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d_raw.txt",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    location_out.open(tmp123, std::ios::out | std::ios::app); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                    location_out << cls_label << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                    location_out.close();
//                    sprintf(tmp, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    cv::imwrite(tmp, image);
                }

                if (obj.prob < 0.62)
                {
                    char tmp123[192];
                    std::ofstream location_out;
//                    sprintf(tmp123, "ai_result_xy_062/%d-%d-%d-%d-%d-%d-%d_raw.txt", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp123, "ai_result_xy_062/%d-%d-%d-%d-%d-%d_raw.txt",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    location_out.open(tmp123, std::ios::out | std::ios::app); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                    location_out << cls_label << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                    location_out.close();
//                    sprintf(tmp, "ai_result_xy_062/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp, "ai_result_xy_062/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    cv::imwrite(tmp, image);
                }

            }

        }

    }

    
    else if (strcmp(ifsave, "savedyxy") == 0)
    {
            for (size_t i = 0; i < objects.size(); i++){
                const Object& obj = objects[i];
                float x1 = obj.rect.x;
                float y1 = obj.rect.y;
                float x2 = obj.rect.x + obj.rect.width;
                float y2 = obj.rect.y + obj.rect.height;
                float center_x = ((x1 + x2) * 0.5) / image.cols;
                float center_y = ((y1 + y2) * 0.5) / image.rows;
                float center_w = (abs((x2 - x1)) * 1.0) / image.cols;
                float center_h = (abs((y2 - y1)) * 1.0) / image.rows;
                std::string cls_label = class_names[obj.label];
                // std::cout<< "obj.prob:"<<obj.prob<<std::endl;
                if (obj.prob >= 0.62)
                {
                    char tmp123[192];
                    std::ofstream location_out;
//                    sprintf(tmp123, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d-%d_raw.txt", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp123, "ai_result_dydy_062/lable/%d-%d-%d-%d-%d-%d_raw.txt",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
//                    location_out.open(tmp123, std::ios::out | std::ios::app); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                    location_out.open(tmp123, std::ios::out); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                    // location_out << cls_label << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                     location_out << 0 << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                    location_out.close();
//                    sprintf(tmp, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp, "ai_result_dydy_062/image/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    cv::imwrite(tmp, image);
                }

                if (obj.prob < 0.62)
                {
                    char tmp123[192];
                    std::ofstream location_out;
//                    sprintf(tmp123, "ai_result_xy_062/%d-%d-%d-%d-%d-%d-%d_raw.txt", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp123, "ai_result_xy_062/lable/%d-%d-%d-%d-%d-%d_raw.txt",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
//                    location_out.open(tmp123, std::ios::out | std::ios::app); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                    location_out.open(tmp123, std::ios::out); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                    // location_out << cls_label << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                     location_out << 0 << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                    location_out.close();
//                    sprintf(tmp, "ai_result_xy_062/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    sprintf(tmp, "ai_result_xy_062/image/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    cv::imwrite(tmp, image);
                }
        }
        
        // std::cout<<"objects.size()"<<objects.size()<<std::endl;
        if(objects.size()>1){
            std::cout<<"objects.size()>1"<<std::endl;
            sprintf(tmp, "ai_result_xy_save/more/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            cv::imwrite(tmp, image);
        }
        //hchen space key save img to pathfile
        static int count_tmp = 0;
        char key = cv::waitKey(10);
        if (key == 's' || count_tmp)
        {
            if(count_tmp == 0)
            {
                count_tmp = 1;
                std::cout<<count_tmp<<std::endl;
            }else{
                 count_tmp++;   
                 if(count_tmp > 150)
                 {
                    std::cout<<"img save successful!!"<<std::endl;
                    sprintf(tmp, "ai_result_xy_save/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                    cv::imwrite(tmp, image);
                    count_tmp = 1;
                }
            }
        }
        static bool save_all =false;
        if(key == 'w'||save_all){
            save_all = true;
            if(objects.size()<1){
                std::cout<<"img saving!!"<<std::endl;
                sprintf(tmp, "ai_result_xy_save/all/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
        }
        if(key == ' ')
        {
                std::cout<<"img save successful!!"<<std::endl;
                sprintf(tmp, "ai_result_xy_save/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
        }
        if(key == 'q'){
            count_tmp = 0;
            save_all = false;
            std::cout<<"stop save-img!!!"<<std::endl;
        }


    }

    else if (strcmp(ifsave, "savenull") == 0)
        {
//            if (NULL==objects.size()){
            if (objects.empty()){
                std::cout<<"NULL"<<std::endl;
//                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
    }


    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        if(obj.label==0) cv::rectangle(image, obj.rect, cv::Scalar(0,255,255),2);
        if(obj.label==1) cv::rectangle(image, obj.rect, cv::Scalar(255,0,0),2);
        if(obj.label==2) cv::rectangle(image, obj.rect, cv::Scalar(255,255,0),2);
        if(obj.label==3) cv::rectangle(image, obj.rect, cv::Scalar(0,255,0),2);
        if(obj.label==4) cv::rectangle(image, obj.rect,cv::Scalar(0,0,255),2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
 
    if (strcmp(ifsave, "save")==0){
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];
            if(obj.label==0 || obj.label==1 || obj.label==2){
//                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_draw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_draw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);}
        }
    }

    if (strcmp(ifsave, "saveall")==0){
//        sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_draw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_draw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        cv::imwrite(tmp, image);
    }

    else if (strcmp(ifsave, "savetest")==0){
//        sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_draw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_draw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        cv::imwrite(tmp, image);
    }
    //测试mp4,并将其存储为mp4的格式,且原图保存至ai_mp4/save_img中
    //命名规则以mp4_num的形式命名
    else if(strcmp(ifsave,"save_mp4")==0){
        std::string filename(filestr);
        if(objects.size()<1){
            std::string path = "save_img/";
            path += filename;
            cv::imwrite(path, image);
        }else{
                std::string path = "ai_result_draw/";
                path += filename;
            for (size_t i = 0; i < objects.size(); i++){
                const Object& obj = objects[i];
                if(obj.prob >= 0.62){
                    cv::imwrite(path, image);
                }
                if(obj.prob < 0.62){
                    cv::imwrite(path, image);
                }
             }
        }
    }

    //测试img_dir,存储,其中filename的值会影响其结果
    else if (strcmp(ifsave, "savedydx")==0){
        
        std::string filename(filestr);
        if(objects.size()<1){
            std::string path = "ai_result_no/";
            path += filename;
            cv::imwrite(path, image);
        }else{
                std::string path = "ai_result_draw/";
                path += filename;
            for (size_t i = 0; i < objects.size(); i++){
                const Object& obj = objects[i];
                if(obj.prob >= 0.62){
                    cv::imwrite(path, image);
                }
                if(obj.prob < 0.62){
                    cv::imwrite(path, image);
                }
             }
        }
    }

    else if (strcmp(ifsave, "saveselect")==0)
    {
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];
            //            if(obj.prob < 0.62){
            //                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_draw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            //                cv::imwrite(tmp, image);
            //            }

            if (obj.prob >= 0.62)
            {
//                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_draw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_draw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
        }
    }

    else if (strcmp(ifsave, "saveseg")==0){
        for (size_t i = 0; i < objects.size(); i++){
            const Object& obj = objects[i];
            if(obj.prob >= 0.62){
//                sprintf(tmp, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d-%d_draw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result_dydy_062/%d-%d-%d-%d-%d-%d_draw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
            if(obj.prob < 0.62){
//                sprintf(tmp, "ai_result_xy_062/%d-%d-%d-%d-%d-%d-%d_draw.jpg",inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result_xy_062/%d-%d-%d-%d-%d-%d_draw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
        }
    }



    // cv::namedWindow("image",cv::WINDOW_NORMAL);
    // local-video/http-video/local-webcam/http-webcam show
    if (is_streaming==1 && strcmp(ifshow, "show")==0){
        char string[10];  // 用于存放帧率的字符串
        static double prevTime = 0.0;
        double currTime, elapsedTime;
        currTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency(); // 获取当前时间
        elapsedTime = currTime - prevTime;
        double fps = 1.0 / elapsedTime;
        prevTime = currTime;
        sprintf(string, "%.1f", fps);
        std::string fpsString("FPS:");
        fpsString += string;
        cv::putText(image, fpsString , cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2); // 修改字体颜色为红色
        cv::imshow("image", image);
        cv::waitKey(1);
        // std::cout<<"1"<<std::endl;
    }

    // local-video/http-video/local-webcam/http-webcam noshow
    if (is_streaming==1 && strcmp(ifshow, "show")!=0){
        cv::waitKey(1);
        // std::cout<<"11"<<std::endl;
    }

    // http img show
    if (is_streaming==2 && strcmp(ifshow, "show")==0){
        cv::resize(image,image, cv::Size(1280, 720));
        cv::imshow("image", image);
        cv::waitKey(0);
        // std::cout<<"111"<<std::endl;
    }

    // http img noshow
    if (is_streaming==2 && strcmp(ifshow, "show")!=0){
        cv::resize(image,image, cv::Size(1280, 720));
        cv::waitKey(0);
        // std::cout<<"1111"<<std::endl;
    }

    // local img show
    if (is_streaming==0 && strcmp(ifshow, "show")==0){
        cv::resize(image,image, cv::Size(1280, 720));
        cv::imshow("image", image);
        cv::waitKey(1);
        // std::cout<<"11111"<<std::endl;
    }

    // local img noshow
    if (is_streaming==0 && strcmp(ifshow, "show")!=0){
        cv::waitKey(1);
    }

    // std::cout<<"mp4 start!!!"<<std::endl;
    return 0;

}




static int init_yolov5(ncnn::Net* yolov5,
                       const char* model_param_path,
                       const char* model_bin_path)
{
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // windows无法使用绑定大小核
    //ncnn::set_cpu_powersave(1); // bind to little cores
    //ncnn::set_cpu_powersave(2); // bind to big cores
    //ncnn::set_cpu_powersave(0); // 0:全部 1:小核 2:大核

//    ncnn::set_cpu_thread_affinity();
//    ncnn::set_flush_denormals();
//    ncnn::set_kmp_blocktime();
//    ncnn::set_omp_dynamic();

    yolov5->opt.lightmode = true;  // 启用时，中间的blob将被回收。默认情况下启用
    yolov5->opt.num_threads = 1;  //  线程数  默认值是由get_cpu_count()返回的值。openmp关闭之后,无法法用
    yolov5->opt.use_packing_layout = true;  // improve all operator performance on all arm devices, will consume more memory
    yolov5->opt.openmp_blocktime = 20;      // openmp线程在进入睡眠状态前忙着等待更多工作的时间 默认值为20ms，以保持内核的正常工作。而不至于在之后有太多的额外功耗，openmp关闭之后,无法法用
    yolov5->opt.use_winograd_convolution = true;  // improve convolution 3x3 stride1 performance, may consume more memory
    yolov5->opt.use_sgemm_convolution = true;  // improve convolution 1x1 stride1 performance, may consume more memory
    yolov5->opt.use_int8_inference = true;  //启用量化的int8推理 对量化模型使用低精度的int8路径 默认情况下启用
    yolov5->opt.use_vulkan_compute = true;  // 使用gpu进行推理计算
    yolov5->opt.use_bf16_storage = true;  // improve most operator performance on all arm devices, may consume more memory
    yolov5->opt.use_packing_layout = true;  //  enable simd-friendly packed memory layout,  improve all operator performance on all arm devices, will consume more memory, enabled by default
    //----------------------------------------------------------------------------------------------------------------------------------------------------



    yolov5->load_param(model_param_path);
    yolov5->load_model(model_bin_path);


    return 0;
}



 int main(int argc, char** argv){

    cv::Mat frame;
    std::vector<Object> objects;
    cv::VideoCapture cap;

    //    n
    const char* devicepath = argv[1];// http://172.23.215.226:8080/?action=stream
    const char* ifsave = argv[2]; //savedyxy
    const char* ifshow = argv[3];  //show
    const char* threshold = argv[4]; //0.25
    const char* model_param_path = argv[5]; //model/yolov5n-xkjc.param model/yolov5n-xkjc.bin panorama
    const char* model_bin_path = argv[6]; //model/yolov5n-xkjc.param model/yolov5n-xkjc.bin panorama
    const char* mode = argv[7];  //panorama
    const char* input_size = argv[8]; //320 
    const char* set_rotation = argv[9];  //n

    // std::cout<<"mode:"<<mode<<std::endl;

//    if (strcmp(mode, "panorama") == 0){
//        static const char* class_names[] = {"nozzle-blob", "spaghetti", "stringing"};
//    }
//
//    if (strcmp(mode, "macro") == 0){
//        static const char* class_names[] = {"layer-matter", "nozzle-matter", "to-high", "to-low"};
//    }




    int is_streaming = 0;
    const char* filename;

    ncnn::Net yolov5;

    init_yolov5(&yolov5, model_param_path, model_bin_path); //We load model and param first!

    // local img ----------------------------------------------------------------------------------------
    if (strstr(devicepath, ".jpg")  && !strstr(devicepath, "http")){
        frame = cv::imread(devicepath, 1);
    }
    // local mp4 ----------------------------------------------------------------------------------------
    if (strstr(devicepath, ".mp4")){
        // size_t pos = devicepath.find_last_of("/\\");
        // std::string temp = devicepath.substr(pos+1);
        // const char* filename = temp.c_str();
        // std::cout << filename << std::endl;

        cap.open(devicepath);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        is_streaming = 1;
        
    }
    // local webcam ----------------------------------------------------------------------------------------
    if (strstr(devicepath, "/dev/video")){
        cap.open(devicepath);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        is_streaming = 1;
    }
    // http webcam ----------------------------------------------------------------------------------------
//    if (strstr(devicepath, "http:")  && strstr(devicepath, "=stream")){
//        cap.open(devicepath);
//        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
//        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
//        is_streaming = 1;
//    }
    if (strstr(devicepath, "http:")  && strstr(devicepath, "=stream")){
        cap.open(devicepath);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        is_streaming = 1;
    }



    // http jpg ----------------------------------------------------------------------------------------
    if (strstr(devicepath, "http:")  && strstr(devicepath, ".jpg")){
        cap.open(devicepath);
        is_streaming = 2;
    }
    // http mp4 ----------------------------------------------------------------------------------------
    if (strstr(devicepath, "http:")  && strstr(devicepath, "mp4")){
        cap.open(devicepath);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        is_streaming = 1;
    }
    // local img-dir ----------------------------------------------------------------------------------------
    if (strstr(devicepath, "img_dir")){
        std::vector<cv::String> fn;
        cv::glob(devicepath, fn, false);
        int count = fn.size(); //number of png files in images folder
        std::cout<<"img_num:"<<count<<std::endl;
        for (int i = 0; i < count; i++)
        {
            frame = cv::imread(fn[i], 1);
            // std::cout<<"fn:"<<fn[i]<<std::endl;
            size_t pos = fn[i].find_last_of("/\\");
            // const char* filename = fn[i].substr(pos+1).c_str();
            // std::cout << filename << std::endl;
            std::string temp = fn[i].substr(pos+1);
            filename = temp.c_str();
            // std::cout << filename << std::endl;
            detect_yolov5(frame, objects, &yolov5,threshold,input_size);
            draw_objects(frame, objects, is_streaming, ifsave, ifshow, mode, i, model_param_path,filename);
            fprintf(stderr, "%d / %d \n", i, count);
        }
        return 0;
    }
    // local mp4-dir ----------------------------------------------------------------------------------------
    if (strstr(devicepath, "video_dir")){
        std::vector<cv::String> fn;
        cv::glob(devicepath, fn, false);
        int count = fn.size(); //number of png files in images folder
        int inum = 0;
        for (int i = 0; i < count; i++)
        {
            frame = cv::imread(fn[i], 1);
            cap.open(fn[i]);
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            is_streaming = 1;
            while (1){
                if (is_streaming){
                    cap >> frame;
                }
                if (frame.empty()){
                    break;
                }
                detect_yolov5(frame, objects, &yolov5,threshold,input_size);
                draw_objects(frame, objects, is_streaming, ifsave, ifshow, mode, inum, model_param_path);
                inum = inum + 1;
            }
        }
    }


    int i = 0;
    while (1){


        if (is_streaming){
            cap >> frame;
        }

//        fps = cap.get(cv::CAP_PROP_FPS);

        //----------------------------------------------------
//        cv::Mat outImg_180;
//        cv::rotate(frame, outImg_180, cv::ROTATE_180);
//        detect_yolov5(outImg_180, objects, &yolov5, threshold, input_size);
//        draw_objects(outImg_180, objects, is_streaming, ifsave, ifshow, mode, i);

            //  img rotate 180 
         if (strstr(set_rotation, "y")){
            cv::Mat outImg_180;
            cv::rotate(frame, outImg_180, cv::ROTATE_180);
            detect_yolov5(outImg_180, objects, &yolov5, threshold, input_size);
            draw_objects(outImg_180, objects, is_streaming, ifsave, ifshow, mode, i, model_param_path);
        }

        if (strstr(set_rotation, "n")){
            
            detect_yolov5(frame, objects, &yolov5,threshold ,input_size);
            draw_objects(frame, objects, is_streaming, ifsave, ifshow, mode, i, model_param_path,filename);
        }



//        detect_yolov5(frame, objects, &yolov5,threshold ,input_size);
//        draw_objects(frame, objects, is_streaming, ifsave, ifshow, mode, i);
        i = i + 1;
        if (!is_streaming){
            return 0;
        }
    }
 }


