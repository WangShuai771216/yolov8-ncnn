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
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

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

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

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

/*** 
 * @description: CLAHE处理
 * @param {Mat&} image
 * @return {*}
 */
static cv::Mat& equalize_CLAHE(const cv::Mat& image) {
    static cv::Mat new_image;

    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return new_image;
    }

    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Ptr<cv::CLAHE> clahe_trans = cv::createCLAHE();
    clahe_trans->setClipLimit(3.0);  
    clahe_trans->setTilesGridSize(cv::Size(8, 8)); 

    for (int i = 0; i < channels.size(); ++i) {
        clahe_trans->apply(channels[i], channels[i]);
    }

    cv::merge(channels, new_image);
    return new_image;
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



    ex.input("in0", in_pad);


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
    // std::cout<<"tmp tmp 1"<<std::endl;
    // stride 8
    {
        ncnn::Mat out;
        // ex.extract("output", out);
        ex.extract("out0", out);

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

        // ex.extract("353", out);
         ex.extract("out1", out);

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

        // ex.extract("367", out);
         ex.extract("out2", out);

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


/*** 
 * @description: 路径创建
 * @param {string} path
 * @return {*}
 */
static bool mkdirPath(std::string path){
    if(std::system(("test -d " + path).c_str()) != 0){//判断路径是否存在
            // 使用系统调用创建目录
        int result = std::system(("mkdir -p " + path).c_str());
        if (result != 0) {
            std::cout << "无法创建目录：" << path << std::endl;
            return false;
        }
    }
    return true;
}
/*** 
 * @description: 创建文件夹list
 * @param {string} path_parant
 * @return {*}
 */
static void mkdirPathList(std::string path_parant){
    const std::string class_path[9] = {"/ai_result_no/","/ai_result_draw/","/ai_l_062/","/ai_h_062/",
    "/ai_save/","/ai_more/","/ai_save_all/","/ai_save_ok/","/ai_save_no/"};
    // std::cout<<class_path->size()<<std::endl;
    for (size_t i = 0; i < 9; i++)
    {
        std::string path = path_parant + class_path[i];
        mkdirPath(path);
        path = path_parant + class_path[i]+"images";
        mkdirPath(path);
        path = path_parant + class_path[i]+"labels";
        mkdirPath(path);
    }
}

static bool objects_position_detect(const std::vector<Object>& objects,std::vector<size_t>& indices){
    //1.读取每个索引的x1,y1,w,h,并计算x2,y2
    std::vector<float> x_s,y_s,x_e,y_e,w,h;
    for( size_t i = 0;i < indices.size();i++){
        const Object& obj = objects[indices[i]];
        x_s.push_back(obj.rect.x);
        y_s.push_back(obj.rect.y);
         w.push_back(obj.rect.width);
         h.push_back(obj.rect.height);
        //  std::cout<<"obj:"<<objects[indices[i]].label<<obj.rect<<std::endl;
    }
    
    if(x_s[1] > (x_s[0] + w[0]/2) && (x_s[1] + w[1]) <( x_s[0] + w[0])){
        if(x_s[2] > (x_s[0] + w[0]/2) && (x_s[2] + w[2]) <( x_s[0] + w[0])){
            if((y_s[1] + h[1]) > (y_s[0] + h[0]) && y_s[1] < ( y_s[0] + h[0] + h[1])){
                if((y_s[2] + h[1]) > (y_s[0] + h[0]) && y_s[2] < ( y_s[01] + h[1])){
                    if(h[2] > h[1] && abs(w[2] - w[1]) < h[1]){
                        std::cout<<"nozzle no_extrusion!!!\n";
                        return true;
                    }
                }

            }
        }
    }
    return false;
}

/*** 
 * @description: 
 * @param  std::vector<Object>& objects:检测到的目标
 * @param  std::vector<char*> class_names:所需要检测的目标
 * @return {*}
 */
static int objects_detect(const std::vector<Object>& objects,const std::vector<const char*> class_names){
    const std::vector<std::string> no_extrusion_names = {"nozzle_all","nozzle_part","no-extrusion"};
    // for(size_t i = 0;i<objects.size();i++){
    //     std::cout<<"obj:"<<objects[i].label<<" "<<objects[i].prob;
    // }
    // std::cout<<std::endl;
    // 1.筛选出以检测到目标的names:
    std::vector<std::string> cls_names;
    for(size_t i = 0;i<objects.size();i++){
        const Object& obj = objects[i];
        cls_names.push_back(class_names[obj.label]);
        // std::cout<<"cls_names:"<<cls_names[i]<<std::endl;
    }
    // 2.对比cls_names与no_extrusion_names的包含关系,判断是否满足悬空条件.
    bool all_present = std::all_of(no_extrusion_names.begin(), no_extrusion_names.end(),
                                   [&cls_names](const std::string& name) {
                                       return std::find(cls_names.begin(), cls_names.end(), name) != cls_names.end();
                                   });
    // 3.检测到all_present=true,判断该三个labels之间的位置关系.
    if (all_present) {

        std::cout << "no_extrusion_names E cls_names" << std::endl;
        // 找到no_extrusion_names中每个元素在cls_names中的索引
        std::vector<size_t> indices;
        std::transform(no_extrusion_names.begin(), no_extrusion_names.end(), std::back_inserter(indices),
                       [&cls_names](const std::string& name) {
                           return std::distance(cls_names.begin(), std::find(cls_names.begin(), cls_names.end(), name));
                       });
        // // 输出索引
        // std::cout << "索引位置为: ";
        // for (size_t index : indices) {
        //     std::cout << index << " ";
        // }
        // std::cout << std::endl;
        bool no_extrusion_flag = objects_position_detect(objects,indices);

    } else {
        // std::cout << "no_extrusion_names 不完全包含于 cls_labels" << std::endl;
    }
    
    return 0;
}


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
        class_names = {"nozzle_all","nozzle_part","no-extrusion"}; // 初始化数组 
    }
    else{
        class_names = {"nozzle_all","nozzle_part","no-extrusion"}; // 初始化数组
    }
    

    objects_detect(objects,class_names);

    

//    time_t nowtime = time(NULL);
//    struct tm *p;
//    p = gmtime(&nowtime);
//    char tmp[192];

    time_t nowtime = time(NULL);
    struct tm *p;
    p = localtime(&nowtime);
    char tmp[192];



    cv::Mat image = bgr.clone();

    
    
    std::string file_save(ifsave);
    static bool MKDIR_FLAG = false;
    if (!MKDIR_FLAG)
    {
       mkdirPathList(file_save);
       MKDIR_FLAG = true;
    }
    
        
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
                cv::imwrite(tmp, image);
            }
        }
    }

    else if (strcmp(ifsave, "saveall") == 0){
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
//          location_out.open(tmp123, std::ios::out | std::ios::app);  //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
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
        }else{
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

    else if (strcmp(ifsave, "savenull") == 0)
        {
//            if (NULL==objects.size()){
            if (objects.empty()){
                // std::cout<<"NULL"<<std::endl;
//                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d-%d_raw.jpg", inum, 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                sprintf(tmp, "ai_result/%d-%d-%d-%d-%d-%d_raw.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
    }

    else if(strcmp(ifsave,"save_mp4") == 0){
        std::string filename;
        cv::String file_mp4(filestr);
        size_t pos = file_mp4.find_last_of(".");
        if (pos != std::string::npos) {
            sprintf(tmp, (file_mp4.substr(0, pos) +"-%d.jpg").c_str(), inum);
        } else {
            sprintf(tmp, "%d-%d-%d-%d-%d-%d.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        }
        
        std::string path_all = file_save +"/ai_result_all/"+ file_mp4.substr(0, pos)+"/";
        mkdirPath(path_all);
        path_all += tmp;
        cv::imwrite(path_all, image);
    }

    else if(strcmp(ifsave,"save_xkjc") == 0){
        if(objects.size()==3){
            if(objects[0].label != objects[1].label != objects[2].label){
                std::cout<<"no-extrusion!!!"<<std::endl;
                sprintf(tmp, "%s/ai_save_ok/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }else{
                sprintf(tmp, "%s/ai_save_all/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
        }else if(objects.size()==2){
             if(objects[0].label != objects[1].label && objects[0].label!= 2 && objects[1].label != 2){
                // std::cout<<"extrusion!!!"<<std::endl;
                sprintf(tmp, "%s/ai_save_no/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            } else{
                sprintf(tmp, "%s/ai_save_all/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }  
        }else{
            sprintf(tmp, "%s/ai_save_all/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            cv::imwrite(tmp, image);
        }
    }

    else if (strcmp(ifsave, "savedyxy") == 0)
    {
        // std::cout<<"savedyxy"<<std::endl;
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
                sprintf(tmp123, "%s/ai_h_062/labels/%d-%d-%d-%d-%d-%d_raw.txt",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                location_out.open(tmp123, std::ios::out); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                location_out << 0 << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                location_out.close();
                sprintf(tmp, "%s/ai_h_062/images/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }else{
                char tmp123[192];
                std::ofstream location_out;
                sprintf(tmp123, "%s/ai_l_062/labels/%d-%d-%d-%d-%d-%d_raw.txt",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                location_out.open(tmp123, std::ios::out); //以写入和在文件末尾添加的方式打开.txt文件，没有的话就创建该文件。
                location_out << 0 << " " << center_x << " " << center_y << " " << center_w << " " << center_h << "\n";
                location_out.close();
                sprintf(tmp, "%s/ai_l_062/images/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
                }
        }
        

         if (std::string(model_param_path_str).find("-xkjc") != std::string::npos){
            if(objects.size()>3){
                std::cout<<"objects.size()>3"<<std::endl;
                sprintf(tmp, "%s/ai_more/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }   
         }else{
            if(objects.size()>1){
                std::cout<<"objects.size()>1"<<std::endl;
                sprintf(tmp, "%s/ai_more/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
         }
        
        //hchen space key save img to pathfile
        static int count_tmp = 0;
        char key = cv::waitKey(10);
        if (key == 's' || count_tmp)
        {
            if(count_tmp == 0)
            {
                count_tmp = 1;
                std::cout<<"start save img!!!"<<std::endl;
            }else{
                 count_tmp++;   
                 if(count_tmp > 150)
                 {
                    std::cout<<"img save successful!!"<<std::endl;
                    sprintf(tmp, "%s/ai_save/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
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
                sprintf(tmp, "%s/ai_save_all/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
                cv::imwrite(tmp, image);
            }
        }
        if(key == ' ')
        {
            std::cout<<"img save successful!!"<<std::endl;
            sprintf(tmp, "%s/ai_save/%d-%d-%d-%d-%d-%d_raw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
            cv::imwrite(tmp, image);
        }
        if(key == 'q'){
            count_tmp = 0;
            save_all = false;
            std::cout<<"stop save-img!!!"<<std::endl;
        }

    }

    // else{

    // }


    //根据objects检测结果在图上标出label
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


    std::string path;
    
    if(objects.size()<1){
        path = file_save+"/ai_result_no/";
    }else{
        path = file_save+"/ai_result_draw/";
    }

  
    //测试mp4,并将其存储为mp4的格式,且原图保存至ai_mp4/save_img中
    //命名规则以mp4_num的形式命名
     if(strcmp(ifsave,"save_mp4")==0){
        std::string filename;
        cv::String file_mp4(filestr);
        size_t pos = file_mp4.find_last_of(".");
        if (pos != std::string::npos) {
            sprintf(tmp, (file_mp4.substr(0, pos) +"-%d.jpg").c_str(), inum);
        } else {
            sprintf(tmp, "%d-%d-%d-%d-%d-%d.jpg",1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        }
        path += tmp;
        cv::imwrite(path, image);
    }

    //测试img_dir,存储,其中filename的值会影响其结果
    else if (strcmp(ifsave, "save_imgdir")==0){
        std::string filename(filestr);
        path += filename;
        cv::imwrite(path, image);
    }
    else {

        // sprintf(tmp, "%s/ai_result_draw/%d-%d-%d-%d-%d-%d_draw.jpg",ifsave,1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
        // cv::imwrite(tmp, image);
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
    const char* model_param_path = argv[5]; //model/yolov5n-xkjc.param model/yolov5n-xkjc.param panorama
    const char* model_bin_path = argv[6]; //model/yolov5n-xkjc.param model/yolov5n-xkjc.bin panorama
    const char* mode = argv[7];  //panorama
    const char* input_size = argv[8]; //320 
    const char* set_rotation = argv[9];  //n


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

    // local img  and http img ---------------------------------------------------------------------------
    if (strstr(devicepath, ".jpg")){
        if(!strstr(devicepath, "http")){ //local img 
            frame = cv::imread(devicepath, 1);
            // frame = equalize_CLAHE(frame);
        }else{ //http img
            cap.open(devicepath);
            is_streaming = 2;
        }
    }
    // local mp4 and http mp4 ------------------------------------------------------------------------------------
    else if (strstr(devicepath, ".mp4")){
        // if(!strstr(devicepath, "http")){  // local mp4 
        //     cv::String file_path(devicepath);
        //     size_t pos = file_path.find_last_of("/\\");
        //     std::string temp = file_path.substr(pos+1);
        //     filename = temp.c_str();//视频名字
        //     cap.open(devicepath);
        //     cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        //     cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        //     is_streaming = 1;
        // }else{//http mp4
        //     cap.open(devicepath);
        //     cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        //     cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        //     is_streaming = 1;
        // }
        cv::String file_path(devicepath);
        size_t pos = file_path.find_last_of("/\\");
        std::string temp = file_path.substr(pos+1);
        filename = temp.c_str();//视频名字
        cap.open(devicepath);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        is_streaming = 1;
        
        
    }
    // local webcam ----------------------------------------------------------------------------------------
    else if (strstr(devicepath, "/dev/video")){
        cap.open(devicepath);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        is_streaming = 1;
    }
    // http webcam ----------------------------------------------------------------------------------------
    else if (strstr(devicepath, "http:")  && strstr(devicepath, "=stream")){
        
        cap.open(devicepath);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        is_streaming = 1;
    }

    // local img-dir ----------------------------------------------------------------------------------------
    else if (strstr(devicepath, "img_dir")){
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
            // frame = equalize_CLAHE(frame);
            detect_yolov5(frame, objects, &yolov5,threshold,input_size);
            draw_objects(frame, objects, is_streaming, ifsave, ifshow, mode, i, model_param_path,filename);
            fprintf(stderr, "%d / %d \n", i, count);
        }
        return 0;
    }
    // local mp4-dir ----------------------------------------------------------------------------------------
    else if (strstr(devicepath, "mp4_dir")){
        std::vector<cv::String> fn;
        cv::glob(devicepath, fn, false);
        int count = fn.size(); //number of png files in images folder
       
        for (int i = 0; i < count; i++)
        {
            int inum = 0;
            // frame = cv::imread(fn[i], 1);
            size_t pos = fn[i].find_last_of("/\\");
            std::string temp = fn[i].substr(pos+1);
            filename = temp.c_str();
            std::cout << filename << std::endl;

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
                // frame = equalize_CLAHE(frame);
                detect_yolov5(frame, objects, &yolov5,threshold,input_size);
                draw_objects(frame, objects, is_streaming, ifsave, ifshow, mode, inum, model_param_path,filename);
                inum = inum + 1;
            }
        }
        return 0;
    }


    int i = 0;
    while (1){

        if (is_streaming){
            // std::cout<<"61"<<std::endl;
            cap >> frame;
            // std::cout<<"51"<<std::endl;
        }

        // frame = equalize_CLAHE(frame); 
   
         

//        fps = cap.get(cv::CAP_PROP_FPS);

            //  img rotate 180 
         if (strstr(set_rotation, "y")){
            cv::Mat outImg_180;
            cv::rotate(frame, outImg_180, cv::ROTATE_180);
            detect_yolov5(outImg_180, objects, &yolov5, threshold, input_size);
            draw_objects(outImg_180, objects, is_streaming, ifsave, ifshow, mode, i, model_param_path,filename);
        }

        if (strstr(set_rotation, "n")){
            // std::cout<<"51"<<std::endl;

            detect_yolov5(frame, objects, &yolov5,threshold ,input_size);
            // auto start = std::chrono::high_resolution_clock::now();
            draw_objects(frame, objects, is_streaming, ifsave, ifshow, mode, i, model_param_path,filename);
            // 获取当前时间点（结束时间）
            // auto end = std::chrono::high_resolution_clock::now();
            // // 计算时间差异
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            // // 输出执行时间
            // std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

        }
        i = i + 1;
        if (!is_streaming){
            return 0;
        }
    }
 }


