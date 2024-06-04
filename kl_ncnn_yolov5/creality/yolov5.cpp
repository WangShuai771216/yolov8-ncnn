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
//#include "cpu.h"  //绑定大小核
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
#include <iostream>
#include "yolov5.h"
using namespace std;
using namespace cv;
#define MAX_STRIDE 64
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

/*** 
 * @description: 用于判断包裹检测两类别的位置关系是否满足条件
 * @return true:包裹
 */
static bool blob_objects_position_detect(const std::vector<Object>& objects,std::vector<size_t>& indices){

    //1.读取每个索引的x,y,w,h
    std::vector<float> x_s,y_s,x_e,y_e,w,h;
    for( size_t i = 0;i < indices.size();i++){
        const Object& obj = objects[indices[i]];
        x_s.push_back(obj.rect.x);
        y_s.push_back(obj.rect.y);
         w.push_back(obj.rect.width);
         h.push_back(obj.rect.height);
    }

    // 如果包裹坐标和喷头坐标没有交集,则false
    if ((x_s[0] + w[0]) < x_s[1] || (x_s[1] + w[1]) < x_s[0] || y_s[0] > y_s[1]) {
        // std::cout<<"blob detect position flase!!!\n";
        return false;
    }
    return true;
}

/*** 
 * @description: 包裹检测
 * @param  std::vector<Object>& objects:当前检测到的目标
 * @param  std::vector<char*> class_names:模型的所有类别class_names
 * @return {*} true:包裹
 */
bool blob_obj_detect(std::vector<Object>& objects,const std::vector<const char*> class_names){
    const std::vector<std::string> blob_names = {"nozzle_all","nozzle_blob"};
    std::vector<size_t> indices;
    // 1.筛选出以检测到目标的names:
    std::vector<std::string> cls_names;
    for(size_t i = 0;i<objects.size();i++){
        const Object& obj = objects[i];
        cls_names.push_back(class_names[obj.label]);
    }

    // 2.对比cls_names与blob_names的包含关系,判断是否满足悬空条件.
    bool all_present = std::all_of(blob_names.begin(), blob_names.end(),
                                   [&cls_names](const std::string& name) {
                                       return std::find(cls_names.begin(), cls_names.end(), name) != cls_names.end();
                                   });
    // 3.检测到all_present=true,判断该两个labels之间的位置关系.
    if (all_present) {
        // 找到blob_names中每个元素在cls_names中的索引
        std::transform(blob_names.begin(), blob_names.end(), std::back_inserter(indices),
                       [&cls_names](const std::string& name) {
                           return std::distance(cls_names.begin(), std::find(cls_names.begin(), cls_names.end(), name));
                       });
        //4.判读位置是否满足
        if(blob_objects_position_detect(objects,indices)){
            return true;
        }             
    } 
    return false;
}
/*** 
 * @description: CLAHE处理
 * @param {Mat&} image
 * @return {*}
 */
static cv::Mat& equalize_CLAHE(const cv::Mat& image) {//数据增强
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
int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects, const char* mode)
{
    ncnn::Net yolov5;
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // windows无法使用绑定大小核
    //ncnn::set_cpu_powersave(1); // bind to little cores
    //ncnn::set_cpu_powersave(2); // bind to big cores
    //ncnn::set_cpu_powersave(0); // 0:全部 1:小核 2:大核
    yolov5.opt.lightmode = true;  // 启用时，中间的blob将被回收。默认情况下启用
    yolov5.opt.num_threads = 1;  //  线程数  默认值是由get_cpu_count()返回的值。openmp关闭之后,无法法用
    yolov5.opt.use_packing_layout = false;  // improve all operator performance on all arm devices, will consume more memory
    yolov5.opt.openmp_blocktime = 0;      // openmp线程在进入睡眠状态前忙着等待更多工作的时间 默认值为20ms，以保持内核的正常工作。而不至于在之后有太多的额外功耗，openmp关闭之后,无法法用
    yolov5.opt.use_winograd_convolution = false;  // improve convolution 3x3 stride1 performance, may consume more memory
    yolov5.opt.use_sgemm_convolution = false;  // improve convolution 1x1 stride1 performance, may consume more memory
    yolov5.opt.use_int8_inference = true;  //启用量化的int8推理 对量化模型使用低精度的int8路径 默认情况下启用
    yolov5.opt.use_vulkan_compute = false;  // 使用gpu进行推理计算
    yolov5.opt.use_bf16_storage = false;  // improve most operator performance on all arm devices, may consume more memory
    yolov5.opt.use_packing_layout = false;  //  enable simd-friendly packed memory layout,  improve all operator performance on all arm devices, will consume more memory, enabled by default
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    cv::Mat new_bgr;
   int  is_blob = 0;//当前是否是为blob检测=1 \喷头包裹 = 2
   if (std::string(mode).find("ywjc") != std::string::npos){//异物、平台板检测 
#ifndef CROSS_COMPILE
        yolov5.load_param("yolov5n-ywjc.param");
        yolov5.load_model("yolov5n-ywjc.bin");
#else
        yolov5.load_param("/usr/lib/yolov5/yolov5n-ywjc.param");
        yolov5.load_model("/usr/lib/yolov5/yolov5n-ywjc.bin");
#endif
   }
    else if(std::string(mode).find("blob") != std::string::npos){//喷头包裹检测 使用目标检测方案
        is_blob = 1;

#ifndef CROSS_COMPILE
        yolov5.load_param("yolov5n-blob.param");
        yolov5.load_model("yolov5n-blob.bin");
#else
        yolov5.load_param("/usr/lib/yolov5/yolov5n-blob.param");
        yolov5.load_model("/usr/lib/yolov5/yolov5n-blob.bin");
#endif
    }
    else if(std::string(mode).find("warp") != std::string::npos){//warp检测 该用目标检测方案
        
        // is_no_extrusion = 2;
#ifndef CROSS_COMPILE
        yolov5.load_param("yolov5n-warp.param"); //plate端
        yolov5.load_model("yolov5n-warp.bin");
#else
        yolov5.load_param("/usr/lib/yolov5/yolov5n-warp.param"); //pc端
        yolov5.load_model("/usr/lib/yolov5/yolov5n-warp.bin");
#endif
    }
    else if(std::string(mode).find("zwjc") != std::string::npos){//脏污检测 该用目标检测方案
#ifndef CROSS_COMPILE
        yolov5.load_param("yolov5n-zwjc.param"); //pc
        yolov5.load_model("yolov5n-zwjc.bin");
#else
        yolov5.load_param("/usr/lib/yolov5/yolov5n-zwjc.param"); //plate
        yolov5.load_model("/usr/lib/yolov5/yolov5n-zwjc.bin");
#endif
    }

   else {
#ifndef CROSS_COMPILE
        yolov5.load_param("yolov5n.param");//意面检测
        yolov5.load_model("yolov5n.bin");
#else
        yolov5.load_param("/usr/lib/yolov5/yolov5n.param");
        yolov5.load_model("/usr/lib/yolov5/yolov5n.bin");
#endif
   }

    const int target_size = 320;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;
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
    }else{
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat in;
    if (std::string(mode).find("warp") != std::string::npos){//翘曲检测
        cv::Mat src = bgr.clone();
        //new_bgr = equalize_CLAHE(src);//数据增强
        new_bgr=src;
        in = ncnn::Mat::from_pixels_resize(new_bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    }else{
        in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    }
    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yolov5.create_extractor();
    //-----------------------------------------------------------------------------------------------
    ex.set_light_mode(true);
    ex.set_num_threads(1);  //openmp关闭之后就是否使用线程控制
    //-----------------------------------------------------------------------------------------------
    ex.input("images", in_pad);
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
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
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
        // std::cout<<"label:"<<objects[i].label<<std::endl;
    }

    //当前包裹检测,需要判断检测的结果是否正确
    if(is_blob==1){
        std::vector<const char*> class_names = {"nozzle_all","nozzle_blob"};//为所有检测的label的集合,需要与检测的label的顺序对应
        bool blob_flag =  blob_obj_detect(objects, class_names);
        if(!blob_flag){//不满足包裹条件,清除本次关于包裹label信息.
            objects.clear();//由于包裹未和其他模型融合,这里直接清除objects结果,后期融合,需要指定清除
            // for(size_t i =0;i < objects.size();i++){
            // }
            std::cout<<"blob detect false!!!\n";
        }else{
            //不做任何处理,保留本次结果.
            std::cout<<"blob detect successful!!!\n";
        }
    }

    return 0;
}

void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, const char* savepath, const char* mode)
{
    
    std::vector<const char*> class_names; // 使用 std::vector 替代 C 数组
    if (std::string(mode).find("ywjc") != std::string::npos) {
        class_names = {"foreign_object"}; // 初始化数组 异物、平台板检测
    }
    // else if (std::string(mode).find("xkjc") != std::string::npos) {
    //     class_names = {"err_gap"}; // 初始化数组
    // }
    else if (std::string(mode).find("zwjc") != std::string::npos) {
        class_names = {"dirty"}; // 初始化数组 脏污检测

    }else if (std::string(mode).find("blob") != std::string::npos) {
         class_names = {"nozzle-all","nozzle-blob"}; // 初始化数组
    }
    else if (std::string(mode).find("warp") != std::string::npos) {
         class_names = {"warping"}; // 初始化数组 
    }else{
        class_names = {"nozzle-blob", "spaghetti", "stringing"}; // 初始化数组 意面检测
    }
    // std::cout<<objects.size()<<std::endl;
    //cv::Mat image = bgr.clone();
    cv::Mat image = bgr;
    
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        if(obj.label==0)
            cv::rectangle(image, obj.rect, cv::Scalar(0,255,255),2);
        if(obj.label==1)
            cv::rectangle(image, obj.rect, cv::Scalar(255,0,0),2);
        if(obj.label==2)
            cv::rectangle(image, obj.rect, cv::Scalar(255,255,0),2);
        if(obj.label==3)
            cv::rectangle(image, obj.rect, cv::Scalar(0,255,0),2);
        if(obj.label==4)
            cv::rectangle(image, obj.rect, cv::Scalar(0,0,255),2);
        if(obj.label==5)
            cv::rectangle(image, obj.rect, cv::Scalar(255,0,255),2);
        if(obj.label==6)
            cv::rectangle(image, obj.rect, cv::Scalar(128,42,42),2);
        if(obj.label==8)
            cv::rectangle(image, obj.rect, cv::Scalar(255,20,147),2);
        if(obj.label==9)
            cv::rectangle(image, obj.rect, cv::Scalar(255,130,71),2);
        if(obj.label==10)
            cv::rectangle(image, obj.rect, cv::Scalar(255,185,15),2);
        if(obj.label==11)
            cv::rectangle(image, obj.rect, cv::Scalar(255,20,147),2);
        if(obj.label==12)
            cv::rectangle(image, obj.rect, cv::Scalar(186,85,211),2);
        if(obj.label==13)
            cv::rectangle(image, obj.rect, cv::Scalar(255,165,0),2);
        if(obj.label==14)
            cv::rectangle(image, obj.rect, cv::Scalar(72,118,255),2);
        if(obj.label==15)
            cv::rectangle(image, obj.rect, cv::Scalar(178,58,238),2);
        if(obj.label==16)
            cv::rectangle(image, obj.rect, cv::Scalar(0,0,139),2);
        if(obj.label==17)
            cv::rectangle(image, obj.rect, cv::Scalar(0,139,139),2);
        if(obj.label==18)
            cv::rectangle(image, obj.rect, cv::Scalar(139,0,139),2);
        if(obj.label==19)
            cv::rectangle(image, obj.rect, cv::Scalar(144,238,144),2);
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
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite(savepath, image);
}
void get_results(const cv::Mat & bgr,const std::vector<Object>& objects, struct Results res[], int* cnt, const char* mode)
{
    std::vector<const char*> class_names; // 使用 std::vector 替代 C 数组
    if (std::string(mode).find("ywjc") != std::string::npos) {
        class_names = {"foreign_object"}; // 初始化数组
    }
    // else if (std::string(mode).find("xkjc") != std::string::npos) {
    //     class_names = {"no-extrusion"}; // 初始化数组
    // }
    else if (std::string(mode).find("zwjc") != std::string::npos) {
        class_names = {"dirty"}; // 初始化数组

    }else if (std::string(mode).find("blob") != std::string::npos) {
         class_names = {"nozzle-all","nozzle-blob"}; // 初始化数组 
    }
    else if (std::string(mode).find("warp") != std::string::npos) {
         class_names = {"warping"}; // 初始化数组 
    }else {
        class_names = {"nozzle-blob", "spaghetti", "stringing"}; // 初始化数组
    }
    
    cv::Mat image = bgr.clone();
    *cnt = objects.size();
    if (*cnt > 20)
    {
        cout << "picture count is too many: " << *cnt << endl;
        *cnt = 20;
    }
    for (size_t i = 0; i < *cnt; i++)
    {
        const Object& obj = objects[i];
        res[i].re_label = obj.label;
        res[i].re_prob = obj.prob;
        res[i].re_obj_rect_x = obj.rect.x;
        res[i].re_obj_rect_y = obj.rect.y;
        res[i].re_obj_rect_width = obj.rect.width;
        res[i].re_obj_rect_height = obj.rect.height;
    }

}

