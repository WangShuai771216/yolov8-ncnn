#ifndef __YOLO_H__
#define __YOLO_H__

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "layer.h"
#include "net.h"

namespace cvx
{
    struct KeyPoint
    {
        int x = 0;
        int y = 0;
        float score = 0.f;
        bool visible = false;
        KeyPoint(int x, int y, float score, bool visible) : \
            x(x), y(y), score(score), visible(visible) {}
    };

    struct Instance
    {
        cv::Mat mask{}; //
        std::vector<KeyPoint> keypoints{};
        cv::Rect box{0, 0, 0, 0};
        int label{-1};
        float prob{0.f};
        Instance(cv::Rect box, int label, float prob,
                 cv::Mat mask = cv::Mat(), std::vector<KeyPoint> keypoints = {}) : \
            box(box), label(label), prob(prob), mask(mask), keypoints(keypoints) {}
    };

    class Yolo8
    {
        public:
            Yolo8() = default;
            Yolo8(const char* param_file, const char* bin_file);
            ~Yolo8();
            static float clamp(float val, float min = 0.f, float max = 1280.f)
            {
                return val > min ? (val < max ? val : max) : min;
            }

            int inference(const cv::Mat &image, std::vector<Instance> &instances) const;
            static void visualize(const cv::Mat &image, 
                                  const std::vector<Instance> &instances, 
                                  int top_pad = 0,
                                  int left_pad = 0,
                                  float scale = 1.f);

        private:
            void decodeInstances(ncnn::Mat &data, std::vector<Instance> &instances) const;

            std::unique_ptr<ncnn::Net> net_{nullptr};
            std::vector<std::string> classes_;
            float score_threshold_{0.8};
            float iou_threshold_{0.2};
            unsigned short kpt_shape_[2];
    };
}


#endif // __YOLO_H__