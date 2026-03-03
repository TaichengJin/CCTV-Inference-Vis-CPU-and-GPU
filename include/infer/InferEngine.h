#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

#include "letterbox.h"
#include "Infer_result.h"

class InferEngine {
public:
    struct Options {
        int input_w = 640;
        int input_h = 640;
        bool use_cuda = false;     // 你目前是 CPU，就先 false
        int intra_op_num_threads = 0; // 0=ORT 自己决定
    };

    explicit InferEngine(const Options& opt = Options());

    void LoadModel(const std::wstring& model_path);

    //
    
    // 最小版本：先返回原始输出 tensors（以后再变成 Detections）
    InferResult Run(const cv::Mat& bgr);

    int InputW() const { return input_w_; }
    int InputH() const { return input_h_; }

    // 打印模型 IO 信息
    void PrintModelInfo() const;

private:
    std::vector<float> PreprocessToCHW(const cv::Mat& bgr, LetterBoxInfo& lb) const;

private:
    Options opt_;
    int input_w_ = 0, input_h_ = 0;

    Ort::Env env_;
    Ort::SessionOptions session_opt_;
    Ort::Session session_{ nullptr };

    // IO names（用 string 保存，避免生命周期问题）
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};