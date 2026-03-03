#include "infer/InferEngine.h"
#include <iostream>

namespace {

    cv::Mat LetterboxBGR(const cv::Mat& src_bgr, int dst_w, int dst_h, LetterBoxInfo& info) {
        int src_w = src_bgr.cols;
        int src_h = src_bgr.rows;

        float r = std::min(
            dst_w / static_cast<float>(src_w),
            dst_h / static_cast<float>(src_h)
        );
        int new_w = static_cast<int>(std::round(src_w * r));
        int new_h = static_cast<int>(std::round(src_h * r));

        cv::Mat resized;
        cv::resize(src_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        int pad_w = dst_w - new_w;
        int pad_h = dst_h - new_h;

        int pad_left = pad_w / 2;
        int pad_right = pad_w - pad_left;
        int pad_top = pad_h / 2;
        int pad_bottom = pad_h - pad_top;

        cv::Mat out;
        cv::copyMakeBorder(
            resized, out,
            pad_top, pad_bottom, pad_left, pad_right,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)
        );

        info.scale = r;
        info.pad_x = pad_left;
        info.pad_y = pad_top;
        info.dst_w = dst_w;
        info.dst_h = dst_h;
        return out;
    }

    void BGRToCHWFloat01_RGB(const cv::Mat& bgr, std::vector<float>& chw) {
        CV_Assert(bgr.type() == CV_8UC3);

        // output: [3,H,W], float, RGB, normalized to 0..1
        int H = bgr.rows;
        int W = bgr.cols;
        int HW = H * W;

        if ((int)chw.size() != 3 * HW) {
            throw std::runtime_error("chw must be preallocated to 3*H*W");
        }

        float* dstR = chw.data();
        float* dstG = chw.data() + HW;
        float* dstB = chw.data() + 2 * HW;

        for (int y = 0; y < H; ++y) {
            const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y); // 模板函数返回Vec3b*类型，包含y行的像素信息
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                const float inv255 = 1.0f / 255.0f;

                const float B = row[x][0] * inv255;
                const float G = row[x][1] * inv255;
                const float R = row[x][2] * inv255;

                dstR[idx] = R;
                dstG[idx] = G;
                dstB[idx] = B;
            }
        }
    }
}

InferEngine::InferEngine(const Options& opt)
    : opt_(opt),
    env_(ORT_LOGGING_LEVEL_WARNING, "CppInferDemo") { }

void InferEngine::LoadModel(const std::wstring& model_path) {
    session_opt_ = Ort::SessionOptions{};  // 确保可重复 Load / Reload

    session_opt_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (opt_.intra_op_num_threads > 0) {
        session_opt_.SetIntraOpNumThreads(opt_.intra_op_num_threads);
    }

    // TODO: opt_.use_cuda 为 true 时，在这里加 CUDA provider

    session_ = Ort::Session(env_, model_path.c_str(), session_opt_);

    // 读取输入输出名
    Ort::AllocatorWithDefaultOptions allocator;

    input_names_.clear();
    output_names_.clear();

    size_t num_inputs = session_.GetInputCount();
    
    if (num_inputs != 1) {
        throw std::runtime_error("InferEngine expects exactly 1 input tensor.");
    }

    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_names_.push_back(name.get()); // std::string 拷贝，OK
    }

    size_t num_outputs = session_.GetOutputCount();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_names_.push_back(name.get());
    }

    // 获取模型输入维度
    auto input_type_info = session_.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto shape = input_tensor_info.GetShape(); // [N,C,H,W]，可能有 -1
    
    if (shape.size() != 4) {
        throw std::runtime_error("InferEngine expects input rank 4: [N,C,H,W].");
    }

    int64_t N = shape[0] > 0 ? shape[0] : 1;
    int64_t C = shape[1] > 0 ? shape[1] : 3;
    int64_t Hm = shape[2] > 0 ? shape[2] : -1;
    int64_t Wm = shape[3] > 0 ? shape[3] : -1;

    if (N != 1 || C != 3) {
        throw std::runtime_error("InferEngine expects input shape [1,3,H,W].");
    }

    // 决定最终 preprocess 尺寸
    if (opt_.input_h > 0 && opt_.input_w > 0) {
        input_h_ = opt_.input_h;
        input_w_ = opt_.input_w;
    }
    else if (Hm > 0 && Wm > 0) {
        input_h_ = static_cast<int>(Hm);
        input_w_ = static_cast<int>(Wm);
    }
    else {
        input_h_ = 640;
        input_w_ = 640;
    }
}

void InferEngine::PrintModelInfo() const {
    std::cout << "Inputs:\n";
    for (size_t i = 0; i < input_names_.size(); ++i) {
        std::cout << "  [" << i << "] " << input_names_[i] << "\n";
    }
    std::cout << "Outputs:\n";
    for (size_t i = 0; i < output_names_.size(); ++i) {
        std::cout << "  [" << i << "] " << output_names_[i] << "\n";
    }
}

// 通过preprocessing得到符合模型输入的数据
std::vector<float> InferEngine::PreprocessToCHW(const cv::Mat& bgr, LetterBoxInfo& lb) const {
    cv::Mat lb_bgr = LetterboxBGR(bgr, input_w_, input_h_, lb);
    std::vector<float> input_chw(3 * input_h_ * input_w_);
    BGRToCHWFloat01_RGB(lb_bgr, input_chw);
    
    return input_chw;
}

InferResult InferEngine::Run(const cv::Mat& bgr) {
    InferResult r;
    r.orig_w = bgr.cols;
    r.orig_h = bgr.rows;

    // 1) preprocess
    r.lb = {};
    auto input_chw = PreprocessToCHW(bgr, r.lb);

    // 2) build input tensor
    std::array<int64_t, 4> input_shape{ 1, 3, input_h_, input_w_ };
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_chw.data(),
        input_chw.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 3) run
    // TODO; in_names换成更明确的变量名
    std::vector<const char*> in_names;
    in_names.reserve(input_names_.size());
    for (auto& s : input_names_) in_names.push_back(s.c_str());

    std::vector<const char*> out_names;
    out_names.reserve(output_names_.size());
    for (auto& s : output_names_) out_names.push_back(s.c_str());

    r.outputs = session_.Run(
        Ort::RunOptions{ nullptr },
        in_names.data(),
        &input_tensor,
        1,
        out_names.data(),
        out_names.size()
    );

    return r;
}
