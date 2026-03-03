#pragma once
#include <vector>
#include <onnxruntime_cxx_api.h>

#include "letterbox.h"
#include "common/det.h"

struct PostprocessOptions {
    float score_thresh = 0.49f;
    bool apply_sigmoid = true;   // RT-DETR 常见导出是 logits
};

// 解析 RT-DETR 输出（默认 outputs[0]） -> dets（原图坐标）
std::vector<Det> PostprocessRTDETR(
    const Ort::Value& out0,          // outputs[0]
    int input_w, int input_h,         // letterbox 目标尺寸（模型输入尺寸）
    const LetterBoxInfo& lb,          // preprocess 产生的 letterbox 信息
    int orig_w, int orig_h,           // 原图尺寸
    const PostprocessOptions& opt = {}
);
