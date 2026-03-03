#pragma once
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "infer/letterbox.h"

struct InferResult {
    std::vector<Ort::Value> outputs; // 原始模型输出
    LetterBoxInfo lb;                // letterbox
    int orig_w = 0;
    int orig_h = 0;
};