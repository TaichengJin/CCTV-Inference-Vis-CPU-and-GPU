#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "infer/postprocess_rtdetr.h"

static inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<Det> PostprocessRTDETR(
    const Ort::Value& out0,
    int input_w, int input_h,
    const LetterBoxInfo& lb,
    int orig_w, int orig_h,
    const PostprocessOptions& opt
) {
    if (!out0.IsTensor()) {
        throw std::runtime_error("RT-DETR output is not a tensor.");
    }

    auto info = out0.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error("Unexpected output rank (expect 3).");
    }

    int64_t num_queries = shape[1];
    int64_t dim = shape[2];
    if (dim < 6) {
        throw std::runtime_error("Unexpected output dim (<6).");
    }

    int class_count = (int)(dim - 4);

    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Output tensor is not float.");
    }

    const float* out_data = out0.GetTensorData<float>();

    std::vector<Det> dets;
    dets.reserve((size_t)num_queries);

    // W/H ÓÃ input_w/input_h£¨letterbox Ä¿±ê³ß´ç£©
    const float W = (float)input_w;
    const float H = (float)input_h;

    for (int64_t q = 0; q < num_queries; ++q) {
        const float* row = out_data + q * dim; // batch=1

        float cx = row[0];
        float cy = row[1];
        float bw = row[2];
        float bh = row[3];

        int best_id = -1;
        float best_logit = -1e9f;
        for (int c = 0; c < class_count; ++c) {
            float v = row[4 + c];
            if (v > best_logit) { best_logit = v; best_id = c; }
        }

        float score = opt.apply_sigmoid ? Sigmoid(best_logit) : best_logit;
        if (score < opt.score_thresh) continue;

        // normalized cxcywh -> letterbox pixel coords
        float x1 = (cx - bw * 0.5f) * W;
        float y1 = (cy - bh * 0.5f) * H;
        float x2 = (cx + bw * 0.5f) * W;
        float y2 = (cy + bh * 0.5f) * H;

        // undo letterbox
        x1 = (x1 - (float)lb.pad_x) / lb.scale;
        y1 = (y1 - (float)lb.pad_y) / lb.scale;
        x2 = (x2 - (float)lb.pad_x) / lb.scale;
        y2 = (y2 - (float)lb.pad_y) / lb.scale;

        // clamp to original image
        x1 = std::max(0.0f, std::min(x1, (float)(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(orig_w - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(orig_h - 1)));

        if (x2 <= x1 || y2 <= y1) continue;

        dets.push_back({ x1, y1, x2, y2, best_id, score });
    }

    return dets;
}
