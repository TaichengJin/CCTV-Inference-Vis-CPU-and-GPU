#pragma once
#include <cstdint>
#include <opencv2/opencv.hpp>

namespace video {

    enum class PixelFormat {
        Unknown,
        BGR24,     // cv::Mat(CV_8UC3)
        // 未来扩展：
        // NV12,
        // YUV420P,
    };

    struct FrameStats {
        double decode_ms = 0.0;   // Read()
        double infer_ms = 0.0;   // Run + Postprocess
        double vis_ms = 0.0;   // Draw + imshow
        int64_t latency_ms = 0;   // NowUs - pts_us
    };

    struct Frame {
        PixelFormat format = PixelFormat::Unknown;
        int width = 0;
        int height = 0;

        // 时间戳（微秒），用于延迟统计、同步、多路对齐等
        int64_t pts_us = 0;

        // 现阶段：直接给上层 cv::Mat（BGR）
        // 将来：可以加 union/variant 或者新增 GPU buffer 字段
        cv::Mat bgr;

        FrameStats stats;

        bool empty() const { return bgr.empty() || width <= 0 || height <= 0; }
    };



} // namespace video
