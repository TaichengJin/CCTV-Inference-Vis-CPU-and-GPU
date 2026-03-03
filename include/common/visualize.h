#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "common/det.h"

void DrawDetections(cv::Mat& img, const std::vector<Det>& dets);

bool DrawAndSaveDetections(
    const cv::Mat& src_bgr,
    const std::vector<Det>& dets,
    const std::string& out_path
);

void DrawHudTopRight(
    cv::Mat& img,
    const std::string& text,
    int margin
);