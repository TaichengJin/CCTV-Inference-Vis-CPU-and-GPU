#pragma once

struct Det {
    float x1, y1, x2, y2; // original image pixels
    int class_id;
    float score;
};

using Dets = std::vector<Det>;