#pragma once

struct LetterBoxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int dst_w = 0; // model input W
    int dst_h = 0; // model input H

    /*explicit LetterBoxInfo lbi(float scale, int pad_x, int pad_y);
    explicit LetterBoxInfo lbi(float scale, int pad_x, int pad_y, int dst_w);*/

};

