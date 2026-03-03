#pragma once
#include <string>
#include "video/frame.h"

namespace video {

    class IVideoSource {
    public:
        virtual ~IVideoSource() = default;

        virtual bool Open(const std::string& url) = 0;
        virtual bool Read(Frame& out) = 0; // 返回 false 表示暂时读不到/结束/断流（由实现决定）
        virtual void Close() = 0;
    };

} // namespace video
