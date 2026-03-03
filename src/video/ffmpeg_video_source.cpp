#include "video/ffmpeg_video_source.h"
#include <stdexcept>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
}

namespace video {

    struct FFmpegNetworkGuard {
        FFmpegNetworkGuard() { avformat_network_init(); }
        ~FFmpegNetworkGuard() { avformat_network_deinit(); }
        FFmpegNetworkGuard(const FFmpegNetworkGuard&) = delete;
        FFmpegNetworkGuard& operator=(const FFmpegNetworkGuard&) = delete;
    };
    void EnsureFFmpegNetwork()
    {
        static FFmpegNetworkGuard guard; // static保证只创建一次
    }

    static int64_t ToUs(int64_t pts, AVRational time_base) {
        if (pts == AV_NOPTS_VALUE) return 0;
        // pts * time_base => seconds
        double sec = pts * av_q2d(time_base);
        return static_cast<int64_t>(sec * 1000000.0);
    }

    FFmpegVideoSource::FFmpegVideoSource() {
    //    // 新版 FFmpeg 通常不需要显式 av_register_all()
    //    /*初始化 socket / 网络库
    //    注册 RTSP / RTP / HTTP / TCP / UDP 等协议
    //    在 Windows 下处理 WSAStartup 等平台相关事项*/
    //    avformat_network_init(); // 进程粒度不需要每个摄像头环境初始化
        EnsureFFmpegNetwork();
    }

    FFmpegVideoSource::~FFmpegVideoSource() {
        Close();
        // 多摄像头会导致deinit()多次导致环境崩溃
        // 之后单独初始化环境(进程粒度)
        //avformat_network_deinit(); // 多线程不需要在析构函数中
    }

    bool FFmpegVideoSource::Open(const std::string& url) {
        Close();
        last_error_.clear();
        abort_.store(false, std::memory_order_release);

        // 为了在open阶段Interrupt生效
        fmt_ = avformat_alloc_context();
        if (!fmt_) {
            last_error_ = "FFmpeg: avformat_alloc_context failed.";
            return false;
        }

        fmt_->interrupt_callback.callback = &FFmpegVideoSource::InterruptCallback;
        fmt_->interrupt_callback.opaque = this;

        // 1) 打开输入（RTSP）
        AVDictionary* opts = nullptr;
        // 低延迟常用参数（先给保守默认，后续你可以系统化调参）
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);   // tcp 更稳（udp 延迟低但丢包敏感）
        av_dict_set(&opts, "stimeout", "5000000", 0);     // 5s 超时（微秒）
        av_dict_set(&opts, "max_delay", "500000", 0);     // 0.5s（有些流有效）
        // av_dict_set(&opts, "fflags", "nobuffer", 0);   // 更激进，先别急

        std::cout << "[Open] step1 before open_input" << std::endl;
        int ret = avformat_open_input(&fmt_, url.c_str(), nullptr, &opts);
        std::cout << "[Open] step1 after open_input ret=" << ret << std::endl;
        av_dict_free(&opts);

        if (ret < 0) {
            SetError("FFmpeg: avformat_open_input failed.");
            Cleanup();
            return false;
        }

        // 2) 读流信息
        std::cout << "[Open] step2 before find_stream_info" << std::endl;
        ret = avformat_find_stream_info(fmt_, nullptr);
        std::cout << "[Open] step2 after find_stream_info ret=" << ret << std::endl;
        if(ret < 0) {
            SetError("FFmpeg: avformat_find_stream_info failed.");
            Cleanup();
            return false;
        }

        // 3) 找到视频流
        video_stream_index_ = -1;
        for (unsigned i = 0; i < fmt_->nb_streams; ++i) {
            if (fmt_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index_ = static_cast<int>(i);
                break;
            }
        }
        if (video_stream_index_ < 0) {
            SetError("FFmpeg: no video stream found.");
            Cleanup();
            return false;
        }

        // 4) 创建解码器
        AVCodecParameters* par = fmt_->streams[video_stream_index_]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(par->codec_id);
        if (!codec) {
            SetError("FFmpeg: decoder not found.");
            Cleanup();
            return false;
        }

        dec_ = avcodec_alloc_context3(codec);
        if (!dec_) {
            SetError("FFmpeg: avcodec_alloc_context3 failed.");
            Cleanup();
            return false;
        }

        // par: stream's codec parameters (demux layer description: codec_id, width/height, extradata...)
        // dec_: decoder instance (decode layer state machine). It needs par to configure itself before open2.
        if (avcodec_parameters_to_context(dec_, par) < 0) {
            SetError("FFmpeg: avcodec_parameters_to_context failed.");
            Cleanup();
            return false;
        }

        // 可选：降低延迟（不保证所有场景有效）
        dec_->flags |= AV_CODEC_FLAG_LOW_DELAY;

        if (avcodec_open2(dec_, codec, nullptr) < 0) {
            SetError("FFmpeg: avcodec_open2 failed.");
            Cleanup();
            return false;
        }

        // 5) 分配 packet / frame
        pkt_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        if (!pkt_ || !frame_) {
            SetError("FFmpeg: alloc packet/frame failed.");
            Cleanup();
            return false;
        }

        return true;
    }

    bool FFmpegVideoSource::Read(Frame& out) {
        if (!fmt_ || !dec_ || video_stream_index_ < 0) return false;

        // 不断读包，直到解出一帧
        while (true) {
            int ret = av_read_frame(fmt_, pkt_);
            if (ret < 0) {
                // 读不到了：可能 EOF/断流/暂时无数据
                return false;
            }

            // 保留视频流，其他的省略跳过
            if (pkt_->stream_index != video_stream_index_) {
                av_packet_unref(pkt_);  // 将指针pkt_的计数减至1并清空
                continue;
            }

            // 送入解码器
            ret = avcodec_send_packet(dec_, pkt_);
            av_packet_unref(pkt_);
            if (ret < 0) {
                // 送包失败（可视为断流/数据错误）
                return false;
            }

            // 从解码器拿帧（可能一次 send 对应多次 receive）
            ret = avcodec_receive_frame(dec_, frame_);
            if (ret == AVERROR(EAGAIN)) {
                // 需要更多 packet
                continue;
            }
            if (ret < 0) {
                // 解码失败或结束
                return false;
            }

            // 拿到一帧：frame_ 通常是 YUV420P / NV12 等
            const int src_w = frame_->width;
            const int src_h = frame_->height;
            const int src_fmt = frame_->format;

            // 初始化/更新 sws（YUV -> BGR）
            InitScalerIfNeeded(src_w, src_h, src_fmt);

            // 准备输出 Mat（BGR）
            if (bgr_.empty() || bgr_.cols != src_w || bgr_.rows != src_h) {
                bgr_.create(src_h, src_w, CV_8UC3);
            }

            uint8_t* dst_data[4] = { bgr_.data, nullptr, nullptr, nullptr };
            int dst_linesize[4] = { static_cast<int>(bgr_.step), 0, 0, 0 };

            sws_scale(
                sws_,
                frame_->data,
                frame_->linesize,
                0,
                src_h,
                dst_data,
                dst_linesize
            );

            // 输出 Frame
            out.format = PixelFormat::BGR24;
            out.width = src_w;
            out.height = src_h;

            // pts 转微秒（用于统计延迟/同步）
            AVRational tb = fmt_->streams[video_stream_index_]->time_base;
            out.pts_us = ToUs(frame_->best_effort_timestamp, tb);

            out.bgr = bgr_; // 浅拷贝：cv::Mat 引用计数（此处复用缓冲区，单线程 OK）
            av_frame_unref(frame_);
            return true;
        }
    }

    void FFmpegVideoSource::RequestStop() {
        abort_.store(true, std::memory_order_release);
    }

    int FFmpegVideoSource::InterruptCallback(void* opaque) {
        auto* self = static_cast<FFmpegVideoSource*>(opaque);
        return self->abort_.load(std::memory_order_relaxed) ? 1 : 0;
    }

    // 必须在退出线程之后使用
    void FFmpegVideoSource::Close() {
        Cleanup();
        abort_.store(false, std::memory_order_release);
    }

    void FFmpegVideoSource::Cleanup() {
        if (sws_) { sws_freeContext(sws_); sws_ = nullptr; }

        if (frame_) { av_frame_free(&frame_); frame_ = nullptr; }
        if (pkt_) { av_packet_free(&pkt_); pkt_ = nullptr; }

        if (dec_) { avcodec_free_context(&dec_); dec_ = nullptr; }

        if (fmt_) { avformat_close_input(&fmt_); fmt_ = nullptr; }

        video_stream_index_ = -1;
        bgr_.release();
    }

    void FFmpegVideoSource::InitScalerIfNeeded(int src_w, int src_h, int src_pix_fmt)
    {
        const AVPixelFormat dst_fmt = AV_PIX_FMT_BGR24;

        if (sws_) {
            sws_freeContext(sws_);
            sws_ = nullptr;
        }

        // -------- 关键：把 deprecated 的 YUVJ* 映射为正常 YUV* + full range --------
        AVPixelFormat src_fmt = static_cast<AVPixelFormat>(src_pix_fmt);

        // 默认：视频更常见 limited；但遇到 YUVJ* 我们明确改成 full
        int srcRange = 0;

        if (src_fmt == AV_PIX_FMT_YUVJ420P) { src_fmt = AV_PIX_FMT_YUV420P; srcRange = 1; }
        else if (src_fmt == AV_PIX_FMT_YUVJ422P) { src_fmt = AV_PIX_FMT_YUV422P; srcRange = 1; }
        else if (src_fmt == AV_PIX_FMT_YUVJ444P) { src_fmt = AV_PIX_FMT_YUV444P; srcRange = 1; }
        else {
            // 非 YUVJ*：按 frame_ 的 color_range 判断（如果你有 frame_）
            if (frame_) {
                if (frame_->color_range == AVCOL_RANGE_JPEG) srcRange = 1;
                else if (frame_->color_range == AVCOL_RANGE_MPEG) srcRange = 0;
                else srcRange = 0; // 不确定时默认 limited（更常见）
            }
        }

        sws_ = sws_getContext(
            src_w, src_h, src_fmt,
            src_w, src_h, dst_fmt,
            SWS_BILINEAR,
            nullptr, nullptr, nullptr
        );
        if (!sws_) throw std::runtime_error("FFmpeg: sws_getContext failed.");

        // -------- 设置 colorspace + range（让颜色更稳）--------
        int dstRange = 1; // BGR/RGB 通常按 full

        AVColorSpace cs = AVCOL_SPC_BT709;
        if (frame_) {
            cs = frame_->colorspace;
            if (cs == AVCOL_SPC_UNSPECIFIED) cs = AVCOL_SPC_BT709;
        }

        const int* coeffs = sws_getCoefficients(cs);
        sws_setColorspaceDetails(
            sws_,
            coeffs, srcRange,
            coeffs, dstRange,
            0, 1 << 16, 1 << 16
        );
    }

} // namespace video
