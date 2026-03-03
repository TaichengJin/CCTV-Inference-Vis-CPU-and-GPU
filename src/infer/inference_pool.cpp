#include <chrono>
#include <onnxruntime_cxx_api.h>

#include "infer/inference_pool.h"
#include "video/camera_context.h" 
#include "infer/postprocess_rtdetr.h"

namespace Inference {

    struct InferencePool::Impl {
        Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "CppInferDemo" };
    };

    InferencePool::InferencePool()
        : impl_(std::make_unique<Impl>()) { }

    InferencePool::~InferencePool() {
        StopThreadPool();
    }

    void InferencePool::Configure(const std::wstring& model_path,
        const InferEngine::Options& opt)
    {
        model_path_ = model_path;
        opt_ = opt;
    }

    void InferencePool::SetCameras(const std::vector<video::CameraContext*>& cams) {
        // 调用方保证 StopThreadPool() 后再改 cameras
        cams_ = cams;
    }

    void InferencePool::StartThreadPool(int workers_num) {
        if (workers_num <= 0) workers_num = 1;

        bool expected = false;
        if (!pool_running_.compare_exchange_strong(expected, true)) {
            // 已经在跑，不重复启动
            return;
        }

        const std::wstring model_path = L"models\\rtdetr-l.onnx";

        InferEngine::Options opt;
        opt.input_w = 640;
        opt.input_h = 640;
        opt.use_cuda = false;

        Configure(model_path, opt);

        workers_.clear();
        workers_.reserve(static_cast<size_t>(workers_num));

        for (int i = 0; i < workers_num; ++i) {
            workers_.emplace_back([this, i] { 
                WorkerLoop(i); });
        }
    }

    void InferencePool::StopThreadPool() {
        bool expected = true;
        if (!pool_running_.compare_exchange_strong(expected, false)) {
            // 已经停止
            return;
        }

        pool_cv_.notify_all();

        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();

        // 可选：停止后清空视图
        // cams_.clear();
    }

    void InferencePool::OnPendingBecameNonEmpty() {
        // 发布端在写好某路 infer_pending 后调用
        pending_count_.fetch_add(1, std::memory_order_release);
        pool_cv_.notify_one();
    }

    bool InferencePool::TryPopPending(video::CameraContext*& out_ctx,
        video::Frame& out_frame,
        uint64_t& out_seq) {
        // 防止在引用逻辑修改后第二次循环使用旧值的bug
        out_ctx = nullptr;
        out_seq = 0;

        const size_t n = cams_.size();
        if (n == 0) return false;

        // round-robin 起点（每次递增，避免总从 0 开始）
        size_t start = rr_cursor_.fetch_add(1, std::memory_order_relaxed);
        start %= n;

        for (size_t k = 0; k < n; ++k) {
            size_t i = (start + k) % n;
            auto* ctx = cams_[i];
            if (!ctx) continue;

            // 尝试拿这一路的 pending
            std::lock_guard<std::mutex> lk(ctx->shared.infer_pending_m);

            if (!ctx->shared.infer_pending.has_value()) continue;

            // move出来 + reset，消费语义（同一帧只会被一个 worker 拿走）
            out_seq = ctx->shared.infer_pending_seq;

            out_frame = std::move(*ctx->shared.infer_pending);
            ctx->shared.infer_pending.reset(); // optional的成员函数，触发Frame的析构

            // 为了防止pending count小于0进行断言，稳定后删除
            auto v = pending_count_.fetch_sub(1, std::memory_order_release) - 1;
            assert(v >= 0);

            out_ctx = ctx;
            return true;
        }
        return false;
    }

    void InferencePool::WorkerLoop(int worker_id) {
        (void)worker_id;

        InferEngine engine(impl_->env, opt_);
        engine.LoadModel(model_path_);
        PostprocessOptions pp;
        pp.score_thresh = 0.65f;

        while (pool_running_.load(std::memory_order_acquire)) {
            // 判断是否有pending或者pool_running_==false则立刻跳出等待并立刻break
            {
                std::unique_lock<std::mutex> ulk(pool_m_);
                pool_cv_.wait(ulk, [&] {
                    return !pool_running_.load(std::memory_order_acquire) ||
                        pending_count_.load(std::memory_order_acquire) > 0;
                    });

                if (!pool_running_.load(std::memory_order_acquire)) break;
            }

            video::CameraContext* ctx = nullptr;
            video::Frame frame;
            uint64_t seq = 0;

            bool got = TryPopPending(ctx, frame, seq);
            if (!got) continue;

            auto ti0 = std::chrono::steady_clock::now();

            InferResult result = engine.Run(frame.bgr);

            auto dets = PostprocessRTDETR(
                result.outputs[0],
                engine.InputW(), engine.InputH(),
                result.lb,
                result.orig_w, result.orig_h,
                pp
            );
            auto ti1 = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lk(ctx->shared.stats_m);
                ctx->shared.stats.infer_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();
            }

            {
                std::lock_guard<std::mutex> lk(ctx->shared.det_m);
                ctx->shared.latest_dets = std::move(dets);
                ctx->shared.det_seq = seq; // 这份 det 是用哪一帧推理出来的
            }
        }
    }

}