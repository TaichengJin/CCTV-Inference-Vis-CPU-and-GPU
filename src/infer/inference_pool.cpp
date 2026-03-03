#include "infer/inference_pool.h"
#include "video/camera_context.h" 
#include <chrono>

namespace Inference {

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

        workers_.clear();
        workers_.reserve(static_cast<size_t>(workers_num));

        for (int i = 0; i < workers_num; ++i) {
            workers_.emplace_back([this, i] { WorkerLoop(i); });
        }
    }

    void InferencePool::StopThreadPool() {
        bool expected = true;
        if (!pool_running_.compare_exchange_strong(expected, false)) {
            // 已经停止
            return;
        }

        // 唤醒所有 worker 让其退出
        {
            std::lock_guard<std::mutex> lk(pool_m_);
            work_signal_.store(1, std::memory_order_release);
        }
        pool_cv_.notify_all();

        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();

        // 可选：停止后清空视图
        // cams_.clear();
    }

    void InferencePool::NotifyWork() {
        // 发布端在写好某路 infer_pending 后调用即可
        work_signal_.fetch_add(1, std::memory_order_release);
        pool_cv_.notify_one();
    }

    bool InferencePool::TryPopPending(video::CameraContext*& out_ctx,
        video::Frame& out_frame,
        uint64_t& out_seq) {
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

            // move 出来 + reset：消费语义（同一帧只会被一个 worker 拿走）
            out_frame = std::move(*ctx->shared.infer_pending);
            ctx->shared.infer_pending.reset(); // optional的成员函数，触发Frame的析构

            out_seq = ctx->shared.infer_pending_seq;
            out_ctx = ctx;
            return true;
        }
        return false;
    }

    void InferencePool::WorkerLoop(int worker_id) {
        (void)worker_id;

        while (pool_running_.load(std::memory_order_acquire)) {
            // 1) 等待“可能有活”或 stop
            {
                std::unique_lock<std::mutex> ulk(pool_m_);
                pool_cv_.wait(ulk, [&] {
                    return !pool_running_.load(std::memory_order_acquire) ||
                        work_signal_.load(std::memory_order_acquire) > 0;
                    });

                if (!pool_running_.load(std::memory_order_acquire)) break;
            }

            // 2) 抢活：从多路 pending 里拿一帧
            video::CameraContext* ctx = nullptr;
            video::Frame frame;
            uint64_t seq = 0;

            bool got = TryPopPending(ctx, frame, seq);
            if (!got) {
                // 可能是“虚假唤醒”或其他 worker 已把活拿走
                // 这里把信号降下去，避免一直热醒
                work_signal_.store(0, std::memory_order_release);
                continue;
            }

            // 3) 锁外做慢推理（你先用占位）
            // TODO: preprocess -> session.Run -> postprocess
            // 先模拟耗时：
            // std::this_thread::sleep_for(std::chrono::milliseconds(5));

            // 4) 写回结果（按你的 SharedState 字段自己接）
            // 例如：
            // {
            //     std::lock_guard<std::mutex> lk(ctx->shared.result_m);
            //     ctx->shared.last_infer_seq = seq;
            //     ctx->shared.dets = ...;
            // }
            // ctx->shared.frame_cv.notify_one(); // 若显示端依赖结果更新，也可以通知

            // 5) 继续循环抢下一帧
        }
    }

} // namespace Inference