#pragma once
#include <atomic>
#include <vector>
#include <cstddef>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace video { class CameraContext; struct Frame; }

namespace Inference{
	class InferencePool {
	public:
		InferencePool() = default;
		~InferencePool() { StopThreadPool(); }

		void SetCameras(const std::vector<video::CameraContext*>& cams);

		void StartThreadPool(int workers_num);
		void StopThreadPool();

		void NotifyWork();

	private:
		void WorkerLoop(int worker_id);

		bool TryPopPending(video::CameraContext*& out_ctx,
			video::Frame& out_frame,
			uint64_t& out_seq);

	private:
		std::atomic<bool> pool_running_{ false };
		std::vector<std::thread> workers_;

		// È«¾Ö wait/notify
		std::mutex pool_m_;
		std::condition_variable pool_cv_;

		std::atomic<int> work_signal_{ 0 };

		std::atomic<size_t> rr_cursor_{ 0 };

		std::vector<video::CameraContext*> cams_;

	};
}