#pragma once
#include <atomic>
#include <vector>
#include <cstddef>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "InferEngine.h"

namespace video { class CameraContext; struct Frame; }

namespace Inference{
	class InferencePool {
	public:
		InferencePool();
		~InferencePool();

		void Configure(const std::wstring& model_path, const InferEngine::Options& opt);

		void SetCameras(const std::vector<video::CameraContext*>& cams);

		void StartThreadPool(int workers_num);
		void StopThreadPool();

		void OnPendingBecameNonEmpty();

	private:
		void WorkerLoop(int worker_id);

		bool TryPopPending(video::CameraContext*& out_ctx,
			video::Frame& out_frame,
			uint64_t& out_seq);

	private:

		struct Impl;						// 前置声明
		std::unique_ptr<Impl> impl_;		// 用于初始化Ort::env

		std::atomic<bool> pool_running_{ false };
		std::vector<std::thread> workers_;

		// 全局 wait/notify
		std::mutex pool_m_;
		std::condition_variable pool_cv_;

		std::atomic<int> pending_count_{ 0 };

		std::atomic<size_t> rr_cursor_{ 0 };

		std::vector<video::CameraContext*> cams_;

		std::wstring model_path_;
		InferEngine::Options opt_;

	};
}