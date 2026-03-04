#pragma once
#include <iostream>
#include <optional>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "ffmpeg_video_source.h"
#include "common/det.h"

namespace Inference { class InferencePool; }
namespace video {
	class CameraContext {

	public:

		struct Policy {
			int fail_threshold = 30;

			std::chrono::milliseconds small_fail_sleep{ 10 };

			int64_t open_timeout_us = 8'000'000;
			int64_t open_kick_cooldown_us = 1'000'000;

			int64_t no_frame_timeout_us = 2'000'000;
			int64_t kick_cooldown_us = 500'000;

		};
		struct CameraRuntime {

			enum class State : uint8_t
			{
				Closed,
				Opening,
				Running,
				Reconnecting,
				Stopping,
				Stopped
			};

			std::atomic<State> state{ State::Closed };
			
			std::atomic<int64_t> last_ok_us;

			std::atomic<int64_t> last_kick_us{ 0 };

			std::atomic<int64_t> open_start_us{ 0 };
			std::atomic<int64_t> last_open_kick_us{ 0 };

			std::chrono::milliseconds backoff{ 100 };
			std::chrono::milliseconds backoff_max{ 2000 };
			std::chrono::milliseconds backoff_init{ 100 };
			
			///
			// backoff部分，避免open()或read()失败立即反复重连
			///

			int fail_count = 0;
			int reconnect_count = 0;
			int open_fail_count = 0;
		};

		struct FrameStats {
			double decode_ms = 0.0;   // Read()
			double infer_ms = 0.0;   // Run + Postprocess
			double vis_ms = 0.0;   // Draw + imshow
			int64_t latency_ms = 0;   // NowUs - pts_us
		};

		struct SharedState {
			// latest frame (capacity=1)
			std::mutex frame_m;
			std::condition_variable frame_cv;
			std::optional<video::Frame> latest_frame;  // “稳定帧”
			uint64_t frame_seq = 0;             // 每发布一次 +1

			// infer pending(capacity=1)
			std::mutex infer_pending_m;
			std::optional<video::Frame> infer_pending;
			uint64_t infer_pending_seq = 0;

			// latest dets
			std::mutex det_m;
			Dets latest_dets;
			uint64_t det_seq = 0;               // det 对应的 frame_seq（推理使用的那帧）

			// stats
			std::mutex stats_m;
			FrameStats stats;
		};

		CameraContext() = default;
		~CameraContext();

		int cam_id = -1;
		SharedState shared;
		CameraRuntime camera_runtime;
		Policy camera_policy;
		std::string url;
		std::unique_ptr<video::FFmpegVideoSource> src;
		std::atomic<bool> thread_running{ false };
		std::thread th;

		Inference::InferencePool* pool = nullptr;

	};
}