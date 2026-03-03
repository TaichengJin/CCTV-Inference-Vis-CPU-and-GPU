// #include <functional> //可以用引用和std::ref(cam)，而不用直接传入指针
#include <thread>
#include "video/camera_manager.h"
#include "video/camera_context.h"
#include "infer/inference_pool.h"

namespace video {
	
	void CameraManager::TickWatchdog()
	{
		const int64_t now_us = NowUs();

		for (auto& cam : cams_)
		{
			if (!cam || !cam->src) continue;

			auto st = cam->camera_runtime.state.load(std::memory_order_relaxed);

			switch (st)
			{
			case CameraContext::CameraRuntime::State::Opening:
			{
				const int64_t t0 = cam->camera_runtime.open_start_us.load(std::memory_order_relaxed);
				if (t0 == 0) break;

				if (now_us - t0 >= cam->camera_policy.open_timeout_us)
				{
					const int64_t last_kick = cam->camera_runtime.last_open_kick_us.load(std::memory_order_relaxed);
					if (last_kick == 0 || now_us - last_kick >= cam->camera_policy.open_kick_cooldown_us)
					{
						cam->camera_runtime.last_open_kick_us.store(now_us, std::memory_order_relaxed);
						cam->src->RequestStop(); // 打断 open_input / find_stream_info
					}
				}
				break;
			}

			case CameraContext::CameraRuntime::State::Running:
			{
				const int64_t last_us = cam->camera_runtime.last_ok_us.load(std::memory_order_relaxed);
				if (last_us == 0) break;

				if (now_us - last_us >= cam->camera_policy.no_frame_timeout_us)
				{
					const int64_t last_kick = cam->camera_runtime.last_kick_us.load(std::memory_order_relaxed);
					if (last_kick == 0 || now_us - last_kick >= cam->camera_policy.kick_cooldown_us)
					{
						cam->camera_runtime.last_kick_us.store(now_us, std::memory_order_relaxed);
						cam->src->RequestStop(); // 打断 av_read_frame
					}
				}
				break;
			}

			default:
				// Reconnecting/Closed/Stopping 等：watchdog 不做事
				break;
			}
		}
	}

	static void SleepInterruptible(CameraContext* ctx, std::chrono::milliseconds timeout) {
		auto deadline = std::chrono::steady_clock::now() + timeout;
		while (ctx->thread_running.load(std::memory_order_acquire)) {
			auto now = std::chrono::steady_clock::now();
			if (now >= deadline) break;
			auto chunk = std::min<std::chrono::milliseconds>(
				std::chrono::milliseconds(50),
				std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
			std::this_thread::sleep_for(chunk);
		}
	}

	void CameraManager::SetInferencePool(Inference::InferencePool* p) { pool_ = p; }

	/// <summary>
	/// 参数是指针，thread使用的是cam.get()，查一下区别
	/// </summary>
	/// <param name="ctx">ctx explanation</param>
	/// <returns>no return</returns>
	void CameraManager::DecodeLoop(CameraContext* ctx) {
		ctx->thread_running.store(true, std::memory_order_release);
		
		// 初始化运行状态
		ctx->camera_runtime.state.store(CameraContext::CameraRuntime::State::Opening, std::memory_order_release);
		ctx->camera_runtime.fail_count = 0;
		// watchdog访问
		auto now_us = NowUs();
		ctx->camera_runtime.last_ok_us.store(now_us, std::memory_order_relaxed);

		Frame tmp;

		while (ctx->thread_running.load(std::memory_order_acquire)) {

			auto state_now = ctx->camera_runtime.state.load(std::memory_order_acquire);
			if (state_now == CameraContext::CameraRuntime::State::Opening) {
				
				std::cout << "Cam " << ctx->cam_id << " Opening..." << std::endl;
				bool ok_open = ctx->src->Open(ctx->url);
				std::cout << "Cam " << ctx->cam_id << " Open ret=" << ok_open << std::endl;

				if (!ok_open) {
					ctx->camera_runtime.open_fail_count++;
					ctx->camera_runtime.state.store(
						CameraContext::CameraRuntime::State::Reconnecting,
						std::memory_order_release);
					continue;
				}
				
				// 成功Open()
				auto now_us = NowUs();
				ctx->camera_runtime.open_fail_count = 0;
				ctx->camera_runtime.last_open_kick_us.store(0);
				ctx->camera_runtime.open_start_us.store(0);
				ctx->camera_runtime.last_ok_us.store(now_us, std::memory_order_relaxed);
				ctx->camera_runtime.backoff = ctx->camera_runtime.backoff_init;
				ctx->camera_runtime.state.store(
					CameraContext::CameraRuntime::State::Running,
					std::memory_order_release);
				continue; // 重新进入循环确认是否有StopCamera()
			}

			// Running
			if (state_now == CameraContext::CameraRuntime::State::Running) {
				auto td0 = std::chrono::steady_clock::now();
				bool ok = ctx->src->Read(tmp);
				auto td1 = std::chrono::steady_clock::now();

				auto now_us = NowUs();

				if (!ok) {
					ctx->camera_runtime.fail_count++;

					auto last_us = ctx->camera_runtime.last_ok_us.load(std::memory_order_relaxed);

					bool timeout =
						(last_us != 0) &&
						(now_us - last_us >= ctx->camera_policy.no_frame_timeout_us);

					if (ctx->camera_runtime.fail_count >= ctx->camera_policy.fail_threshold || timeout) {
						ctx->camera_runtime.state.store(
							CameraContext::CameraRuntime::State::Reconnecting,
							std::memory_order_release);
						continue;
					}

					SleepInterruptible(ctx, ctx->camera_policy.small_fail_sleep);
					continue;
				}
				
				// 确保Read()成功再计算时间
				{
					std::lock_guard<std::mutex> lk(ctx->shared.stats_m);
					ctx->shared.stats.decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
				}

				ctx->camera_runtime.fail_count = 0;
				ctx->camera_runtime.last_ok_us.store(now_us, std::memory_order_relaxed);
				ctx->camera_runtime.backoff = ctx->camera_runtime.backoff_init;

				// 发布稳定帧
				Frame published;
				uint64_t now_seq;
				Frame snap;
				published.width = tmp.bgr.cols;
				published.height = tmp.bgr.rows;
				published.bgr = tmp.bgr.clone();

				{
					std::lock_guard<std::mutex> lk(ctx->shared.frame_m);
					ctx->shared.latest_frame = std::move(published); // 覆盖旧帧（capacity=1）
					snap = *ctx->shared.latest_frame;
					now_seq = ++ctx->shared.frame_seq;
				}

				bool need_notify = false;

				{
					std::lock_guard<std::mutex> lk(ctx->shared.infer_pending_m);
					auto was_empty = !ctx->shared.infer_pending.has_value();
					ctx->shared.infer_pending = std::move(snap);
					ctx->shared.infer_pending_seq = now_seq;

					need_notify = was_empty;
				}

				if (need_notify && pool_) {
					pool_->OnPendingBecameNonEmpty();
				}

				ctx->shared.frame_cv.notify_one();


				continue;
			}

			// Reconnecting
			if (state_now == CameraContext::CameraRuntime::State::Reconnecting) {
				ctx->camera_runtime.reconnect_count++;
				std::cout << ctx->camera_runtime.reconnect_count << "\n";
				// 必须先Close兜底
				ctx->src->Close();
				SleepInterruptible(ctx, ctx->camera_runtime.backoff);

				// 退避指数增长并规定上限
				auto next = ctx->camera_runtime.backoff * 2;
				ctx->camera_runtime.backoff = (next > ctx->camera_runtime.backoff_max) ? ctx->camera_runtime.backoff_max : next;

				// 回到Opening尝试Open()
				ctx->camera_runtime.open_start_us.store(NowUs(), std::memory_order_relaxed);
				ctx->camera_runtime.state.store(
					CameraContext::CameraRuntime::State::Opening,
					std::memory_order_release);
					continue;
			}
			// 暂时没有实际功能，后续将状态变换加到StopCamera中
			if (state_now == CameraContext::CameraRuntime::State::Stopping) break;
		}

		ctx->src->Close();
		ctx->camera_runtime.state.store(CameraContext::CameraRuntime::State::Closed, std::memory_order_release);
		// 主线程统一回收资源
		//ctx->src->Close();
	}

	void video::CameraManager::AddCamera(int id, std::string url)
	{
		auto cam = std::make_unique<CameraContext>();
		cam->cam_id = id;
		cam->url = std::move(url);
		cams_.push_back(std::move(cam));
	}

	void video::CameraManager::StartAllCams()
	{
		for (auto& cam : cams_)
		{
			if (!cam) continue;

			// 已经启动就跳过（防止重复 start）
			if (cam->thread_running.load(std::memory_order_acquire))
				continue;

			cam->thread_running.store(true, std::memory_order_release);

			cam->src = std::make_unique<FFmpegVideoSource>();

			cam->th = std::thread(&video::CameraManager::DecodeLoop, cam.get());
		}
	}

	void video::CameraManager::StopAllCams()
	{
		for (auto& cam : cams_)
		{
			if (!cam) continue;
			cam->thread_running.store(false, std::memory_order_release);
			if (cam->src) cam->src->RequestStop();
		}

		// 统一join
		for (auto& cam : cams_)
		{
			if (!cam) continue;
			if (cam->th.joinable()) cam->th.join();
		}

		// 统一Close + 释放资源
		for (auto& cam : cams_)
		{
			if (!cam) continue;
			if (cam->src) cam->src->Close();
			cam->src.reset();
		}
	}

	//void StartCamera(
	//	CameraContext& cam, int cam_id, 
	//	const std::string& url) {
	//	cam.cam_id = cam_id;
	//	cam.url = url;
	//	cam.src = std::make_unique<FFmpegVideoSource>();
	//	cam.th = std::thread(DecodeLoop, &cam);
	//}
	//void StopCamera(CameraContext& cam) {
	//	cam.thread_running.store(false, std::memory_order_release);

	//	if (cam.src) cam.src->RequestStop(); // 解除阻塞

	//	if (cam.th.joinable()) cam.th.join(); // 等线程真的退出

	//	if (cam.src) cam.src->Close(); // 释放资源
	//}
}
