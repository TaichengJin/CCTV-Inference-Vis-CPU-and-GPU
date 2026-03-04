#pragma once
#include"camera_context.h"


namespace Inference { class InferencePool; }

namespace video {
	class CameraManager {
	public:
		void AddCamera(int id, std::string url);
		void StartAllCams();
		void StopAllCams();
		//void StartCamera(CameraContext& cam, int cam_id, const std::string& url);
		//void StopCamera(CameraContext& cam);
		void TickWatchdog();
		const std::vector<CameraContext*>& CameraPtrs() const { return cam_ptrs_; }
		auto& Cameras() { return cams_; };
		static inline int64_t NowUs() {
			using namespace std::chrono;
			return duration_cast<microseconds>(
				steady_clock::now().time_since_epoch()
			).count();
		}
		void Reserve(size_t n) {
			cams_.reserve(n);
		}

		// 为了简化访问到线程池实例设置此函数
		void SetInferencePool(Inference::InferencePool* p);

	private:
		static void DecodeLoop(CameraContext* ctx);

	private:
		std::vector<std::unique_ptr<CameraContext>> cams_;
		// 为了外部访问而维护的raw指针
		std::vector<video::CameraContext*> cam_ptrs_;

		// 为了简化访问到线程池实例设置此指针
		Inference::InferencePool* pool_ = nullptr;
	};

}