#pragma once
#include"camera_context.h"

namespace video {
	class CameraManager {
	public:
		void AddCamera(int id, std::string url);
		void StartAllCams();
		void StopAllCams();
		//void StartCamera(CameraContext& cam, int cam_id, const std::string& url);
		//void StopCamera(CameraContext& cam);
		void TickWatchdog();
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

	private:
		static void DecodeLoop(CameraContext* ctx);

	private:
		std::vector<std::unique_ptr<CameraContext>> cams_;
	};

}