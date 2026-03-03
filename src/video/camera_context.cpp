#include "video/camera_context.h"

namespace video {
	CameraContext::~CameraContext() {
        if (thread_running.load(std::memory_order_acquire))
            thread_running.store(false, std::memory_order_release);

        if (src)
            src->RequestStop();

        if (th.joinable())
            th.join();
	}
}