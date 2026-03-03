#include <iostream>
#include <stdexcept>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <thread>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "infer/InferEngine.h"
#include "infer/postprocess_rtdetr.h"
#include "common/visualize.h"
#include "video/ffmpeg_video_source.h"
#include "video/camera_context.h"
#include "video/camera_manager.h"

enum class ExitCode : int {
    Ok = 0,
    InputError = 1,
    OrtError = 2,
    RuntimeError = 3
};

struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

int main() {
    try {
        const std::wstring model_path = L"models\\rtdetr-l.onnx";
        //const std::string image_path = "assets\\test_frame.png";

        InferEngine::Options opt;
        opt.input_w = 640;
        opt.input_h = 640;
        opt.use_cuda = false;

        double last_vis_ms = 0.0;

        InferEngine engine(opt);
        engine.LoadModel(model_path);



        PostprocessOptions pp;
        pp.score_thresh = 0.65f;

        cv::namedWindow("RT-DETR Live", cv::WINDOW_NORMAL);

        // Decode Thread
        // ª˘”⁄…„œÒÕ∑≤‚ ‘
        const std::vector<std::string> urls = {
            "rtsp://admin:%40%40admin7434@192.168.1.100:554/0/onvif/profile1/media.smp",
        };
        video::CameraManager mgr;
        mgr.Reserve(4);
        mgr.AddCamera(0, urls[0]);

        mgr.StartAllCams();

        // Display(Main Thread)
        std::vector<uint64_t> last_seq(mgr.Cameras().size(), 0);

        std::atomic<bool> running{ true };
        while (running.load()) {
            mgr.TickWatchdog();
            for (size_t i = 0; i < mgr.Cameras().size(); ++i) {
                std::optional<video::Frame> frame_now;
                uint64_t seq_now = 0;

                {
                    std::lock_guard<std::mutex> lk(mgr.Cameras()[i]->shared.frame_m);
                    if (mgr.Cameras()[i]->shared.latest_frame.has_value()) {
                        frame_now = mgr.Cameras()[i]->shared.latest_frame;
                        seq_now = mgr.Cameras()[i]->shared.frame_seq;
                    }
                }

                if (!frame_now.has_value() || frame_now->bgr.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                Dets dets_now;
                uint64_t det_seq = 0;
                {
                    std::lock_guard<std::mutex> lk(mgr.Cameras()[i]->shared.det_m);
                    dets_now = mgr.Cameras()[i]->shared.latest_dets;
                    det_seq = mgr.Cameras()[i]->shared.det_seq;
                }

                video::CameraContext::FrameStats stats_snap;
                {
                    std::lock_guard<std::mutex> lk(mgr.Cameras()[i]->shared.stats_m);
                    stats_snap = mgr.Cameras()[i]->shared.stats;
                }

                std::string hud = cv::format(
                    "dec:%.1fms  inf:%.1fms  vis:%.1fms",
                    stats_snap.decode_ms,
                    stats_snap.infer_ms,
                    stats_snap.vis_ms
                );

                auto tv0 = std::chrono::steady_clock::now();


                static cv::Mat vis;
                frame_now->bgr.copyTo(vis);

                int sta_margin = 10;
                DrawHudTopRight(vis, hud, sta_margin);

                cv::imshow("RT-DETR Live", vis);

                auto tv1 = std::chrono::steady_clock::now();

                {
                    std::lock_guard<std::mutex> lk(mgr.Cameras()[i]->shared.stats_m);
                    mgr.Cameras()[i]->shared.stats.vis_ms = std::chrono::duration<double, std::milli>(tv1 - tv0).count();
                    last_vis_ms = mgr.Cameras()[i]->shared.stats.vis_ms;
                }

            }

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                running.store(false);
                break;
            }
        }

        mgr.StopAllCams();

        cv::destroyAllWindows();



        /*if (!DrawAndSaveDetections(bgr, dets, "assets\\vis_result.jpg")) {
            throw std::runtime_error("Failed to write assets\\vis_result.jpg");
        }*/

        std::cout << "[INFO] Done.\n";
        return static_cast<int>(ExitCode::Ok);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ORT ERROR] " << e.what() << "\n";
        return static_cast<int>(ExitCode::OrtError);
    }
    catch (const InputError& e) {
        std::cerr << "[INPUT ERROR] " << e.what() << "\n";
        return static_cast<int>(ExitCode::InputError);
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return static_cast<int>(ExitCode::RuntimeError);
    }
}
