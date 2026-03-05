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
#include "infer/inference_pool.h"

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
        opt.use_cuda = true;
        opt.intra_op_num_threads = 0;

        double last_vis_ms = 0.0;

        Inference::InferencePool pool;
        pool.Configure(model_path, opt);

        const std::vector<std::string> urls = {
            "rtsp://admin:%40%40admin7434@192.168.1.100:554/0/onvif/profile2/media.smp",
            "rtsp://admin:%40%40admin7434@192.168.1.100:554/0/onvif/profile2/media.smp",
            "rtsp://admin:%40%40admin7434@192.168.1.100:554/0/onvif/profile2/media.smp",
            "rtsp://admin:%40%40admin7434@192.168.1.100:554/0/onvif/profile2/media.smp"
        };
        video::CameraManager mgr;
        mgr.Reserve(4);
        mgr.SetInferencePool(&pool);

        mgr.AddCamera(0, urls[0]);
        mgr.AddCamera(1, urls[1]); 
        mgr.AddCamera(2, urls[2]);
        mgr.AddCamera(3, urls[3]);

        pool.SetCameras(mgr.CameraPtrs());

        pool.StartThreadPool(2);
        mgr.StartAllCams();

        // Display(Main Thread)
        cv::namedWindow("RT-DETR Mosaic", cv::WINDOW_NORMAL);

        std::vector<uint64_t> last_seq(mgr.Cameras().size(), 0);

        std::atomic<bool> running{ true };

        //  Mosaic profile
        auto& cams = mgr.Cameras();

        const int tile_w = 640;   // 每格显示大小
        const int tile_h = 360;
        const int cols = 2;       // 2列
        const int rows = static_cast<int>((cams.size() + cols - 1) / cols);

        cv::Mat mosaic(rows * tile_h, cols * tile_w, CV_8UC3, cv::Scalar(0, 0, 0));

        while (running.load()) {
            mgr.TickWatchdog();

            mosaic.setTo(cv::Scalar(0, 0, 0));

            for (size_t i = 0; i < cams.size(); ++i) {
                auto& cam = cams[i];

                std::optional<video::Frame> frame_now;
                uint64_t seq_now = 0;

                {
                    std::lock_guard<std::mutex> lk(cam->shared.frame_m);
                    if (cam->shared.latest_frame.has_value()) {
                        frame_now = cam->shared.latest_frame;
                        seq_now = cam->shared.frame_seq;
                    }
                }

                if (!frame_now.has_value() || frame_now->bgr.empty()) continue;

                Dets dets_now;
                uint64_t det_seq = 0;
                {
                    std::lock_guard<std::mutex> lk(cam->shared.det_m);
                    dets_now = cam->shared.latest_dets;
                    det_seq = cam->shared.det_seq;
                }

                video::CameraContext::FrameStats stats_snap;
                {
                    std::lock_guard<std::mutex> lk(cam->shared.stats_m);
                    stats_snap = cam->shared.stats;
                }

                std::string hud = cv::format(
                    "dec:%.1fms  inf:%.1fms  vis:%.1fms",
                    stats_snap.decode_ms,
                    stats_snap.infer_ms,
                    stats_snap.vis_ms
                );

                auto tv0 = std::chrono::steady_clock::now();


                cv::Mat vis;
                frame_now->bgr.copyTo(vis);

                DrawDetections(vis, dets_now);

                int sta_margin = 10;
                DrawHudTopRight(vis, hud, sta_margin);

                // resize 到 tile，并贴到 mosaic ROI
                cv::Mat tile;
                cv::resize(vis, tile, cv::Size(tile_w, tile_h));

                int r = static_cast<int>(i / cols);
                int c = static_cast<int>(i % cols);
                cv::Rect roi(c * tile_w, r * tile_h, tile_w, tile_h);
                tile.copyTo(mosaic(roi));

                auto tv1 = std::chrono::steady_clock::now();

                {
                    std::lock_guard<std::mutex> lk(cam->shared.stats_m);
                    cam->shared.stats.vis_ms = std::chrono::duration<double, std::milli>(tv1 - tv0).count();
                    last_vis_ms = cam->shared.stats.vis_ms;
                }

            }

            cv::imshow("RT-DETR Mosaic", mosaic);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                running.store(false);
                break;
            }
        }

        
        pool.StopThreadPool();
        mgr.SetInferencePool(nullptr);
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
