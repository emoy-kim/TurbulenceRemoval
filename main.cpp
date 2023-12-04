#include "ProjectPath.h"
#include "TurbulenceRemoval.h"
#include <chrono>

void getTestset(std::vector<std::string>& testset)
{
   const std::string video_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   testset = {
      video_directory_path + "/test1.mp4",
      video_directory_path + "/test2.mp4",
      video_directory_path + "/test3.mp4",
      video_directory_path + "/test4.mp4",
      video_directory_path + "/test5.mp4"
   };
}

#define PAUSE ' '
int displayStabilizedFrame(const cv::Mat& frame, const cv::Mat& turbulence_removed, bool to_pause, bool screen_merged = false)
{
   if (screen_merged) {
      const cv::Mat result = cv::Mat::zeros(frame.rows, frame.cols * 2, CV_8UC3);
      const int col_diff_size = (frame.cols - turbulence_removed.cols) / 2;
      const int row_diff_size = (frame.rows - turbulence_removed.rows) / 2;
      frame.copyTo( result(cv::Rect(0, 0, frame.cols, frame.rows)) );
      turbulence_removed.copyTo( result(cv::Rect(frame.cols + col_diff_size, row_diff_size, turbulence_removed.cols, turbulence_removed.rows)) );
      imshow( "Input | Turbulence Removed", result );
   }
   else {
      imshow( "Input", frame );
      imshow( "Turbulence Removed", turbulence_removed );
   }

   const int key = cv::waitKey( 1 );
   if (to_pause) return PAUSE;
   return key;
}

#define ESC 27
#define TO_BE_CLOSED true
#define TO_BE_CONTINUED false
bool processKeyPressed(bool& to_pause, const int& key_pressed)
{
   switch (key_pressed) {
   case PAUSE: {
      int key;
      while ((key = cv::waitKey( 1 )) != PAUSE && key != 'f') {}
      to_pause = key == 'f';
   } break;
   case ESC:
      return TO_BE_CLOSED;
   default:
      break;
   }
   return TO_BE_CONTINUED;
}

void playVideoAndRemoveTurbulence(cv::VideoCapture& cam, TurbulenceRemoval& turbulence_remover)
{
   int key_pressed = -1;
   bool to_pause = false;
   cv::Mat frame, turbulence_removed;
   while (true) {
      cam >> frame;
      if (frame.empty()) break;

      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      turbulence_remover.removeTurbulence( turbulence_removed, frame );
      const std::chrono::duration<double> turbulence_removal_process_time = (std::chrono::system_clock::now() - start) * 1000.0;
      std::cout << "PROCESS TIME: " << turbulence_removal_process_time.count() << " ms... \r";
      
      key_pressed = displayStabilizedFrame( frame, turbulence_removed, to_pause, true );
      if (processKeyPressed( to_pause, key_pressed ) == TO_BE_CLOSED) break;
   }
}

void runTestSet(const std::vector<std::string>& testset)
{
   cv::VideoCapture cam;
   for (auto const &test_data : testset) {
      cam.open( test_data );
      if (!cam.isOpened()) continue;

      const int width = static_cast<int>(cam.get( cv::CAP_PROP_FRAME_WIDTH ));
      const int height = static_cast<int>(cam.get( cv::CAP_PROP_FRAME_HEIGHT ));
      std::cout << "*** TEST SET(" << width << " x " << height << "): " << test_data.c_str() << "***\n";

      TurbulenceRemoval turbulence_remover(10);
      playVideoAndRemoveTurbulence( cam, turbulence_remover );
      cam.release();
   }
}

int main()
{
   std::vector<std::string> testset;
   getTestset( testset );
   runTestSet( testset );
   return 0;
}