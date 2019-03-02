#include "TurbulenceRemoval.h"
#include <chrono>

using namespace chrono;

void getTestset(vector<string>& testset)
{
   testset = {
      "VideoSamples/test1.avi",
      "VideoSamples/test2.avi",
      "VideoSamples/test3.avi",
      "VideoSamples/test4.avi",
      "VideoSamples/test5.avi"
   };
}

#define PAUSE ' '
int displayStabilizedFrame(const Mat& frame, const Mat& turbulence_removed, const bool& to_pause, const bool& screen_merged = false)
{
   if (screen_merged) {
      const Mat result = Mat::zeros(frame.rows, frame.cols * 2, CV_8UC3);
      const int col_diff_size = (frame.cols - turbulence_removed.cols) / 2;
      const int row_diff_size = (frame.rows - turbulence_removed.rows) / 2;
      frame.copyTo( result(Rect(0, 0, frame.cols, frame.rows)) );
      turbulence_removed.copyTo( result(Rect(frame.cols + col_diff_size, row_diff_size, turbulence_removed.cols, turbulence_removed.rows)) );
      imshow( "Input | Turbulence Removed", result );
   }
   else {
      imshow( "Input", frame );
      imshow( "Turbulence Removed", turbulence_removed );
   }

   const int key = waitKey( 1 );
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
      while ((key = waitKey( 1 )) != PAUSE && key != 'f') {}
      to_pause = key == 'f';
   } break;
   case ESC:
      return TO_BE_CLOSED;
   default:
      break;
   }
   return TO_BE_CONTINUED;
}

void playVideoAndRemoveTurbulence(VideoCapture& cam, TurbulenceRemoval& turbulence_remover)
{
   int key_pressed = -1;
   bool to_pause = false;
   Mat frame, turbulence_removed;
   while (true) {
      cam >> frame;
      if (frame.empty()) break;
      
      time_point<system_clock> start = system_clock::now();
      turbulence_remover.removeTurbulence( turbulence_removed, frame );
      const duration<double> turbulence_removal_process_time = (system_clock::now() - start) * 1000.0;
      cout << "PROCESS TIME: " << turbulence_removal_process_time.count() << " ms... \r";
      
      key_pressed = displayStabilizedFrame( frame, turbulence_removed, to_pause, false );
      if (processKeyPressed( to_pause, key_pressed ) == TO_BE_CLOSED) break;
   }
}

void runTestSet(const vector<string>& testset)
{
   VideoCapture cam;
   for (auto const &test_data : testset) {
      cam.open( test_data );
      if (!cam.isOpened()) continue;

      const int width = static_cast<int>(cam.get( CV_CAP_PROP_FRAME_WIDTH ));
      const int height = static_cast<int>(cam.get( CV_CAP_PROP_FRAME_HEIGHT ));
      cout << "*** TEST SET(" << width << " x " << height << "): " << test_data.c_str() << "***" << endl;

      TurbulenceRemoval turbulence_remover(10);
      playVideoAndRemoveTurbulence( cam, turbulence_remover );
      cam.release();
   }
}

int main()
{
   vector<string> testset;
   getTestset( testset );
   runTestSet( testset );
   return 0;
}