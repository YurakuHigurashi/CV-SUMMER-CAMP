/*#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/timer/timer.hpp>
*/
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <videocap.h>

#include <detector.h>


using namespace std;
using namespace cv;
using namespace cv::dnn;

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;


static cv::Mat drawRectsAndPoints(const cv::Mat &img,
	const std::vector<rectPoints> data) {
	cv::Mat outImg;
	img.convertTo(outImg, CV_8UC3);
	for (auto &d : data) {
		cv::rectangle(outImg, d.first, cv::Scalar(0, 0, 255));
		auto pts = d.second;
		for (size_t i = 0; i < pts.size(); ++i) {
			cv::circle(outImg, pts[i], 3, cv::Scalar(0, 0, 255));
		}
	}
	return outImg;
}

int videocap(string video_name,int start_frame, int frame_step,MTCNNDetector &detector,cv::Mat& mat) {

	///
	VideoCapture cap;
	cap.open(video_name);
	cap.set(CAP_PROP_POS_FRAMES, start_frame);
	if (!cap.isOpened())
	{
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}
	Mat frame;
	namedWindow("smiling", 1);
	int frame_counter = -1;
	int64 time_total = 0;
	bool paused = false;
	std::vector<Face> faces;
	for (;; )
	{
		if (paused)
		{
			char c = (char)waitKey(30);
			if (c == 'p')
				paused = !paused;
			if (c == 'q')
				break;
			continue;
		}

		cap >> frame;
		if (frame.empty()) {
			break;
		}
		frame_counter++;
		if (frame_counter < start_frame)
			continue;
		if (frame_counter % frame_step != 0)
			continue;
		int64 frame_time = getTickCount();

		faces = detector.detect(frame, 20.f, 0.709f);

		std::vector<rectPoints> data;

		// show the image with faces in it
		for (size_t i = 0; i < faces.size(); ++i) {
			std::vector<cv::Point> pts;
			for (int p = 0; p < NUM_PTS; ++p) {
				pts.push_back(
					cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
			}
			auto rect = faces[i].bbox.getRect();
			auto d = std::make_pair(rect, pts);
			data.push_back(d);
		}
		auto resultImg = drawRectsAndPoints(frame, data);
		cv::imshow("test-oc", resultImg);
		char c = (char)waitKey(2);
		if (c == 'q')
			break;
		if (c == 'p')
			paused = !paused;
	}


	return 0;
}
