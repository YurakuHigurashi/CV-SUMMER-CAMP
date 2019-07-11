#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
	virtual vector<DetectedObject> detect (Mat image, const float minFaceSize, const float scaleFactor) = 0;
};

class DnnDetector : public Detector
{
private:
	Net pnet,rnet,onet;
	int width, height;
public:
	DnnDetector(String pModel, String pConfig, String rModel, String rConfig, String oModel, String oConfig, int w, int h);
	vector<DetectedObject> detect(Mat image, const float minFaceSize, const float scaleFactor) override;
};
