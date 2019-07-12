/*#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/timer/timer.hpp>
*/
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <detector_.h>
#include <detector.h>


//namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv::dnn;
//using namespace cv::tbm;

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

const char* cmdOptions =
{ "{video_name       | | video name                       }"
"{start_frame      |0| Start frame                      }"
"{frame_step       |1| Frame step                       }"
"{Proposal_model   | | Path to detector's Caffe model   }"
"{Proposal_weights | | Path to detector's Caffe weights }"
"{Refine_model   | | Path to Refine's Caffe model   }"
"{Refine_weights | | Path to Refine's Caffe weights }"
"{Output_model   | | Path to Output's Caffe model   }"
"{Output_weights | | Path to Output's Caffe weights }"
};
string smile = "..\..\CV-SUMMER-CAMP\data\smile.png";
/*
-start_frame=0
-frame_step=2
-Proposal_model="..\..\CV-SUMMER-CAMP\models\object_detection\common\mtcnn\p\caffe\mtcnn-p.caffemodel"
-Proposal_weights="..\..\CV-SUMMER-CAMP\models\object_detection\common\mtcnn\p\caffe\mtcnn-p.prototxt"
-Refine_model="..\..\CV-SUMMER-CAMP\models\object_detection\common\mtcnn\r\caffe\mtcnn-r.caffemodel"
-Refine_weights="..\..\CV-SUMMER-CAMP\models\object_detection\common\mtcnn\r\caffe\mtcnn-r.prototxt"
-Output_model="..\..\CV-SUMMER-CAMP\models\object_detection\common\mtcnn\o\caffe\mtcnn-o.caffemodel"
-Output_weights="..\..\CV-SUMMER-CAMP\models\object_detection\common\mtcnn\o\caffe\mtcnn-o.prototxt"
-video_name="..\..\CV-SUMMER-CAMP\data\faces1.jpg"

*/


static cv::Mat drawRectsAndPoints(const cv::Mat &img,
                                  const std::vector<rectPoints> data) {
  cv::Mat outImg;
  img.convertTo(outImg, CV_8UC3);
  cv::Mat tempImg;
  tempImg=imread(smile);
  for (auto &d : data) {
    cv::rectangle(outImg, d.first, cv::Scalar(0, 0, 255));
    auto pts = d.second;
    for (size_t i = 0; i < pts.size(); ++i) {
      cv::circle(outImg, pts[i], 3, cv::Scalar(0, 0, 255));
    }
  }
  return outImg;
}

int main(int argc, char **argv) {

	/*if (argc < 3) {
	std::cerr << "Usage " << argv[0] << ": "
				<< "<model-dir> "
				<< " "
				<< "<test-image>\n";
	return -1;
	}*/
	CommandLineParser parser(argc, argv, cmdOptions);

	//cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();

	String video_name = parser.get<String>("video_name");
	int start_frame = parser.get<int>("start_frame");
	int frame_step = parser.get<int>("frame_step");

	//fs::path modelDir = fs::path(argv[1]);

	ProposalNetwork::Config pConfig;
	pConfig.caffeModel = parser.get<String>("Proposal_model");
	pConfig.protoText = parser.get<String>("Proposal_weights");
	pConfig.threshold = 0.6f;

	RefineNetwork::Config rConfig;
	rConfig.caffeModel = parser.get<String>("Refine_model");
	rConfig.protoText = parser.get<String>("Refine_weights");
	rConfig.threshold = 0.7f;

	OutputNetwork::Config oConfig;
	oConfig.caffeModel = parser.get<String>("Output_model");
	oConfig.protoText = parser.get<String>("Output_weights");
	oConfig.threshold = 0.7f;
  
	MTCNNDetector detector(pConfig, rConfig, oConfig);
	cv::Mat img = cv::imread(parser.get<string>("video_name"));
	
	std::vector<Face> faces;
	//std::vector<Faces> faces;
	
	//MTCNNDetector detector(pConfig.caffeModel,pConfig.protoText, rConfig.caffeModel, rConfig.protoText, oConfig.caffeModel, oConfig.protoText, 300,300);
	
	//boost::timer::auto_cpu_timer t(3, "%w seconds\n");
	faces = detector.detect(img, 20.f, 0.709f);
	

	std::cout << "Number of faces found in the supplied image - " << faces.size()
			<< std::endl;

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

	auto resultImg = drawRectsAndPoints(img, data);
	cv::imshow("test-oc", resultImg);
	cv::waitKey(0);
	
	return 0;
}
