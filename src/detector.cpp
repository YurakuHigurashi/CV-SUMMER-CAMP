#include "detector.h"

/*#include "detector_.h"
#include "detectedobject.h"
*/
#include "fstream"
/*
DnnDetector::DnnDetector(String pModel, String pConfig, String rModel, String rConfig, String oModel, String oConfig, int weight, int height): width(weight), height(height) {

	pnet = readNet(pModel, pConfig);
	pnet.setPreferableBackend(0);
	pnet.setPreferableTarget(0);

	rnet = readNet(rModel, rConfig);
	rnet.setPreferableBackend(0);
	rnet.setPreferableTarget(0);

	onet = readNet(oModel, oConfig);
	onet.setPreferableBackend(0);
	onet.setPreferableTarget(0);

	*/
	/*
	std::ifstream input(path_to_label);

	int numObj = 0;

	input.seekg(0, ios::beg);

	labels.resize(21);

	if (input.is_open())

		while (getline(input, labels[numObj]) && numObj < 20)

			numObj++;

	input.close();
	*/


//}




/*vector<DetectedObject> DnnDetector::detect(Mat image, const float minFaceSize,
	const float scaleFactor)

{

	vector<DetectedObject> res;

	Mat inputTensor, imageresized;

	//resize(image, imageresized, Size(300, 300));

	blobFromImage(image, inputTensor, 0.00781, Size(width, height), Scalar(127.5, 127.5, 127.5), true, false);

	pnet.setInput(inputTensor);

	Mat prob = pnet.forward();

	prob = prob.reshape(1, 1);

	prob = prob.reshape(1, prob.cols / 7);

	DetectedObject obj;

	vector<DetectedObject> result;

	for (int i = 0; i < prob.rows; i++)

	{

		if (prob.at<float>(i, 2) >= 0.2)

		{

			obj.uuid = static_cast<int>(prob.at<float>(i, 1));

			obj.score = prob.at<float>(i, 2);

			obj.Left = static_cast<int>(prob.at<float>(i, 3)*image.cols);

			obj.Bottom = static_cast<int>(prob.at<float>(i, 4)*image.rows);

			obj.Right = static_cast<int>(prob.at<float>(i, 5)*image.cols);

			obj.Top = static_cast<int>(prob.at<float>(i, 6)*image.rows);

			//obj.classname = labels[obj.uuid];

			result.push_back(obj);

		}

	}



	return result;

}
*/

MTCNNDetector::MTCNNDetector(const ProposalNetwork::Config &pConfig,
	const RefineNetwork::Config &rConfig,
	const OutputNetwork::Config &oConfig) {
	_pnet = std::unique_ptr<ProposalNetwork>(new ProposalNetwork(pConfig));
	_rnet = std::unique_ptr<RefineNetwork>(new RefineNetwork(rConfig));
	_onet = std::unique_ptr<OutputNetwork>(new OutputNetwork(oConfig));
}

std::vector<Face> MTCNNDetector::detect(const cv::Mat &img,
                                        const float minFaceSize,
                                        const float scaleFactor) {

  cv::Mat rgbImg;
  if (img.channels() == 3) {
    cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
  } else if (img.channels() == 4) {
    cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
  }
  if (rgbImg.empty()) {
    return std::vector<Face>();
  }
  rgbImg.convertTo(rgbImg, CV_32FC3);
  rgbImg = rgbImg.t();

  // Run Proposal Network to find the initial set of faces
  std::vector<Face> faces = _pnet->run(rgbImg, minFaceSize, scaleFactor);

  // Early exit if we do not have any faces
  if (faces.empty()) {
    return faces;
  }

  // Run Refine network on the output of the Proposal network
  faces = _rnet->run(rgbImg, faces);

  // Early exit if we do not have any faces
  if (faces.empty()) {
    return faces;
  }

  // Run Output network on the output of the Refine network
  faces = _onet->run(rgbImg, faces);

  for (size_t i = 0; i < faces.size(); ++i) {
    std::swap(faces[i].bbox.x1, faces[i].bbox.y1);
    std::swap(faces[i].bbox.x2, faces[i].bbox.y2);
    for (int p = 0; p < NUM_PTS; ++p) {
      std::swap(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]);
    }
  }

  return faces;
}
