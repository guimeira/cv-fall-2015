#ifndef _BOLD_LIB_H
#define _BOLD_LIB_H

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <unordered_set>
#include <iostream>

using namespace cv;
using namespace std;

namespace BoldLib {
	struct Keypoint {
		Point2f midpoint;
		Point2f startPoint;
		Point2f endPoint;
		float orientation;
		int scale;
		float size;
		float score;

		Keypoint() : midpoint(0,0), startPoint(0,0), endPoint(0,0), orientation(0), scale(0),
				size(0), score(0) {}

		bool operator==(const Keypoint &other) const;
		bool operator!=(const Keypoint &other) const;
	};

	struct BOLDDescriptor {
		vector<Keypoint> keypoints;
		vector<Mat> features;
		Mat *pyramid;
		vector<Point2f> *rTable;
		Point2f referencePoint;
		Scalar color;
	};

	struct BOLDTraining {
		vector<flann::Index*> indices;
		vector<BOLDDescriptor*> descriptors;
	};

	class BOLD {
		Ptr<line_descriptor::LSDDetector> lineDetector;
		vector<int> k;
		int scaleSpaceLevels;
		float scaleSpaceFactor;
		int angularBins;
		int houghThr;
		int houghNumBins;

		void computePrimitives(Keypoint kp1, Keypoint kp2, Mat gradientX, Mat gradientY, double &alpha, double &beta);
		int adjustIndex(int index, int limit);

	public:
		BOLD();
		~BOLD();
		int extract(Mat image, Scalar color, BOLDDescriptor &descriptor);
		void match(Mat sceneImg, vector<BOLDDescriptor*> &modelDescriptors);

		void setK(vector<int> k);
		void setScaleSpaceLevels(int levels);
		void setScaleSpaceFactor(float factor);
		void setNumAngularBins(int bins);
		void setHoughThr(int thr);
		void setHoughNumBins(int numBins);
	};
}

#endif
