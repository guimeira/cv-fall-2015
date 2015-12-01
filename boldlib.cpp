#include "boldlib.h"

using namespace BoldLib;

/**
 * Constructor. Instantiates the line detector.
 */
BOLD::BOLD() {
	lineDetector = line_descriptor::LSDDetector::createLSDDetector();
}

/**
 * Basically a mod operation, to make sure we don't access a position outside
 * of the array. Using a function here because % behaves in some weird ways
 * with negative numbers.
 */
int BOLD::adjustIndex(int index, int limit) {
	while(index >= limit) {
		index -= limit;
	}

	while(index < 0) {
		index += limit;
	}

	return index;
}

/**
 * Extracts the features from image.
 */
int BOLD::extract(cv::Mat image, Scalar color, BOLDDescriptor &descriptor) {
	//Detect lines on the image:
	vector<line_descriptor::KeyLine> keylines;
	lineDetector->detect(image,keylines, scaleSpaceFactor, scaleSpaceLevels);

	vector<Keypoint> keypoints[scaleSpaceLevels];
	vector<Keypoint> scaledKeypoints[scaleSpaceLevels];
	descriptor.referencePoint = Point2f(0,0);
	int totalKeypoints = 0;

	//Iterate over the lines storing the keypoints. Also average the midpoints of
	//the lines and that will be our reference point for the Generalized Hough Transform:
	for(vector<line_descriptor::KeyLine>::iterator it = keylines.begin(); it != keylines.end(); it++) {
		Keypoint scaledKp, kp;
		line_descriptor::KeyLine line = *it;

		double startX = line.startPointX, startY = line.startPointY, endX = line.endPointX, endY = line.endPointY;

		//if(line.lineLength >= 10) {
			kp.midpoint = Point2f((startX+endX)/2, (startY+endY)/2);
			kp.startPoint = Point2f(startX,startY);
			kp.endPoint = Point2f(endX,endY);
			kp.orientation = line.angle;
			kp.size = line.lineLength;
			kp.scale = line.octave;
			keypoints[kp.scale].push_back(kp);

			for(int i = 0; i < line.octave; i++) {
				startX *= 2;
				startY *= 2;
				endX *= 2;
				endY *= 2;
			}

			scaledKp.midpoint = Point2f((startX+endX)/2, (startY+endY)/2);
			scaledKp.startPoint = Point2f(startX,startY);
			scaledKp.endPoint = Point2f(endX,endY);
			scaledKp.orientation = line.angle;
			scaledKp.size = line.lineLength;
			scaledKp.scale = line.octave;
			scaledKeypoints[scaledKp.scale].push_back(scaledKp);

			descriptor.referencePoint.x += scaledKp.midpoint.x;
			descriptor.referencePoint.y += scaledKp.midpoint.y;
			totalKeypoints++;
		//}
	}

	//Let's put the reference point in the middle of the detected points:
	descriptor.referencePoint.x /= totalKeypoints;
	descriptor.referencePoint.y /= totalKeypoints;

	//Allocate R table for the Generalized Hough Transform:
	descriptor.rTable = new vector<Point2f>[houghNumBins];

	//Compute the image pyramid:
	descriptor.pyramid = new Mat[scaleSpaceLevels];
	descriptor.pyramid[0] = image;

	Mat previous = image;
	for(int i = 1; i < scaleSpaceLevels; i++) {
		Mat scaled;
		pyrDown(previous,scaled);
		descriptor.pyramid[i] = scaled;
		previous = scaled;
	}

	//Compute the gradients for the pyramid:
	vector<Mat> gradientX, gradientY;

	for(int i = 0; i < scaleSpaceLevels; i++) {
		Mat m = descriptor.pyramid[i];
		Mat gx,gy;
		Scharr(m,gx,CV_64F,1,0,1,0,BORDER_DEFAULT);
		Scharr(m,gy,CV_64F,0,1,1,0,BORDER_DEFAULT);
		gradientX.push_back(gx);
		gradientY.push_back(gy);
	}

	//Put all midpoints on a flann index, for fast nearest neighbor search:
	flann::Index *knnIndices[scaleSpaceLevels];
	Mat knnFeatures[scaleSpaceLevels];

	for(int i = 0; i < scaleSpaceLevels; i++) {
		knnFeatures[i].create(keypoints[i].size(),2,CV_32F);
		flann::LinearIndexParams params;

		int pos = 0;
		for(vector<Keypoint>::iterator it = keypoints[i].begin(); it != keypoints[i].end(); it++) {
			Keypoint kp = *it;

			knnFeatures[i].at<float>(pos,0) = kp.midpoint.x;
			knnFeatures[i].at<float>(pos,1) = kp.midpoint.y;
			pos++;
		}

		knnIndices[i] = new flann::Index(knnFeatures[i],params);
	}

	//The algorithm can use multiple K's. Let's find the biggest one, so that we can perform
	//the KNN search only once:
	int maxK = 0;
	for(vector<int>::iterator it = k.begin(); it != k.end(); it++) {
		if(*it > maxK) {
			maxK = *it;
		}
	}

	//The nearest neighbor is always the point itself, so let's find one more:
	maxK++;

	//For each scale size on the pyramid:
	for(int i = 0; i < scaleSpaceLevels; i++) {
		//Find K nearest neighbors of all points:
		Mat nearestNeighbors, distances;
		nearestNeighbors.create(knnFeatures[i].rows,maxK,CV_32SC1);
		distances.create(knnFeatures[i].rows,maxK,CV_32FC1);
		flann::SearchParams params(-1);
		knnIndices[i]->knnSearch(knnFeatures[i],nearestNeighbors,distances,maxK,params);

		//For each point:
		for(int j = 0; j < knnFeatures[i].rows; j++) {
			//Generate a 2D histogram:
			Mat hist = Mat::zeros(angularBins,angularBins,CV_32FC1);
			Keypoint kp1 = keypoints[i][j];
			Keypoint scaledKp1 = scaledKeypoints[i][j];

			//For each value of K:
			for(vector<int>::iterator it = k.begin(); it != k.end(); it++) {
				int kVal = MIN(*it,keypoints[i].size()-1);

				//For each K nearest neighbor:
				for(int k = 1; k <= kVal; k++) { //first nearest neighbor is the point itself
					Keypoint kp2 = keypoints[i][nearestNeighbors.at<int>(j,k)];

					//Compute the BOLD primitives alpha and beta:
					double alpha,beta;
					computePrimitives(kp1,kp2,gradientX[i],gradientY[i],alpha,beta);

					//A bunch of boring math to determine the 4 bins on the histogram that are
					//closest to the point (alpha,beta) and to increment them proportionaly
					//to the distance to the point:
					float scaledAlpha = (alpha/(2*CV_PI))*(angularBins-1);
					float posAlpha1, posAlpha2;

					if(scaledAlpha > floor(scaledAlpha) + 0.5) {
						posAlpha1 = floor(scaledAlpha);
						posAlpha2 = posAlpha1+1;
					} else {
						posAlpha2 = floor(scaledAlpha);
						posAlpha1 = posAlpha2-1;
					}

					float scaledBeta = (beta/(2*CV_PI))*(angularBins-1);
					float posBeta1, posBeta2;

					if(scaledBeta > floor(scaledBeta) + 0.5) {
						posBeta1 = floor(scaledBeta);
						posBeta2 = posBeta1+1;
					} else {
						posBeta2 = floor(scaledBeta);
						posBeta1 = posBeta2-1;
					}

					float dist1 = sqrt((alpha-posAlpha1)*(alpha-posAlpha1)+(beta-posBeta1)*(beta-posBeta1));
					float dist2 = sqrt((alpha-posAlpha1)*(alpha-posAlpha1)+(beta-posBeta2)*(beta-posBeta2));
					float dist3 = sqrt((alpha-posAlpha2)*(alpha-posAlpha2)+(beta-posBeta1)*(beta-posBeta1));
					float dist4 = sqrt((alpha-posAlpha2)*(alpha-posAlpha2)+(beta-posBeta2)*(beta-posBeta2));

					float sum = dist1+dist2+dist3+dist4;
					dist1 /= sum;
					dist2 /= sum;
					dist3 /= sum;
					dist4 /= sum;

					int row1, col1, row2, col2;
					row1 = adjustIndex(posAlpha1,angularBins);
					row2 = adjustIndex(posAlpha2,angularBins);
					col1 = adjustIndex(posBeta1,angularBins);
					col2 = adjustIndex(posBeta2,angularBins);

					hist.at<float>(row1,col1) += 1-dist1;
					hist.at<float>(row1,col2) += 1-dist2;
					hist.at<float>(row2,col1) += 1-dist3;
					hist.at<float>(row2,col2) += 1-dist4;
				}
			}

			//Turn the histogram into a vector, that's our feature vector:
			Mat feature = hist.reshape(0,1);
			Mat normalizedFeature;
			normalize(feature,normalizedFeature,1,0,NORM_L2);
			descriptor.features.push_back(normalizedFeature);
			descriptor.keypoints.push_back(scaledKp1);
		}
	}

	//Free some memory:
	for(int i = 0; i < scaleSpaceLevels; i++) {
		delete knnIndices[i];
	}

	//Fill the R-Table for the Generalized Hough Transform:
	for(vector<Keypoint>::iterator it = descriptor.keypoints.begin(); it != descriptor.keypoints.end(); it++) {
		Keypoint kp = *it;
		double gX = gradientX[0].at<double>(kp.midpoint.y,kp.midpoint.x);
		double gY = gradientY[0].at<double>(kp.midpoint.y,kp.midpoint.x);
		double phi = atan2(gY,gX);
		if(phi < 0) {
			phi += 2*CV_PI;
		}
		Point2f dist = descriptor.referencePoint-kp.midpoint;
		int rPos = floor((phi/(2*CV_PI))*houghNumBins);
		descriptor.rTable[rPos].push_back(dist);
	}

	descriptor.color = color;

	return 0;
}

/**
 * Make sure acos doesn't go crazy if we give it a number slightly greater than 1
 * due to imprecisions.
 */
double safeAcos(double x) {
	if(x < -1.0) x = -1.0;
	else if(x > 1.0) x = 1.0;
	return acos(x);
}

/**
 * Computes the alpha and beta values for a pair of lines.
 * Equations taken from the paper.
 */
void BOLD::computePrimitives(Keypoint kp1, Keypoint kp2, Mat gradientX, Mat gradientY, double &alpha, double &beta) {
	//Midpoints:
	Mat mi = (Mat_<double>(3,1) << kp1.midpoint.x, kp1.midpoint.y, 0);
	Mat mj = (Mat_<double>(3,1) << kp2.midpoint.x, kp2.midpoint.y, 0);

	//Segments connecting the midpoints:
	Mat tij = mj-mi;
	Mat tji = mi-mj;

	//Endpoints:
	Mat ei1 = (Mat_<double>(3,1) << kp1.startPoint.x, kp1.startPoint.y, 0);
	Mat ei2 = (Mat_<double>(3,1) << kp1.endPoint.x, kp1.endPoint.y, 0);
	Mat ej1 = (Mat_<double>(3,1) << kp2.startPoint.x, kp2.startPoint.y, 0);
	Mat ej2 = (Mat_<double>(3,1) << kp2.endPoint.x, kp2.endPoint.y, 0);

	//Gradient:
	int posX = mi.at<double>(0,0), posY = mi.at<double>(1,0);
	Mat gmi = (Mat_<double>(3,1) << gradientX.at<double>(posY,posX), gradientY.at<double>(posY,posX), 0);

	posX = mj.at<double>(0,0), posY = mj.at<double>(1,0);
	Mat gmj = (Mat_<double>(3,1) << gradientX.at<double>(posY,posX), gradientY.at<double>(posY,posX), 0);

	//Normal vector:
	Mat normal = (Mat_<double>(3,1) << 0,0,1);

	//Sign:
	Mat vecSignI = (ei2-ei1).cross(gmi);
	Mat vecSignJ = (ej2-ej1).cross(gmj);
	int signI = vecSignI.at<double>(2,0) > 0 ? 1 : -1;
	int signJ = vecSignJ.at<double>(2,0) > 0 ? 1 : -1;

	//Canonically oriented line segment:
	Mat si = signI*(ei2-ei1);
	Mat sj = signJ*(ej2-ej1);

	double alphaStar = safeAcos(si.dot(tij)/(norm(si)*norm(tij)));
	double betaStar = safeAcos(sj.dot(tji)/(norm(sj)*norm(tji)));

	if((si.cross(tij)/norm(si.cross(tij))).dot(normal) > 0) {
		alpha = alphaStar;
	} else {
		alpha = 2*CV_PI-alphaStar;
	}

	if((sj.cross(tji)/norm(sj.cross(tji))).dot(normal) > 0) {
		beta = betaStar;
	} else {
		beta = 2*CV_PI-betaStar;
	}
}

/**
 * Performs a match between each descriptor in `modelDescriptors` and `sceneImg`.
 */
void BOLD::match(Mat sceneImg, vector<BOLDDescriptor*> &modelDescriptors) {
	int numModels = modelDescriptors.size();
	BOLDDescriptor sceneDescriptor;

	//Extract descriptors from `sceneImg`:
	extract(sceneImg,Scalar(0,0,0),sceneDescriptor);

	//For each model:
	for(int i = 0; i < numModels; i++) {
		Mat &modelImg = modelDescriptors[i]->pyramid[0];
		BOLDDescriptor *modelDescriptor = modelDescriptors[i];

		//Generate flann indexes for KNN search:
		int numFeatures = sceneDescriptor.features.size();
		Mat allSceneFeatures(numFeatures,angularBins*angularBins,CV_32F);

		for(int j = 0; j < numFeatures; j++) {
			Mat featureMat = sceneDescriptor.features[j];
			featureMat.copyTo(allSceneFeatures.row(j));
		}

		int totalModelFeatures = modelDescriptor->features.size();

		Mat features(totalModelFeatures,angularBins*angularBins,CV_32F);

		int f = 0;
		for(vector<Mat>::iterator matIt = modelDescriptor->features.begin(); matIt != modelDescriptor->features.end(); matIt++) {
			Mat m = *matIt;
			m.copyTo(features.row(f));
			f++;
		}

		flann::LinearIndexParams par;
		flann::Index modelIndex(features,par);

		Mat knnIndices(numFeatures,2,CV_32S);
		Mat knnDistances(numFeatures,2,CV_32S);
		flann::SearchParams params(-1);
		modelIndex.knnSearch(allSceneFeatures,knnIndices,knnDistances,2,params);

		//Number of bins for the Hough Transform (these should be parameters of the class, but...):
		int xBins = sceneImg.cols/16;
		double xBinSize = sceneImg.cols/xBins;
		int yBins = sceneImg.rows/16;
		double yBinSize = sceneImg.rows/yBins;
		double scaleMin = 0.25;
		int scaleBins = 5;
		double rotationBinSize = CV_PI/6;
		int rotationBins = (2*CV_PI)/rotationBinSize;

		//Initialize Hough matrix:
		unordered_set<int> ****houghMatrix = new unordered_set<int>***[xBins];
		for(int j = 0; j < xBins; j++) {
			houghMatrix[j] = new unordered_set<int>**[yBins];
			for(int k = 0; k < yBins; k++) {
				houghMatrix[j][k] = new unordered_set<int>*[scaleBins];
				for(int l = 0; l < scaleBins; l++) {
					houghMatrix[j][k][l] = new unordered_set<int>[rotationBins];
				}
			}
		}
		int houghMax = 0, houghMaxX = 0, houghMaxY = 0, houghMaxScale = 0, houghMaxRotation = 0;
		Mat gradientX, gradientY;
		Scharr(sceneImg,gradientX,CV_64F,1,0,1,0,BORDER_DEFAULT);
		Scharr(sceneImg,gradientY,CV_64F,0,1,1,0,BORDER_DEFAULT);

		vector<Point2f> modelPoints, scenePoints;
		int count = 0;

		//For each point:
		for(int j = 0; j < numFeatures; j++) {
			float nearest = knnDistances.at<float>(j,0);
			float nextNearest = knnDistances.at<float>(j,1);

			//If the nearest neighbor is "closer" enough than the second nearest neighbor,
			//we consider it valid (see SIFT paper):
			if(nearest/nextNearest < 0.8) {
				Keypoint sceneKp = sceneDescriptor.keypoints[j];
				Keypoint modelKp = modelDescriptor->keypoints[knnIndices.at<int>(j,0)];
				modelPoints.push_back(modelKp.midpoint);
				scenePoints.push_back(sceneKp.midpoint);

				//Compute positions to be incremented on the Hough matrix:
				double gx = gradientX.at<double>(sceneKp.midpoint.y,sceneKp.midpoint.x);
				double gy = gradientY.at<double>(sceneKp.midpoint.y,sceneKp.midpoint.x);
				double phi = atan2(gy,gx);
				if(phi < 0) {
					phi += 2*CV_PI;
				}

				double rotation = 0;
				for(int rotationBin = 0; rotationBin < rotationBins; rotationBin++) {
					int rPos = adjustIndex(floor(((phi-rotation)/(2*CV_PI))*houghNumBins),houghNumBins);
					double scale = scaleMin;

					for(int scaleBin = 0; scaleBin < scaleBins; scaleBin++) {
						for(vector<Point2f>::iterator it = modelDescriptor->rTable[rPos].begin(); it != modelDescriptor->rTable[rPos].end(); it++) {
							Point2f possibleRefPoint = *it;
							double xp = sceneKp.midpoint.x + scale*(possibleRefPoint.x*cos(rotation)-possibleRefPoint.y*sin(rotation));
							double yp = sceneKp.midpoint.y + scale*(possibleRefPoint.x*sin(rotation)+possibleRefPoint.y*cos(rotation));
							int xBin = floor(xp/xBinSize);
							int yBin = floor(yp/yBinSize);

							if(!(xBin < 0 || xBin >= xBins || yBin < 0 || yBin >= yBins)) {
								houghMatrix[xBin][yBin][scaleBin][rotationBin].insert(count);

								if(houghMatrix[xBin][yBin][scaleBin][rotationBin].size() > houghMax) {
									houghMax = houghMatrix[xBin][yBin][scaleBin][rotationBin].size();
									houghMaxX = xBin;
									houghMaxY = yBin;
									houghMaxScale = scaleBin;
									houghMaxRotation = rotationBin;
								}
							}
						}

						scale *= 1.6;
					}

					rotation += rotationBinSize;
				}
				count++;
			}
		}

		Mat sceneColorImg;
		cvtColor(sceneImg,sceneColorImg,CV_GRAY2BGR);
		double rotScaleDiff = INFINITY;

		//Go through Hough matrix checking all the bins that have maximum value:
		for(int j = 0; j < xBins; j++) {
			for(int k = 0; k < yBins; k++) {
				double scale = scaleMin;
				for(int l = 0; l < scaleBins; l++) {
					for(int m = 0; m < rotationBins; m++) {
						if(houghMatrix[j][k][l][m].size() >= houghMax) {
							//If this bin has maximum value, we compute a transformation between the
							//points on the model and the points on the scene.
							//To choose what is the best bin, we compare the scale and orientation
							//computed by the transformation with the scale and orientation of
							//the bin in the Hough matrix.
							vector<Point2f> modelPts, scenePts, projectedPts;
							for(unordered_set<int>::iterator it = houghMatrix[j][k][l][m].begin(); it != houghMatrix[j][k][l][m].end(); it++) {
								modelPts.push_back(modelPoints[*it]);
								scenePts.push_back(scenePoints[*it]);
							}

							try {
								Mat mask;
								Mat rigid = estimateRigidTransform(modelPts,scenePts,false);
								//cout << rigid << endl;
								Scalar color(rand()%255,rand()%255,rand()%255);
								double x = j*xBinSize;
								double y = k*yBinSize;
								double rotScaleCos = cos(m*rotationBinSize)*scale;
								//circle(sceneColorImg,Point2f(x,y),5,color,5);

								if(!rigid.empty() && fabs(rotScaleCos-rigid.at<double>(0,0)) < rotScaleDiff){
									rotScaleDiff = fabs(rotScaleCos-rigid.at<double>(0,0));
									houghMaxX = j;
									houghMaxY = k;
									houghMaxScale = l;
									houghMaxRotation = m;
								}


								//cout << color << endl;
							} catch(Exception &ex) {}

						}
					}
					scale *= 1.6;
				}
			}
		}

		//cout << houghMax << " " << houghMaxX*xBinSize << " " << houghMaxY*yBinSize << " " << houghMaxScale << " " << houghMaxRotation*rotationBinSize << endl;

		//Create a rectangle around the model:
		vector<Point2f> modelBorder,sceneBorder;
		modelBorder.push_back(Point2f(0,0));
		modelBorder.push_back(Point2f(0,modelImg.rows-1));
		modelBorder.push_back(Point2f(modelImg.cols-1,modelImg.rows-1));
		modelBorder.push_back(Point2f(modelImg.cols-1,0));

		//Compute scale, position and rotation:
		double scale = scaleMin*pow(1.6,houghMaxScale);
		double rotation = rotationBinSize*houghMaxRotation;
		double x = xBinSize*houghMaxX;
		double y = yBinSize*houghMaxY;

cout << "SCALE " << scale << " ROTATION " << rotation << " x " << x << " y " << y << endl;
cout << "RefPoint " << modelDescriptor->referencePoint << endl;

		//Translate, scale and rotate the points of the model to the scene:
		for(vector<Point2f>::iterator it = modelBorder.begin(); it != modelBorder.end(); it++) {
			Point2f pt = *it;

			//Put the reference point in the origin:
			Point2f translatedPt = pt - modelDescriptor->referencePoint;

			//Apply scaling:
			Point2f scaledPt = translatedPt*scale;

			//Rotate around the origin:
			Point2f rotatedPt;
			rotatedPt.x = scaledPt.x*cos(rotation)-scaledPt.y*sin(rotation);
			rotatedPt.y = scaledPt.x*sin(rotation)+scaledPt.y*cos(rotation);

			//Apply translation:
			pt.x = rotatedPt.x+x;
			pt.y = rotatedPt.y+y;

			sceneBorder.push_back(pt);
		}

		//Draw a rectangle:
		line(sceneColorImg,sceneBorder[0],sceneBorder[1],modelDescriptor->color,2);
		line(sceneColorImg,sceneBorder[1],sceneBorder[2],modelDescriptor->color,2);
		line(sceneColorImg,sceneBorder[2],sceneBorder[3],modelDescriptor->color,2);
		line(sceneColorImg,sceneBorder[3],sceneBorder[0],modelDescriptor->color,2);

		//circle(sceneColorImg,Point2f(houghMaxX*xBinSize,houghMaxY*yBinSize),5,Scalar(255,0,0),5);
		/*perspectiveTransform(modelBorder,sceneBorder,homography);
		line(sceneColorImg,sceneBorder[0],sceneBorder[1],Scalar(0,255,0),2);
		line(sceneColorImg,sceneBorder[1],sceneBorder[2],Scalar(0,255,0),2);
		line(sceneColorImg,sceneBorder[2],sceneBorder[3],Scalar(0,255,0),2);
		line(sceneColorImg,sceneBorder[3],sceneBorder[0],Scalar(0,255,0),2);*/

		//Save image:
		imshow("Detected model",sceneColorImg);
		stringstream ss;
		ss << "model_" << i << ".png";
		imwrite(ss.str(),sceneColorImg);

		//Draw matches:
		vector<KeyPoint> matchesModel, matchesScene;
		vector<DMatch> matches;
		int j = 0;
		for(unordered_set<int>::iterator it = houghMatrix[houghMaxX][houghMaxY][houghMaxScale][houghMaxRotation].begin(); it !=  houghMatrix[houghMaxX][houghMaxY][houghMaxScale][houghMaxRotation].end(); it++) {
			Point2f m = modelPoints[*it];
			Point2f s = scenePoints[*it];

			matchesModel.push_back(KeyPoint(m,1,0,0,0));
			matchesScene.push_back(KeyPoint(s,1,0,0,0));
			matches.push_back(DMatch(j,j,0));
			j++;
		}

		Mat matchesImg;
		drawMatches(modelImg,matchesModel,sceneImg,matchesScene,matches,matchesImg);
		imshow("Matches",matchesImg);

		while((waitKey(0)&0xff) != 'c');
	}
}

void BOLD::setK(vector<int> k) {
	this->k = k;
}

void BOLD::setScaleSpaceLevels(int levels) {
	scaleSpaceLevels = levels;
}

void BOLD::setScaleSpaceFactor(float factor) {
	scaleSpaceFactor = factor;
}

void BOLD::setNumAngularBins(int bins) {
	angularBins = bins;
}

void BOLD::setHoughThr(int thr) {
	houghThr = thr;
}

void BOLD::setHoughNumBins(int numBins) {
	houghNumBins = numBins;
}

void loadTraining(const char *filename, Mat &features, vector<Keypoint> &keypoints) {
	FILE *in = fopen(filename,"r");
	int numDesc, descLen;
	fscanf(in,"%d %d\n",&numDesc,&descLen);
	features.create(numDesc,descLen,CV_32F);

	for(int i = 0; i < numDesc; i++) {
		for(int j = 0; j < descLen; j++) {
			float d;
			fscanf(in,"%f ",&d);
			features.at<float>(i,j) = d;
		}
	}

	int numKp;
	fscanf(in,"%d\n",&numKp);

	for(int i = 0; i < numKp; i++) {
		float x,y,orientation,scale,size;
		fscanf(in,"%f %f %f %f %f\n",&x,&y,&orientation,&scale,&size);
		Keypoint kp;
		kp.midpoint = Point2f(x,y);
		kp.orientation = orientation;
		kp.scale = 0;
		kp.size = size;
		keypoints.push_back(kp);
	}
	fclose(in);
}

void showKeypoints(Mat img, vector<Keypoint> &kps) {
	vector<KeyPoint> kpVec;
	for(vector<Keypoint>::iterator it = kps.begin(); it != kps.end(); it++) {
		arrowedLine(img,(*it).startPoint,(*it).endPoint,Scalar(0,255,0));
		//Keypoint kp = *it;
		//kpVec.push_back(KeyPoint(kp.midpoint,0,0,0,0,0));
	}
	/*Mat kpImg;
	drawKeypoints(img,kpVec,kpImg);
	imshow("keypoints",kpImg);*/
	while((waitKey(0)&0xff) != 'c');
}

BOLD::~BOLD() {}

int main() {
	Mat s0 = imread("scene6.jpg",0);
	Mat scene0;
	GaussianBlur(s0,scene0,Size(3,3),0,0);

	Mat m0 = imread("model0.jpg",0);
	Mat model0;
	GaussianBlur(m0,model0,Size(3,3),0,0);

	Mat m1 = imread("model1.jpg",0);
	Mat model1;
	GaussianBlur(m1,model1,Size(3,3),0,0);

	Mat m2 = imread("model2.jpg",0);
	Mat model2;
	GaussianBlur(m2,model2,Size(3,3),0,0);

	BOLD bold;
	vector<int> k;
	k.push_back(5);
	k.push_back(10);
	k.push_back(15);
	k.push_back(20);
	k.push_back(25);
	bold.setK(k);
	bold.setScaleSpaceLevels(3);
	bold.setScaleSpaceFactor(2.0f);
	bold.setNumAngularBins(12);
	bold.setHoughNumBins(12);
	BOLDDescriptor descriptor0;
	bold.extract(model0,Scalar(255,0,0),descriptor0);
	BOLDDescriptor descriptor1;
	bold.extract(model1,Scalar(0,255,0),descriptor1);
	BOLDDescriptor descriptor2;
	bold.extract(model2,Scalar(0,0,255),descriptor2);

	vector<BOLDDescriptor*> descriptors;
	descriptors.push_back(&descriptor0);
	descriptors.push_back(&descriptor1);
	descriptors.push_back(&descriptor2);

	bold.match(scene0,descriptors);
}
