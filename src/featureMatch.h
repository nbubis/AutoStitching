#pragma once
#include "util.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>

class PointMatcher
{

public:
  	static const int minimumMatches;
	PointMatcher(std::vector<std::string> & imgNameList, int limitImageMatchNum = 0, float resizedFactorForFeatures = 1.0f);
	void getfeatures(int imgIndex, std::vector<cv::Point2d> &imageKeyPts, cv::Mat &imageDescriptors, int ratio = 1);
	bool getMatchPoints(int imgIndex1, int imgIndex2, std::vector<cv::Point2d> &pointSet1, std::vector<cv::Point2d> &pointSet2);
	PointMatcher getSubset(int imgIndex1, int imgIndex2);
private:

	void featureExtractor();
	bool tentativeMatcher(int imgIndex1, int imgIndex2);
	bool featureMatcher(int imgIndex1, int imgIndex2);
	void saveMatchPts(int imgIndex1, int imgIndex2, std::vector<cv::Point2d> &pointSet1, std::vector<cv::Point2d> &pointSet2);
	void pointConvert(cv::Mat_<double> homoMat, cv::Point2d src, cv::Point2d &dst);

private:

	int _imgNum;
	float _resizedFactorForFeatures;
	std::vector<cv::Size> _imgSizeList;
	std::vector<std::string> _imgPathList;
	std::vector<std::vector<cv::KeyPoint>> _keyPts;
	std::vector<cv::Mat> _descriptors;
	std::vector<std::vector<std::vector<std::pair<cv::Point2d, cv::Point2d>>>> _matches;

public:
	const auto & imgNum() const {return _imgNum;};
	const auto & imgSizeList() const {return _imgSizeList;};
	const auto & imgPathList() const {return _imgPathList;};
	const auto & keyPts() const {return _keyPts;};
	const auto & descriptors() const {return _descriptors;};
	const auto & matches() const {return _matches;};
	bool matches(int i, int j) {return _matches[i][j].size() > 0;};
};
