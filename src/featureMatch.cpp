#include <limits>
#include <execution>
#include "featureMatch.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

const int PointMatcher::minimumMatches = 10;

PointMatcher::PointMatcher(std::vector<std::string> &imgPathList, int limitImageMatchNum, float resizedFactorForFeatures) :
	_resizedFactorForFeatures(resizedFactorForFeatures), _imgNum(imgPathList.size()), _imgPathList(imgPathList)
{

	featureExtractor();

	std::vector<int> imageRange(_imgNum);
	std::iota(imageRange.begin(), imageRange.end(), 0);
	std::cout << "Matching features ..." << std::endl;

	if (limitImageMatchNum <= 0)
	{
		limitImageMatchNum = _imgNum;
	}
	float percent = 0.0;

	_matches.resize(_imgNum);
	std::for_each(_matches.begin(), _matches.end(), [&](auto & match_i) {match_i.resize(_imgNum);});

	std::for_each(std::execution::par_unseq, imageRange.begin(), imageRange.end(), [&](auto & i)
	{
		for (int j = i + 1; j < std::min(i + limitImageMatchNum + 1, _imgNum); j++) { 
			featureMatcher(i, j);
		}
		
		percent += 1.0f / (float)_imgNum;
		Utils::printProgress(percent); 
	});

	std::cout << std::endl;
};

PointMatcher::PointMatcher(PointMatcher existing, int imgIndex1, int imgIndex2)
{
	_imgNum = imgIndex2 - imgIndex1;
	_resizedFactorForFeatures = existing._resizedFactorForFeatures;
	_imgSizeList.clear();
	_imgPathList.clear();
	_keyPts.clear();
	_descriptors.clear();
	_matches.clear();
	_matches.resize(_imgNum);
	for (int i = imgIndex1; i < imgIndex2; i++) {
		_imgPathList.push_back(existing._imgPathList[i]);
		_imgSizeList.push_back(existing._imgSizeList[i]);
		_keyPts.push_back(existing._keyPts[i]);
		_descriptors.push_back(existing._descriptors[i]);
		_matches[i - imgIndex1].resize(_imgNum);
		for (int j = imgIndex1; j < imgIndex2; j++) {
			_matches[i - imgIndex1][j - imgIndex1] = existing._matches[i][j];
		}
	}
}

void PointMatcher::featureExtractor()
{
	// descriptor_extractor->descriptorSize() << " Type: " << descriptor_extractor->descriptorType()
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);
	_keyPts.resize(_imgNum);
	_imgSizeList.resize(_imgNum);
	_descriptors.resize(_imgNum);

	std::cout << "Reading images and extracting features ..." << std::endl;

	float percent = 0.0f;
	std::for_each(std::execution::par_unseq, _imgPathList.begin(), _imgPathList.end(), [&](std::string & imgName) {

		cv::Mat image = cv::imread(imgName);
		int i = &imgName - &_imgPathList[0];

		_imgSizeList[i] = image.size();

		if (_resizedFactorForFeatures > 0) {
			cv::resize(image, image, cv::Size(image.cols / _resizedFactorForFeatures , int(image.rows / _resizedFactorForFeatures)));
		}

		std::vector<cv::KeyPoint> keyPts;
		cv::Mat descriptors;

		detector->detectAndCompute(image, cv::noArray(), keyPts, descriptors);

		// upscale features back to original image dimensions

		for (auto & keyPt : keyPts) {
			keyPt.pt = keyPt.pt * _resizedFactorForFeatures;
		}

		_keyPts[i] = keyPts;
		_descriptors[i] = descriptors;
		percent += 1.0f / (float)_imgNum;
		Utils::printProgress(percent); 
	});
	std::cout << std::endl;
}

void PointMatcher::getfeatures(int imgIndex, std::vector<cv::Point2d> &imageKeyPts, cv::Mat &imageDescriptors, int ratio)
{
	imageKeyPts.clear();
	imageDescriptors.release();
	imageDescriptors = cv::Mat(_keyPts[imgIndex].size() / ratio, 64, CV_32FC1, cv::Scalar(0.0f));
	for (int i = 0; i < _keyPts[imgIndex].size(); i++)
	{
		if (i % ratio == 0)
		{
			imageKeyPts.push_back(_keyPts[imgIndex][i].pt);
			_descriptors[imgIndex].row(i).copyTo(imageDescriptors.row(i / ratio));
		}
	}
}

bool PointMatcher::featureMatcher(int imgIndex1, int imgIndex2)
{
	std::vector<cv::Point2d> keyPts1, keyPts2;
	cv::Mat descriptors1, descriptors2;

	getfeatures(imgIndex1, keyPts1, descriptors1);
	getfeatures(imgIndex2, keyPts2, descriptors2);

	if (keyPts1.size() < minimumMatches || keyPts2.size() < minimumMatches)
	{
		return false;
	}

	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> knnmatches;

	matcher.knnMatch(descriptors1, descriptors2, knnmatches, 5);

	double minimalDistance = std::numeric_limits<double>::max();
	for (int i = 0; i < knnmatches.size(); i++)
	{
		double dist = knnmatches[i][1].distance;
		if (dist < minimalDistance)
		{
			minimalDistance = dist;
		}
	}

	double fitedThreshold = minimalDistance * 5;
	int keypointsize = knnmatches.size();
	std::vector<cv::DMatch> m_Matches;

	for (int i = 0; i < keypointsize; i++)
	{
		const cv::DMatch nearDist1 = knnmatches[i][0];
		const cv::DMatch nearDist2 = knnmatches[i][1];
		double distanceRatio = nearDist1.distance / nearDist2.distance;
		if (nearDist1.distance < fitedThreshold && distanceRatio < 0.8)
		{
			m_Matches.push_back(nearDist1);
		}
	}

	std::vector<cv::Point2d> iniPts1, iniPts2;
	for (int i = 0; i < m_Matches.size(); i++) // get initial match pairs
	{
		int queryIndex = m_Matches[i].queryIdx;
		int trainIndex = m_Matches[i].trainIdx;
		cv::Point2d tempPt1 = keyPts1[queryIndex];
		cv::Point2d tempPt2 = keyPts2[trainIndex];
		iniPts1.push_back(tempPt1);
		iniPts2.push_back(tempPt2);
	}
	if (iniPts1.size() < minimumMatches)
	{
		return false;
	}

	std::vector<uchar> status;
	cv::Mat Fmatrix = cv::findFundamentalMat(iniPts1, iniPts2, cv::RANSAC, 1.5, 0.99, status);
	if (Fmatrix.empty())
	{
		return false;
	}

	std::vector<cv::Point2d> pointSet1, pointSet2;

	for (int i = 0; i < status.size(); i++)
	{
		if (status[i] == 1)
		{
			pointSet1.push_back(iniPts1[i]);
			pointSet2.push_back(iniPts2[i]);
		}
	}
	if (pointSet1.size() < minimumMatches)
	{
		return false;
	}

	cv::Mat homography = findHomography(pointSet1, pointSet2, cv::RANSAC, 2.5);
	if (homography.empty())
	{
		return false;
	}
	std::vector<cv::Point2d> goodPts1, goodPts2, warpedPoints;

	cv::perspectiveTransform(pointSet1, warpedPoints, homography);

	for (int i = 0; i < pointSet1.size(); i++)
	{
		if (cv::norm(warpedPoints[i] - pointSet2[i]) < 3.0)
		{
			goodPts1.push_back(pointSet1[i]);
			goodPts2.push_back(pointSet2[i]);
		}
	}

	pointSet1 = goodPts1;
	pointSet2 = goodPts2;

	if (pointSet1.size() < minimumMatches)
	{
		return false;
	}

	saveMatchPts(imgIndex1, imgIndex2, pointSet1, pointSet2);

	return true;
}

void PointMatcher::saveMatchPts(int imgIndex1, int imgIndex2, std::vector<cv::Point2d> &pointSet1, std::vector<cv::Point2d> &pointSet2)
{
	std::vector<std::pair<cv::Point2d, cv::Point2d>> matchPairs;

	if (imgIndex2 > imgIndex1)
	{
		for (int i = 0; i < pointSet1.size(); i++)
		{
			matchPairs.push_back({pointSet1[i], pointSet2[i]});
		}
	}
	else
	{
		for (int i = 0; i < pointSet1.size(); i++)
		{
			matchPairs.push_back({pointSet2[i], pointSet1[i]});
		}
	}

	_matches[imgIndex1][imgIndex2] = matchPairs;
	_matches[imgIndex2][imgIndex1] = matchPairs;
}

bool PointMatcher::getMatchPoints(int imgIndex1, int imgIndex2, std::vector<cv::Point2d> &pointSet1, std::vector<cv::Point2d> &pointSet2)
{

	if (imgIndex2 > imgIndex1)
	{
		for (int i = 0; i < _matches[imgIndex1][imgIndex2].size(); i++)
		{
			pointSet1.push_back(_matches[imgIndex1][imgIndex2][i].first);
			pointSet2.push_back(_matches[imgIndex1][imgIndex2][i].second);
		}
	}
	else
	{
		for (int i = 0; i < _matches[imgIndex2][imgIndex1].size(); i++)
		{
			pointSet1.push_back(_matches[imgIndex2][imgIndex1][i].second);
			pointSet2.push_back(_matches[imgIndex2][imgIndex1][i].first);
		}
	}
	return true;
}

bool PointMatcher::tentativeMatcher(int imgIndex1, int imgIndex2)
{
	// std::vector<cv::Point2d> keyPts1, keyPts2;
	// cv::Mat descriptors1, descriptors2;
	// getfeatures(imgIndex1, keyPts1, descriptors1, 3);
	// getfeatures(imgIndex2, keyPts2, descriptors2, 3);

	// // Matching descriptor vectors using FLANN matcher
	// vector<DMatch> m_Matches;
	// FlannBasedMatcher matcher;
	// vector<vector<DMatch>> knnmatches;
	// int num1 = keyPts1.size(), num2 = keyPts2.size();
	// int kn = min(min(num1, num2), 5);
	// try
	// {
	// 	matcher.knnMatch(descriptors1, descriptors2, knnmatches, kn);
	// }
	// catch (std::exception const &e)
	// {
	// 	std::cout << "Exception: " << e.what() << std::endl;
	// }
	// int i, j;
	// double minimaDsit = 99999;
	// for (i = 0; i < knnmatches.size(); i++)
	// {
	// 	double dist = knnmatches[i][0].distance;
	// 	if (dist < minimaDsit)
	// 	{
	// 		minimaDsit = dist;
	// 	}
	// }
	// double fitedThreshold = minimaDsit * 5;
	// int keypointsize = knnmatches.size();
	// for (i = 0; i < keypointsize; i++)
	// {
	// 	const DMatch nearDist1 = knnmatches[i][0];
	// 	const DMatch nearDist2 = knnmatches[i][1];
	// 	double distanceRatio = nearDist1.distance / nearDist2.distance;
	// 	if (nearDist1.distance < fitedThreshold && distanceRatio < 0.7)
	// 	{
	// 		m_Matches.push_back(nearDist1);
	// 	}
	// }
	// vector<Point2d> iniPts1, iniPts2;
	// for (i = 0; i < m_Matches.size(); i++) // get initial match pairs
	// {
	// 	int queryIndex = m_Matches[i].queryIdx;
	// 	int trainIndex = m_Matches[i].trainIdx;
	// 	Point2d tempPt1 = keyPts1[queryIndex];
	// 	Point2d tempPt2 = keyPts2[trainIndex];
	// 	iniPts1.push_back(tempPt1);
	// 	iniPts2.push_back(tempPt2);
	// }
	// if (iniPts1.size() < 15)
	// {
	// 	return false;
	// }
	// Mat_<double> homography = findHomography(iniPts1, iniPts2, RANSAC, 5.0); // initial solution : from image2 to image1

	// if (homography.empty()) {
	// 	return false;
	// }

	// vector<Point2d> goodPts1, goodPts2, warpedPoints;

	// cv::perspectiveTransform(iniPts1, warpedPoints, homography);

	// for (int i = 0; i < iniPts1.size(); i++)
	// {
	// 	if (cv::norm(warpedPoints[i] - iniPts2[i]) < 5.0)
	// 	{
	// 		goodPts1.push_back(iniPts1[i]);
	// 		goodPts2.push_back(iniPts2[i]);
	// 	}
	// }

	// if (goodPts1.size() < 5)
	// {
	// 	return false;
	// }

	return true;
}
