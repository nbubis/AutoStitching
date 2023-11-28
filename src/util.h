#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include "opencv2/opencv.hpp"

namespace Utils
{

	struct TreeNode
	{
		TreeNode(){};
		TreeNode(int im, int re, int le)
		{
			imgNo = im;
			refNo = re;
			level = le;
		};

		int level;       //! the level of node in the tree
		int imgNo;       //! node no.
		int refNo;       //! parent node no.
	};

	std::vector<std::string> get_filelist(std::string foldname);
	cv::Mat_<double> buildCostGraph(const cv::Mat_<int> &similarMat);
	cv::Point2d pointTransform(cv::Mat_<double> homoMat, cv::Point2d srcPt);
	void pointTransform(cv::Mat_<double> homoMat, cv::Point2d srcPt, cv::Point2d &dstPt);
	void pointTransform(cv::Mat_<double> homoMat, std::vector<cv::Point2d> &pointSet);
	double calPointDist(cv::Point2d point1, cv::Point2d point2);
	double calVecDot(cv::Point2d vec1, cv::Point2d vec2);

	void printProgress(float percent);
}
