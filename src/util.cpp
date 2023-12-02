#include "util.h"
#include <filesystem>
#include <cmath>

cv::Mat_<double> Utils::buildCostGraph(const cv::Mat_<int> &similarMat)
{
	int nodeNum = similarMat.rows;
	//! considering the precise and robustness, we take logarithm as the weight function
	cv::Mat_<double> costGraph = cv::Mat(nodeNum, nodeNum, CV_64FC1, cv::Scalar(-1));
	for (int i = 0; i < nodeNum-1; i ++)
	{
		for (int j = i+1; j < nodeNum; j ++)
		{
			int num = similarMat(i,j);
			if (num == 0)
			{
				continue;
			}
			double cost = 6/log(num+50.0);
			costGraph(i,j) = cost;
			costGraph(j,i) = cost;
		}
	}
	return costGraph;
}


cv::Point2d Utils::pointTransform(cv::Mat_<double> homoMat, cv::Point2d srcPt)
{
	cv::Mat_<double> srcX = (cv::Mat_<double>(3,1)<< srcPt.x, srcPt.y, 1);
	cv::Mat_<double> dstX = homoMat * srcX;
	cv::Point2d dstPt = cv::Point2d(dstX(0)/dstX(2), dstX(1)/dstX(2));
	return dstPt;
}


void Utils::pointTransform(cv::Mat_<double> homoMat, cv::Point2d srcPt, cv::Point2d &dstPt)
{
	cv::Mat_<double> srcX = (cv::Mat_<double>(3,1)<< srcPt.x, srcPt.y, 1);
	cv::Mat_<double> dstX = homoMat * srcX;
	dstPt = cv::Point2d(dstX(0)/dstX(2), dstX(1)/dstX(2));
}


void Utils::pointTransform(cv::Mat_<double> homoMat, std::vector<cv::Point2d> &pointSet)
{
	for (int i = 0; i < pointSet.size(); i ++)
	{
		cv::Mat_<double> srcX = (cv::Mat_<double>(3,1)<< pointSet[i].x, pointSet[i].y, 1);
		cv::Mat_<double> dstX = homoMat * srcX;
		cv::Point2d dstPt = cv::Point2d(dstX(0)/dstX(2), dstX(1)/dstX(2));
		pointSet[i] = dstPt;
	}
}


double Utils::calPointDist(cv::Point2d point1, cv::Point2d point2)
{
	return sqrt((point1.x-point2.x)*(point1.x-point2.x) + (point1.y-point2.y)*(point1.y-point2.y));
}


double Utils::calVecDot(cv::Point2d vec1, cv::Point2d vec2)
{
	return vec1.x*vec2.x+vec1.y*vec2.y;
}

void Utils::printProgress(float percent) 
{
	int barLength = 30;
	int progress = std::min((int)std::ceil(barLength * percent), barLength);
	std::mutex coutMutex;
	coutMutex.lock();
	std::cout << "\r[" << std::string(progress, '=') << std::string(barLength - progress, ' ') << "] " << std::fixed <<
		std::setprecision(1) << std::setw(6) << 100.0f * percent << "%  " << std::flush;
	coutMutex.unlock();
}
