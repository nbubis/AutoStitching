#include "alignment.h"
#include <omp.h>
#include <numeric>
#include <execution>
#include <opencv2/stitching/detail/blenders.hpp>
#include <exiv2/exiv2.hpp>

float ImageAligner::_penaltyCoeffLM = 0.1;

ImageAligner::ImageAligner(PointMatcher & pointMatcher, bool forceSimilarity, std::string outputDir) : 
	_forceSimilarity(forceSimilarity), _matcher(pointMatcher)
{
	_imgNum = pointMatcher.imgNum();
	_imgSizeList = _matcher.imgSizeList();
	_filePathList = _matcher.imgPathList();
	_refImgNo = 0;
	_outputDir = outputDir;
	if (forceSimilarity) {
		_penaltyCoeffBA = 0.10;
	} else {
		_penaltyCoeffBA = 0.01;
	}
};

void ImageAligner::sortImageOrder(int referNo, bool shallLoad, bool isInorder)
{
	TopoFinder topoBar(_matcher, _outputDir);

	_similarityMat = topoBar.findTopology(shallLoad, isInorder);
	Mat_<double> costGraph = Utils::buildCostGraph(_similarityMat);
	if (referNo == -1)
	{
		_visitOrder = Graph::FloydForPath(costGraph);
		if (_visitOrder.size() < _imgNum) {
			throw std::logic_error("Wrong _visitOrder size");
		}
		_refImgNo = _visitOrder[0].imgNo;
	}
	else
	{
		_visitOrder = Graph::DijkstraForPath(costGraph, referNo);
		_refImgNo = _visitOrder[0].imgNo;
	}

	divideImageGroups();
}


void ImageAligner::divideImageGroups()
{
	//! the first group imgIndex
	_groupCusorList.push_back(0);
	//! the following group cursors
	int offset = OPT_GROUP_NUM;
	for (int imgIndex = offset; imgIndex < _imgNum; imgIndex += offset)
	{
		if (_imgNum - imgIndex < offset/2)
		{
			imgIndex = _imgNum-1;
		}
		_groupCusorList.push_back(imgIndex);
	}
	int groupNum = _groupCusorList.size();
	if (_groupCusorList[groupNum-1] < _imgNum-1)
	{
		_groupCusorList.push_back(_imgNum-1);
	}
}


//-------------------------------------------------alignment
void ImageAligner::imageStitcherbyGroup(int referNo)
{

	bool shallLoad = false, isInOrder = true;     //! ### set this for new data
	sortImageOrder(referNo, shallLoad, isInOrder);
	fillImageMatchNet();

	Mat_<double> identMatrix = Mat::eye(3,3,CV_64FC1);     //cvtMatrix of reference image
	_alignModelList.push_back(identMatrix);
	_initModelList.push_back(identMatrix);
	Quadra bar;
	bar.imgSize = _imgSizeList[_visitOrder[0].imgNo];
	bar.centroid = Point2d(bar.imgSize.width/2, bar.imgSize.height/2);
	_projCoordSet.push_back(bar);
	for (int i = 1; i < _groupCusorList.size(); i++)
	{
		int sIndex = _groupCusorList[i-1]+1;
		int eIndex = _groupCusorList[i];


		if (_forceSimilarity) 
		{
			solveGroupModelsS(sIndex, eIndex);
		} else 
		{
			solveGroupModels(sIndex, eIndex);
		}

		if (i == _groupCusorList.size()-1)
		{
			bundleAdjustinga(1, eIndex);
			sIndex = 0;
			RefineAligningModels(sIndex, eIndex);
		}
	}
}


void ImageAligner::imageStitcherbySolos(int referNo)
{
	//! =============== extract features ===============
	// _matcher = new PointMatcher(_filePathList, _outputDir);
	// _imgSizeList = _matcher->_imgSizeList;
	//! =============== Topology sorting ===============
	bool shallLoad = false, isInOrder = true;     //! ### set this for new data
	sortImageOrder(referNo, shallLoad, isInOrder);
	//	return false;
	//! =============== build match net ===============
	fillImageMatchNet();

	cout << "#Sequential image alignment start ..."<< endl;
	Mat_<double> identMatrix = Mat::eye(3,3,CV_64FC1);     //cvtMatrix of reference image
	_alignModelList.push_back(identMatrix);
	_initModelList.push_back(identMatrix);
	bool needRefine = false;
	for (int i = 1; i < _imgNum; i ++)
	{
		cout<<"Aligning Image "<<_visitOrder[i].imgNo<<"  ";
		solveSingleModel(i);
		if (needRefine && 0)
		{
			for (int j = 0; j < _groupCusorList.size(); j ++)
			{
				if (i == _groupCusorList[j])
				{
					int sIndex = _groupCusorList[j-1]+1;
					int eIndex = _groupCusorList[j];
					bundleAdjustingA(sIndex, eIndex);
					break;
				}
			}
		}
		cout<<"Done!"<<endl;
	}
	if (needRefine)
	{
		cout<<"-Bundle adjustment ..."<<endl;
		bundleAdjustingA(1, _imgNum-1);
	}

	cout<<"-Completed!"<<endl;
	// drawTopologyNet();
	// outputPrecise();
	cout<<"== Mosaic completed successfully!\n";
}


void ImageAligner::fillImageMatchNet()
{
	//!initialization
	for (int i = 0; i < _imgNum; i++)
	{
		Match_Net curBar;
		curBar.imgNo = i;
		_matchNetList.push_back(curBar);
	}
	int sum = 0;
	//! fill matching data
	for (int i = 0; i < _imgNum - 1; i++)
	{
		for (int j = i + 1; j < _imgNum; j++)
		{
			int PtNum = _similarityMat(i,j);
			if (PtNum == 0)
			{
				continue;
			}
			vector<Point2d> PtSet1, PtSet2;
			if (!_matcher.matches(i,j))
			{
				continue;
			}
			sum += _matcher.matches()[i][j].size();
			for (auto pair : _matcher.matches()[i][j]) {
				PtSet1.push_back(pair.first);
				PtSet2.push_back(pair.second);
			}
			int indexj = findVisitIndex(j);
			_matchNetList[i].relatedImgs.push_back(indexj);
			_matchNetList[i].PointSet.push_back(PtSet1);
			int indexi = findVisitIndex(i);
			_matchNetList[j].relatedImgs.push_back(indexi);
			_matchNetList[j].PointSet.push_back(PtSet2);
		}
	}
}


void ImageAligner::solveGroupModels(int sIndex, int eIndex)
{
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d>> pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				measureNum += pointSet[j].size();
			}
		}
	}
	int paramNum = 6*(eIndex-sIndex+1);
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	int rn = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			int neigIndex = relatedNos[j];
			if (neigIndex > i)
			{
				continue;
			}
			int neigImgNo = _visitOrder[neigIndex].imgNo;
			vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;

			vector<Point2d> curPts, neigPts;
			curPts = pointSet[j];
			//! case 1 : aligning with aligned image
			if (neigIndex < sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t ++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						Utils::pointTransform(_alignModelList[neigIndex], neigPts);
						break;
					}
				}
				int fillPos = 6*(i-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos)     = curPts[k].x; A(2*rn,fillPos+1)   = curPts[k].y; A(2*rn,fillPos+2) = 1;
					A(2*rn+1,fillPos+3) = curPts[k].x; A(2*rn+1,fillPos+4) = curPts[k].y; A(2*rn+1,fillPos+5) = 1;
					L(2*rn)   = neigPts[k].x;
					L(2*rn+1) = neigPts[k].y;
					rn ++;
				}
			}
			//! case 2 : aligning with unaligned image
			else if (neigIndex >= sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t ++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						break;
					}
				}
				int fillPos1 = 6*(i-sIndex), fillPos2 = 6*(neigIndex-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos1)     = curPts[k].x; A(2*rn,fillPos1+1)   = curPts[k].y; A(2*rn,fillPos1+2) = 1;
					A(2*rn+1,fillPos1+3) = curPts[k].x; A(2*rn+1,fillPos1+4) = curPts[k].y; A(2*rn+1,fillPos1+5) = 1;
					A(2*rn,fillPos2)     = -neigPts[k].x; A(2*rn,fillPos2+1)   = -neigPts[k].y; A(2*rn,fillPos2+2) = -1;
					A(2*rn+1,fillPos2+3) = -neigPts[k].x; A(2*rn+1,fillPos2+4) = -neigPts[k].y; A(2*rn+1,fillPos2+5) = -1;
					L(2*rn)   = 0;
					L(2*rn+1) = 0;
					rn ++;
				}
			}
		}
	}

	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	for (int i = 0; i < paramNum; i += 6)
	{
		Mat_<double> affineModel = (Mat_<double>(3,3) << X(i)  , X(i+1), X(i+2),
			                                            X(i+3), X(i+4), X(i+5),
														     0,      0,     1);
		_alignModelList.push_back(affineModel);
		_initModelList.push_back(affineModel);
	}
}


void ImageAligner::solveGroupModelsS(int sIndex, int eIndex)
{
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d>> pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				measureNum += pointSet[j].size();
			}
		}
	}
	int paramNum = 4*(eIndex-sIndex+1);
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	int rn = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			int neigIndex = relatedNos[j];
			if (neigIndex > i)
			{
				continue;
			}
			int neigImgNo = _visitOrder[neigIndex].imgNo;
			vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;

			vector<Point2d> curPts, neigPts;
			curPts = pointSet[j];
			//! case 1 : aligning with aligned image
			if (neigIndex < sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						Utils::pointTransform(_alignModelList[neigIndex], neigPts);
						break;
					}
				}

				int fillPos = 4*(i-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos)   = curPts[k].x; A(2*rn,fillPos+1)   = -curPts[k].y; A(2*rn,fillPos+2)   = 1;
					A(2*rn+1,fillPos) = curPts[k].y; A(2*rn+1,fillPos+1) =  curPts[k].x; A(2*rn+1,fillPos+3) = 1;
					L(2*rn)   = neigPts[k].x;
					L(2*rn+1) = neigPts[k].y;
					rn ++;
				}
			}
			//! case 2 : aligning with unaligned image
			else if (neigIndex >= sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t ++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						break;
					}
				}

				int fillPos1 = 4*(i-sIndex), fillPos2 = 4*(neigIndex-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos1)     = curPts[k].x; A(2*rn,fillPos1+1)   = -curPts[k].y; A(2*rn,fillPos1+2) = 1;
					A(2*rn+1,fillPos1) = curPts[k].y; A(2*rn+1,fillPos1+1) = curPts[k].x; A(2*rn+1,fillPos1+3) = 1;
					A(2*rn,fillPos2)     = -neigPts[k].x; A(2*rn,fillPos2+1)   = neigPts[k].y; A(2*rn,fillPos2+2) = -1;
					A(2*rn+1,fillPos2) = -neigPts[k].y; A(2*rn+1,fillPos2+1) = -neigPts[k].x; A(2*rn+1,fillPos2+3) = -1;
					L(2*rn)   = 0;
					L(2*rn+1) = 0;
					rn ++;
				}
			}
		}
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);

	for (int i = 0; i < paramNum; i += 4)
	{
		// force area preserving similarity transform 
		double scaleFactor = std::sqrt(X(i)*X(i) + X(i+1)*X(i+1));
		
		if (! _forceSimilarity) {
			scaleFactor = 1.0;
		}
		Mat_<double> affineModel = (Mat_<double>(3,3) << X(i)   / scaleFactor, -X(i+1) / scaleFactor, X(i+2),
			                                             X(i+1) / scaleFactor,  X(i)   / scaleFactor, X(i+3),
			                                             0,      0,     1);

		_alignModelList.push_back(affineModel);
		_initModelList.push_back(affineModel);
	}
}


void ImageAligner::solveSingleModel(int imgIndex)
{
	int measureNum = 0;
	int imgNo = _visitOrder[imgIndex].imgNo;
	vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
	vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
	for (int i = 0; i < relatedNos.size(); i ++)
	{
		if (relatedNos[i] < imgIndex)
		{
			measureNum += pointSet[i].size();
		}
	}
	int paramNum = 6;
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	int rn = 0;
	vector<Point2d> PtSet1, PtSet2;
	for (int i = 0; i < relatedNos.size(); i ++)
	{
		int neigIndex = relatedNos[i];
		if (neigIndex > imgIndex)
		{
			continue;
		}
		vector<Point2d> curPts, neigPts;
		curPts = pointSet[i];
		int neigImgNo = _visitOrder[neigIndex].imgNo;
		vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
		for (int t = 0; t < neigRelatedNos.size(); t ++)
		{
			if (neigRelatedNos[t] == imgIndex)
			{
				neigPts = _matchNetList[neigImgNo].PointSet[t];
				Utils::pointTransform(_alignModelList[neigIndex], neigPts);
				break;
			}
		}
		for (int k = 0; k < curPts.size(); k ++)
		{
			A(2*rn,0)   = curPts[k].x; A(2*rn,1)   = curPts[k].y; A(2*rn,2)   = 1;
			A(2*rn+1,3) = curPts[k].x; A(2*rn+1,4) = curPts[k].y; A(2*rn+1,5) = 1;
			L(2*rn)   = neigPts[k].x;
			L(2*rn+1) = neigPts[k].y;
			rn ++;
		}
		//! for homogaphy
		//for (int k = 0; k < curPts.size(); k ++)
		//{
		//	PtSet1.push_back(curPts[k]);
		//	PtSet2.push_back(neigPts[k]);
		//}
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	Mat_<double> affineModel = (Mat_<double>(3,3) << X(0), X(1), X(2),
		                                            X(3), X(4), X(5),
		                                               0,    0,    1);

//	Mat_<double> homoMat = findHomography(PtSet1, PtSet2, CV_RANSAC, 1.5);

	_alignModelList.push_back(affineModel);
	_initModelList.push_back(affineModel);
}


void ImageAligner::bundleAdjusting(int sIndex, int eIndex)
{
	std::cout << "Bundle adjusting ...(" << eIndex - sIndex + 1 << " images)" << std::endl;
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d>> pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				int num = pointSet[j].size();
				num = num%3 == 0 ? (num/3) : (num/3+1);
				measureNum += num;     //! only 1/3 of matching pairs for optimization
			}
		}
	}
	int paramNum = 8*(eIndex-sIndex+1);    //! optimizing homgraphic model with 8 DoF
	double *X = new double[paramNum];
	double *initX = new double[6*(eIndex-sIndex+1)];
	buildIniSolution(X, initX, sIndex, eIndex);
	//! parameters setting of least square optimization
	double lambada = _penaltyCoeffBA;
	int max_iters = 10;

	int rn = 0, ite = 0;
	while (1)
	{
		double meanBias = 0;
		rn = 0;
		Mat_<double> AtA = Mat(paramNum, paramNum, CV_64FC1, Scalar(0));
		Mat_<double> AtL = Mat(paramNum, 1, CV_64FC1, Scalar(0));
		for (int i = sIndex; i <= eIndex; i ++)
		{
			//! prepare relative data or parameters of current image
			int imgNo = _visitOrder[i].imgNo;
			vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
			vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
			for (int j = 0; j < relatedNos.size(); j ++)
			{
				int neigIndex = relatedNos[j];
				if (neigIndex > i)
				{
					continue;
				}
				vector<Point2d> curPts, neigPts;
				curPts = pointSet[j];
				int neigImgNo = _visitOrder[neigIndex].imgNo;
				vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
				for (int k = 0; k < neigRelatedNos.size(); k ++)
				{
					if (neigRelatedNos[k] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[k];
						break;
					}
				}

				int curse0 = i-sIndex, curse1 = neigIndex-sIndex;
				int fillPos0 = curse0*8, fillPos1 = curse1*8;
				int num = curPts.size(), n = 0;
				Mat_<double> Ai = Mat(num, paramNum, CV_64FC1, Scalar(0));
				Mat_<double> Li = Mat(num, 1, CV_64FC1, Scalar(0));
				double *AiPtr = (double*)Ai.data;
				double *LiPtr = (double*)Li.data;
				//! case 1 : with a fixed image
				if (neigIndex < sIndex)
				{
					Utils::pointTransform(_alignModelList[neigIndex], neigPts);
					for (int t = 0; t < curPts.size(); t += 3)
					{
						int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;		
						double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
						double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
						double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

						double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];	
						double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];

						double K1 = 2*(hX0/hW0-x1) + 2*lambada*(hX0/hW0-orgx0);
						double K2 = 2*(hY0/hW0-y1) + 2*lambada*(hY0/hW0-orgy0);

						//! for : x = ...
						AiPtr[n*paramNum+fillPos0]   = K1*x0/hW0; 
						AiPtr[n*paramNum+fillPos0+1] = K1*y0/hW0;
						AiPtr[n*paramNum+fillPos0+2] = K1*1/hW0;
						AiPtr[n*paramNum+fillPos0+3] = K2*x0/hW0;
						AiPtr[n*paramNum+fillPos0+4] = K2*y0/hW0;
						AiPtr[n*paramNum+fillPos0+5] = K2*1/hW0;
						AiPtr[n*paramNum+fillPos0+6] = -(K1+K2)*x0*hX0/(hW0*hW0);
						AiPtr[n*paramNum+fillPos0+7] = -(K1+K2)*y0*hX0/(hW0*hW0);

						double delta_d = (hX0/hW0-x1)*(hX0/hW0-x1) + (hY0/hW0-y1)*(hY0/hW0-y1);
						double delta_r = lambada*((hX0/hW0-orgx0)*(hX0/hW0-orgx0) + (hY0/hW0-orgy0)*(hY0/hW0-orgy0));
						LiPtr[n] = -(delta_d+delta_r);

						double bias = sqrt(delta_d+delta_r);
						meanBias += bias;
						n ++;
						rn ++;
					}
					//! get in normal equation matrix
					Mat_<double> Ait = Ai.t();
					Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
					AtA += barA;
					AtL += barL;
					continue;
				}

				//! case 2 : with a remain optimized image
				for (int t = 0; t < curPts.size(); t += 3)
				{
					int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;			
					double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
					double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
					double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1				
					double hX1 = X[fillPos1+0]*x1 + X[fillPos1+1]*y1 + X[fillPos1+2];     //! h1'*x1 + h2'*y1 + h3'
					double hY1 = X[fillPos1+3]*x1 + X[fillPos1+4]*y1 + X[fillPos1+5];     //! h4'*x1 + h5'*y1 + h6'
					double hW1 = X[fillPos1+6]*x1 + X[fillPos1+7]*y1 + 1;                 //! h7'*x1 + h8'*y1 + 1

					double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
					double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
					double orgx1 = initX[6*curse1+0]*x1 + initX[6*curse1+1]*y1 + initX[6*curse1+2];					
					double orgy1 = initX[6*curse1+3]*x1 + initX[6*curse1+4]*y1 + initX[6*curse1+5];

					double K1 = 2*(hX0/hW0-hX1/hW1) + 2*lambada*(hX0/hW0-orgx0);
					double K2 = 2*(hY0/hW0-hY1/hW1) + 2*lambada*(hY0/hW0-orgy0);
					double K3 = -2*(hX0/hW0-hX1/hW1) + 2*lambada*(hX1/hW1-orgx1);
					double K4 = -2*(hY0/hW0-hY1/hW1) + 2*lambada*(hY1/hW1-orgy1);

					//! for : x = ...
					//! cur-image
					AiPtr[n*paramNum+fillPos0]   = K1*x0/hW0;
					AiPtr[n*paramNum+fillPos0+1] = K1*y0/hW0;
					AiPtr[n*paramNum+fillPos0+2] = K1*1/hW0;
					AiPtr[n*paramNum+fillPos0+3] = K2*x0/hW0;
					AiPtr[n*paramNum+fillPos0+4] = K2*y0/hW0;
					AiPtr[n*paramNum+fillPos0+5] = K2*1/hW0;
					AiPtr[n*paramNum+fillPos0+6] = -(K1+K2)*x0*hX0/(hW0*hW0);
					AiPtr[n*paramNum+fillPos0+7] = -(K1+K2)*y0*hX0/(hW0*hW0);
					//! neig-image
					AiPtr[n*paramNum+fillPos1]   = K3*x1/hW1;
					AiPtr[n*paramNum+fillPos1+1] = K3*y1/hW1;
					AiPtr[n*paramNum+fillPos1+2] = K3*1/hW1;
					AiPtr[n*paramNum+fillPos1+3] = K4*x1/hW1;
					AiPtr[n*paramNum+fillPos1+4] = K4*y1/hW1;
					AiPtr[n*paramNum+fillPos1+5] = K4*1/hW1;
					AiPtr[n*paramNum+fillPos1+6] = -(K3+K4)*x1*hX1/(hW1*hW1);
					AiPtr[n*paramNum+fillPos1+7] = -(K3+K4)*y1*hX1/(hW1*hW1);

					double delta_d = (hX0/hW0-hX1/hW1)*(hX0/hW0-hX1/hW1) + (hY0/hW0-hY1/hW1)*(hY0/hW0-hY1/hW1);
					double delta_r = lambada*((hX0/hW0-orgx0)*(hX0/hW0-orgx0) + (hY0/hW0-orgy0)*(hY0/hW0-orgy0)
						                    + (hX1/hW1-orgx1)*(hX1/hW1-orgx1) + (hY1/hW1-orgy1)*(hY1/hW1-orgy1));
					LiPtr[n] = -(delta_d+delta_r);

					double bias = sqrt(delta_d+delta_r);
					meanBias += bias;
					rn ++;
					n ++;
				}
				//! get in normal equation matrix
				Mat_<double> Ait = Ai.t();
				Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
				AtA += barA;
				AtL += barL;
			}
		}
		meanBias = meanBias/rn;
		cout<<"Iteration: "<<ite<<" with cost: "<<meanBias<<endl;
		Mat_<double> dX = AtA.inv()*AtL;
		double *dXPtr = (double*)dX.data;
		double delta = 0;      //! record the translation parameters of images
		int num = 0;
		for (int i = 0; i < paramNum; i ++)
		{
			X[i] += dXPtr[i];
			if ((i+1)%8 == 3 || (i+1)%8 == 6)
			{
//				cout<<dX(i)<<endl;
				delta += abs(dXPtr[i]);
				num ++;
			}
		}
		delta = delta/num;
		if (delta < 0.08)
		{
			cout<<"Iteration has converged!"<<endl;
			break;
		}	
		if (ite++ == max_iters)
		{
			break;
		}
	}
	//! update the optimized parameters
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		double *data = (double*)_alignModelList[i].data;
		for (int j = 0; j < 8; j ++)
		{
			data[j] = X[cnt++];
		}
	}
	delete []X;
	delete []initX;
	cout<<"This optimization round is over!"<<endl;
}


void ImageAligner::bundleAdjustingA(int sIndex, int eIndex)
{
	cout<<"Bundle adjusting ...("<<eIndex-sIndex+1<<" images)"<<endl;
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				int num = pointSet[j].size();
				num = num%3 == 0 ? (num/3) : (num/3+1);
				measureNum += num;     //! only 1/3 of matching pairs for optimization
			}
		}
	}
	int paramNum = 8*(eIndex-sIndex+1);    //! optimizing homgraphic model with 8 DoF
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	double *APtr = (double*)A.data;
	double *LPtr = (double*)L.data;

	double *X = new double[paramNum];
	double *initX = new double[6*(eIndex-sIndex+1)];
	buildIniSolution(X, initX, sIndex, eIndex);
	//! parameters setting of least square optimization
	double lambada = _penaltyCoeffBA;
	int max_iters = 10;

	int rn = 0, ite = 0;
	while (1)
	{
		double meanBias = 0;
		rn = 0;
		for (int i = sIndex; i <= eIndex; i ++)
		{
			//! prepare relative data or parameters of current image
			int imgNo = _visitOrder[i].imgNo;
			vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
			vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
			for (int j = 0; j < relatedNos.size(); j ++)
			{
				int neigIndex = relatedNos[j];
				if (neigIndex > i)
				{
					continue;
				}
				vector<Point2d> curPts, neigPts;
				curPts = pointSet[j];
				int neigImgNo = _visitOrder[neigIndex].imgNo;
				vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
				for (int k = 0; k < neigRelatedNos.size(); k ++)
				{
					if (neigRelatedNos[k] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[k];
						break;
					}
				}

				int curse0 = i-sIndex, curse1 = neigIndex-sIndex;
				int fillPos0 = curse0*8, fillPos1 = curse1*8;
				//! case 1 : with a fixed image
				if (neigIndex < sIndex)
				{
					Utils::pointTransform(_alignModelList[neigIndex], neigPts);
					for (int t = 0; t < curPts.size(); t += 3)
					{
						int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;		
						double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
						double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
						double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

						//! for : x = ...
						//A(2*rn,fillPos0)   = (1+lambada)*x0/hW0;            A(2*rn,fillPos0+1) = (1+lambada)*y0/hW0;            A(2*rn,fillPos0+2) = (1+lambada)*1/hW0;
						//A(2*rn,fillPos0+6) = -(1+lambada)*x0*hX0/(hW0*hW0); A(2*rn,fillPos0+7) = -(1+lambada)*y0*hX0/(hW0*hW0);
						APtr[2*rn*paramNum+fillPos0] = (1+lambada)*x0/hW0; APtr[2*rn*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; APtr[2*rn*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
						APtr[2*rn*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  APtr[2*rn*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
						double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
						//L(2*rn) = lambada*(orgx0)+x1 - ((1+lambada)*hX0/hW0);
						LPtr[2*rn] = lambada*(orgx0)+x1 - ((1+lambada)*hX0/hW0);

						//! for : y = ...
						//A(2*rn+1,fillPos0+3) = (1+lambada)*x0/hW0;            A(2*rn+1,fillPos0+4) = (1+lambada)*y0/hW0;            A(2*rn+1,fillPos0+5) = (1+lambada)*1/hW0;
						//A(2*rn+1,fillPos0+6) = -(1+lambada)*x0*hY0/(hW0*hW0); A(2*rn+1,fillPos0+7) = -(1+lambada)*y0*hY0/(hW0*hW0);
						APtr[(2*rn+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
						APtr[(2*rn+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); APtr[(2*rn+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
						double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
						//L(2*rn+1) = lambada*(orgy0)+y1 - ((1+lambada)*hY0/hW0);
						LPtr[2*rn+1] = lambada*(orgy0)+y1 - ((1+lambada)*hY0/hW0);

						double bias = (L(2*rn)*L(2*rn) + L(2*rn+1)*L(2*rn+1));
						meanBias += sqrt(bias);
						rn ++;
					}
					continue;
				}

				//! case 2 : with a remain optimized image
				for (int t = 0; t < curPts.size(); t += 3)
				{
					int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;			
					double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
					double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
					double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

					double hX1 = X[fillPos1+0]*x1 + X[fillPos1+1]*y1 + X[fillPos1+2];     //! h1'*x1 + h2'*y1 + h3'
					double hY1 = X[fillPos1+3]*x1 + X[fillPos1+4]*y1 + X[fillPos1+5];     //! h4'*x1 + h5'*y1 + h6'
					double hW1 = X[fillPos1+6]*x1 + X[fillPos1+7]*y1 + 1;                 //! h7'*x1 + h8'*y1 + 1

					//! for : x = ...
					//! cur-image
					//A(2*rn,fillPos0)   = (1+lambada)*x0/hW0;            A(2*rn,fillPos0+1) = (1+lambada)*y0/hW0;            A(2*rn,fillPos0+2) = (1+lambada)*1/hW0;
					//A(2*rn,fillPos0+6) = -(1+lambada)*x0*hX0/(hW0*hW0); A(2*rn,fillPos0+7) = -(1+lambada)*y0*hX0/(hW0*hW0);
					APtr[2*rn*paramNum+fillPos0] = (1+lambada)*x0/hW0; APtr[2*rn*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; APtr[2*rn*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
					APtr[2*rn*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  APtr[2*rn*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
					//! neig-image
					//A(2*rn,fillPos1)   = (lambada-1)*x1/hW1;            A(2*rn,fillPos1+1) = (lambada-1)*y1/hW1;            A(2*rn,fillPos1+2) = (lambada-1)*1/hW1;
					//A(2*rn,fillPos1+6) = -(lambada-1)*x1*hX1/(hW1*hW1); A(2*rn,fillPos1+7) = -(lambada-1)*y1*hX1/(hW1*hW1);
					APtr[2*rn*paramNum+fillPos1] = (lambada-1)*x1/hW1; APtr[2*rn*paramNum+fillPos1+1] = (lambada-1)*y1/hW1; APtr[2*rn*paramNum+fillPos1+2] = (lambada-1)*1/hW1;
					APtr[2*rn*paramNum+fillPos1+6] = -(lambada-1)*x1*hX1/(hW1*hW1);  APtr[2*rn*paramNum+fillPos1+7] = -(lambada-1)*y1*hX1/(hW1*hW1);

					double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
					double orgx1 = initX[6*curse1+0]*x1 + initX[6*curse1+1]*y1 + initX[6*curse1+2];
					//L(2*rn) = lambada*(orgx0+orgx1) - ((1+lambada)*hX0/hW0 + (lambada-1)*hX1/hW1);
					LPtr[2*rn] = lambada*(orgx0+orgx1) - ((1+lambada)*hX0/hW0 + (lambada-1)*hX1/hW1);

					//! for : y = ...
					//! cur-image
					//A(2*rn+1,fillPos0+3) = (1+lambada)*x0/hW0;            A(2*rn+1,fillPos0+4) = (1+lambada)*y0/hW0;            A(2*rn+1,fillPos0+5) = (1+lambada)*1/hW0;
					//A(2*rn+1,fillPos0+6) = -(1+lambada)*x0*hY0/(hW0*hW0); A(2*rn+1,fillPos0+7) = -(1+lambada)*y0*hY0/(hW0*hW0);
					APtr[(2*rn+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
					APtr[(2*rn+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); APtr[(2*rn+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
					//! neig-image
					//A(2*rn+1,fillPos1+3) = (lambada-1)*x1/hW1;            A(2*rn+1,fillPos1+4) = (lambada-1)*y1/hW1;            A(2*rn+1,fillPos1+5) = (lambada-1)*1/hW1;
					//A(2*rn+1,fillPos1+6) = -(lambada-1)*x1*hY1/(hW1*hW1); A(2*rn+1,fillPos1+7) = -(lambada-1)*y1*hY1/(hW1*hW1);
					APtr[(2*rn+1)*paramNum+fillPos1+3] = (lambada-1)*x1/hW1; APtr[(2*rn+1)*paramNum+fillPos1+4] = (lambada-1)*y1/hW1; APtr[(2*rn+1)*paramNum+fillPos1+5] = (lambada-1)*1/hW1;
					APtr[(2*rn+1)*paramNum+fillPos1+6] = -(lambada-1)*x1*hY1/(hW1*hW1); APtr[(2*rn+1)*paramNum+fillPos1+7] = -(lambada-1)*y1*hY1/(hW1*hW1);

					double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
					double orgy1 = initX[6*curse1+3]*x1 + initX[6*curse1+4]*y1 + initX[6*curse1+5];
					//L(2*rn+1) = lambada*(orgy0+orgy1) - ((1+lambada)*hY0/hW0 + (lambada-1)*hY1/hW1);
					LPtr[2*rn+1] = lambada*(orgy0+orgy1) - ((1+lambada)*hY0/hW0 + (lambada-1)*hY1/hW1);

					double bias = (L(2*rn)*L(2*rn) + L(2*rn+1)*L(2*rn+1));
					meanBias += sqrt(bias);
					rn ++;
				}
			}
		}
		meanBias = meanBias/rn;
		cout<<"Iteration: "<<ite<<" with cost: "<<meanBias<<endl;
		Mat_<double> At = A.t();
		Mat_<double> dX = (At*A).inv()*(At*L);
		//		cout<<A.t()*A<<endl<<(A.t()*L)<<endl;
		double *dXPtr = (double*)dX.data;
		//		cout<<dX<<endl;
		double delta = 0;      //! record the translation parameters of images
		int num = 0;
		for (int i = 0; i < paramNum; i ++)
		{
			X[i] += dXPtr[i];
			if ((i+1)%8 == 3 || (i+1)%8 == 6)
			{
				//				cout<<dX(i)<<endl;
				delta += abs(dXPtr[i]);
				num ++;
			}
		}
		delta = delta/num;
		if (delta < 0.08)
		{
			cout<<"Iteration has converged!"<<endl;
			break;
		}	
		if (ite++ == max_iters)
		{
			break;
		}
	}
	//! update the optimized parameters
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		double *data = (double*)_alignModelList[i].data;
		for (int j = 0; j < 8; j ++)
		{
			data[j] = X[cnt++];
		}
	}
	delete []X;
	delete []initX;
	cout<<"This optimization round is over!"<<endl;
}


void ImageAligner::bundleAdjustinga(int sIndex, int eIndex)
{
	std::cout << "Bundle adjusting ..." << std::endl;
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				int num = pointSet[j].size();
				num = num%3 == 0 ? (num/3) : (num/3+1);
				measureNum += num;     //! only 1/3 of matching pairs for optimization
			}
		}
	}
	int paramNum = 8*(eIndex-sIndex+1);    //! optimizing homgraphic model with 8 DoF
	double *X = new double[paramNum];
	double *initX = new double[6*(eIndex-sIndex+1)];
	buildIniSolution(X, initX, sIndex, eIndex);
	//! parameters setting of least square optimization
	double lambada = _penaltyCoeffBA;
	int max_iters = 20;

	int rn = 0, ite = 0;
	bool converged = false;
	while (1)
	{
		double meanBias = 0;
		rn = 0;
		Mat_<double> AtA = Mat(paramNum, paramNum, CV_64FC1, Scalar(0));
		Mat_<double> AtL = Mat(paramNum, 1, CV_64FC1, Scalar(0));

  		std::vector<int> ivec(eIndex - sIndex + 1);
  		std::iota(ivec.begin(), ivec.end(), sIndex);

		std::for_each(std::execution::par_unseq, ivec.begin(), ivec.end(), [&](auto i) {

			// std::cout << "Bundle adjusting index " << i << " / " << eIndex << " " << _outputDir << std::endl;

			//! prepare relative data or parameters of current image
			int imgNo = _visitOrder[i].imgNo;
			vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
			vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
			for (int j = 0; j < relatedNos.size(); j ++)
			{
				int neigIndex = relatedNos[j];
				if (neigIndex > i)
				{
					continue;
				}
				vector<Point2d> curPts, neigPts;
				curPts = pointSet[j];
				int neigImgNo = _visitOrder[neigIndex].imgNo;
				vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
				for (int k = 0; k < neigRelatedNos.size(); k ++)
				{
					if (neigRelatedNos[k] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[k];
						break;
					}
				}

				int curse0 = i-sIndex, curse1 = neigIndex-sIndex;
				int fillPos0 = curse0*8, fillPos1 = curse1*8;
				int num = curPts.size(), n = 0;
				Mat_<double> Ai = Mat(2*num, paramNum, CV_64FC1, Scalar(0));
				Mat_<double> Li = Mat(2*num, 1, CV_64FC1, Scalar(0));
				double *AiPtr = (double*)Ai.data;
				double *LiPtr = (double*)Li.data;
				//! case 1 : with a fixed image
				if (neigIndex < sIndex)
				{
					Utils::pointTransform(_alignModelList[neigIndex], neigPts);
					for (int t = 0; t < curPts.size(); t += 3)
					{
						int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;		
						double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
						double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
						double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

						//! for : x = ...
						AiPtr[2*n*paramNum+fillPos0]   = (1+lambada)*x0/hW0; 
						AiPtr[2*n*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; 
						AiPtr[2*n*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
						AiPtr[2*n*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  
						AiPtr[2*n*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
						double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];

						LiPtr[2*n] = lambada*(orgx0)+x1 - ((1+lambada)*hX0/hW0);

						//! for : y = ...
						AiPtr[(2*n+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0;
						AiPtr[(2*n+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; 
						AiPtr[(2*n+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
						AiPtr[(2*n+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); 
						AiPtr[(2*n+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
						double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];

						LiPtr[2*n+1] = lambada*(orgy0)+y1 - ((1+lambada)*hY0/hW0);

						double bias = (LiPtr[2*n]*LiPtr[2*n] + LiPtr[2*n+1]*LiPtr[2*n+1]);
						meanBias += sqrt(bias);
						n ++;
						rn ++;
					}
					//! get in normal equation matrix
					Mat_<double> Ait = Ai.t();
					Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
					AtA += barA;
					AtL += barL;

					continue;
				}

				//! case 2 : with a remain optimized image
				for (int t = 0; t < curPts.size(); t += 3)
				{
					int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;			
					double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
					double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
					double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

					double hX1 = X[fillPos1+0]*x1 + X[fillPos1+1]*y1 + X[fillPos1+2];     //! h1'*x1 + h2'*y1 + h3'
					double hY1 = X[fillPos1+3]*x1 + X[fillPos1+4]*y1 + X[fillPos1+5];     //! h4'*x1 + h5'*y1 + h6'
					double hW1 = X[fillPos1+6]*x1 + X[fillPos1+7]*y1 + 1;                 //! h7'*x1 + h8'*y1 + 1

					//! for : x = ...
					//! cur-image
					AiPtr[2*n*paramNum+fillPos0]   = (1+lambada)*x0/hW0; 
					AiPtr[2*n*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; 
					AiPtr[2*n*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
					AiPtr[2*n*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  
					AiPtr[2*n*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
					//! neig-image
					AiPtr[2*n*paramNum+fillPos1]   = (lambada-1)*x1/hW1; 
					AiPtr[2*n*paramNum+fillPos1+1] = (lambada-1)*y1/hW1; 
					AiPtr[2*n*paramNum+fillPos1+2] = (lambada-1)*1/hW1;
					AiPtr[2*n*paramNum+fillPos1+6] = -(lambada-1)*x1*hX1/(hW1*hW1);  
					AiPtr[2*n*paramNum+fillPos1+7] = -(lambada-1)*y1*hX1/(hW1*hW1);

					double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
					double orgx1 = initX[6*curse1+0]*x1 + initX[6*curse1+1]*y1 + initX[6*curse1+2];

					LiPtr[2*n] = lambada*(orgx0+orgx1) - ((1+lambada)*hX0/hW0 + (lambada-1)*hX1/hW1);

					//! for : y = ...
					//! cur-image
					AiPtr[(2*n+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0; 
					AiPtr[(2*n+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; 
					AiPtr[(2*n+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
					AiPtr[(2*n+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); 
					AiPtr[(2*n+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
					//! neig-image
					AiPtr[(2*n+1)*paramNum+fillPos1+3] = (lambada-1)*x1/hW1; 
					AiPtr[(2*n+1)*paramNum+fillPos1+4] = (lambada-1)*y1/hW1; 
					AiPtr[(2*n+1)*paramNum+fillPos1+5] = (lambada-1)*1/hW1;
					AiPtr[(2*n+1)*paramNum+fillPos1+6] = -(lambada-1)*x1*hY1/(hW1*hW1); 
					AiPtr[(2*n+1)*paramNum+fillPos1+7] = -(lambada-1)*y1*hY1/(hW1*hW1);

					double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
					double orgy1 = initX[6*curse1+3]*x1 + initX[6*curse1+4]*y1 + initX[6*curse1+5];

					LiPtr[2*n+1] = lambada*(orgy0+orgy1) - ((1+lambada)*hY0/hW0 + (lambada-1)*hY1/hW1);

					double bias = (LiPtr[2*n]*LiPtr[2*n] + LiPtr[2*n+1]*LiPtr[2*n+1]);
					meanBias += sqrt(bias);
					n ++;
					rn ++;
				}
				//! get in normal equation matrix
				Mat_<double> Ait = Ai.t();
				Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
				AtA += barA;
				AtL += barL;
			}
		});

		meanBias = meanBias/rn;
		// cout<<"Iteration: "<<ite<<" with cost: "<<meanBias<<endl;

		Mat_<double> dX = AtA.inv()*AtL;
		double *dXPtr = (double*)dX.data;
		//		cout<<dX<<endl;
		double delta = 0;      //! record the translation parameters of images
		int num = 0;
		for (int i = 0; i < paramNum; i ++)
		{
			X[i] += dXPtr[i];
			if ((i+1)%8 == 3 || (i+1)%8 == 6)
			{
				//				cout<<dX(i)<<endl;
				delta += abs(dXPtr[i]);
				num ++;
			}
		}
		delta = delta/num;

		if (delta < 0.08)
		{
			break;
		}	
		if (ite++ == max_iters)
		{
			// if (meanBias > 3.0) {
			// 	throw std::logic_error("Bundle adjustment diverged!");
			// } 
			break;
		}
		converged = true;
	}

	delete []X;
	delete []initX;
	
	if (!converged) 
	{
		return;
	}
	//! update the optimized parameters
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		double *data = (double*)_alignModelList[i].data;
		for (int j = 0; j < 8; j ++)
		{
			data[j] = X[cnt++];
		}
	}

	// cout<<"This optimization round is over!"<<endl;
}


void ImageAligner::buildIniSolution(double* X, double* initX, int sIndex, int eIndex)
{
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		Mat_<double> tempMat = _alignModelList[i];
		double *data = (double*)tempMat.data;
		for (int j = 0; j < 8; j ++)
		{
			X[cnt++] = data[j];
		}
	}
	cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		Mat_<double> tempMat = _initModelList[i];
		double *data = (double*)tempMat.data;
		for (int j = 0; j < 6; j ++)
		{
			initX[cnt++] = data[j];
		}
	}
}


void ImageAligner::buildIniSolution(double* X, int sIndex, int eIndex)
{
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		Mat_<double> tempMat = _alignModelList[i];
		double *data = (double*)tempMat.data;
		for (int j = 0; j < 8; j ++)
		{
			X[cnt++] = data[j];
		}
	}
}


void ImageAligner::RefineAligningModels(int sIndex, int eIndex)
{
	std::cout << "Refining stitch parameters ..." << std::endl;
	int m = 0, n = 0, max_its = 200;
	m = (eIndex-sIndex+1) * 8;       //without optimizing the start one
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int curNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[curNo].PointSet;
		vector<int> relatedNos = _matchNetList[curNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)
			{
				//! using only one third of corresponding for optimization
				n += pointSet[j].size()/3;
			}
		}
	}
	double *d = new double[n];
	for (int i = 0; i < n; i ++)
	{
		d[i] = 0;
	}
	double *X = new double[m];
	buildIniSolution(X, sIndex, eIndex);

	LMData *LMInput = new LMData;        //! prepare input data for optimization
	LMInput->sIndex = sIndex;
	LMInput->eIndex = eIndex;
	LMInput->matchPtr = &_matchNetList;
	LMInput->modelPtr = &_alignModelList;
	LMInput->visitOrder = _visitOrder;

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=1E-20; opts[1]=1E-30; opts[2]=1E-30; opts[3]=1E-30;	opts[4]= 1e-9;

	int ret = dlevmar_dif(OptimizationFunction, X, d, m, n, max_its, opts, info, NULL, NULL, (void*)LMInput);
	for (int i = sIndex; i <= eIndex; i ++)    //stock optimized homographic matrix
	{
		double *homoVec = X + 8*(i-sIndex);
		Mat_<double> homoMat(3,3,CV_64FC1);
		homoMat(0,0) = homoVec[0]; homoMat(0,1) = homoVec[1]; homoMat(0,2) = homoVec[2];
		homoMat(1,0) = homoVec[3]; homoMat(1,1) = homoVec[4]; homoMat(1,2) = homoVec[5];
		homoMat(2,0) = homoVec[6]; homoMat(2,1) = homoVec[7]; homoMat(2,2) = 1.0;
		_alignModelList[i] = homoMat;
	}

	delete []d;      //free stack
	delete []X;
	delete LMInput;
}


void ImageAligner::OptimizationFunction(double* X, double* d, int m, int n, void* data)
{
	LMData *dataPtr = (LMData *)data;
	vector<Match_Net> *matchNetPtr = dataPtr->matchPtr;
	int startIndex = dataPtr->sIndex, endIndex = dataPtr->eIndex;
	vector<Mat_<double> > *modelListPtr = dataPtr->modelPtr;
	vector<TreeNode> visitOrder = dataPtr->visitOrder;
	int cnt = 0;
	double meanError = 0;

	std::vector<int> ivec(endIndex - startIndex + 1);
  	std::iota(ivec.begin(), ivec.end(), startIndex);

	std::for_each(std::execution::par_unseq, ivec.begin(), ivec.end(), [&](auto i) {
	// for (int i = startIndex; i <= endIndex; i ++)       //without the start image
	// {
		int curNo = visitOrder[i].imgNo;
		Match_Net matchNet = (*matchNetPtr)[curNo];
		vector<int> relatedImgs = matchNet.relatedImgs;
		double *homoVec1, *homoVec2;
		double *iniHomoVec1, *iniHomoVec2;
		homoVec1 = X + 8*(i-startIndex);
		iniHomoVec1 = (double*)(*modelListPtr)[i].data;

		for (int j = 0; j < relatedImgs.size(); j ++)
		{
			int neigIndex = relatedImgs[j];
			if (neigIndex > i)
			{
				continue;
			}
			vector<Point2d> ptSet1, ptSet2;
			ptSet1 = matchNet.PointSet[j];         //! points on cur_image
			int neigNo = visitOrder[neigIndex].imgNo;
			Match_Net neigMatchNet = (*matchNetPtr)[neigNo];
			for (int k = 0; k < neigMatchNet.relatedImgs.size(); k ++)
			{
				if (neigMatchNet.relatedImgs[k] == i)
				{
					ptSet2 = neigMatchNet.PointSet[k];
					break;
				}
			}
			bool fixedNeigh = false;
			if (neigIndex < startIndex)
			{
				double *data = (double*)(*modelListPtr)[neigIndex].data;
				homoVec2 = data;
				fixedNeigh = true;
			}
			else
			{
				homoVec2 = X + 8*(neigIndex-startIndex);
				iniHomoVec2 = (double*)(*modelListPtr)[neigIndex].data;
			}
			//! using only one third of corresponding for optimization
			for (int t = 0; t < ptSet1.size(); t += 3)
			{
				double x1 = ptSet1[t].x, y1 = ptSet1[t].y;
				double x2 = ptSet2[t].x, y2 = ptSet2[t].y;
				double mosaic_x1 = (homoVec1[0]*x1 + homoVec1[1]*y1 + homoVec1[2])/(homoVec1[6]*x1 + homoVec1[7]*y1 + 1.0);
				double mosaic_y1 = (homoVec1[3]*x1 + homoVec1[4]*y1 + homoVec1[5])/(homoVec1[6]*x1 + homoVec1[7]*y1 + 1.0);
				double mosaic_x2 = (homoVec2[0]*x2 + homoVec2[1]*y2 + homoVec2[2])/(homoVec2[6]*x2 + homoVec2[7]*y2 + 1.0);
				double mosaic_y2 = (homoVec2[3]*x2 + homoVec2[4]*y2 + homoVec2[5])/(homoVec2[6]*x2 + homoVec2[7]*y2 + 1.0);
				double bias = sqrt((mosaic_x1-mosaic_x2)*(mosaic_x1-mosaic_x2)+(mosaic_y1-mosaic_y2)*(mosaic_y1-mosaic_y2));

				//! perspective penalty items
				double penalty = 0;
				if (!fixedNeigh)
				{
					double iniMosaic_x1 = (iniHomoVec1[0]*x1 + iniHomoVec1[1]*y1 + iniHomoVec1[2])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					double iniMosaic_y1 = (iniHomoVec1[3]*x1 + iniHomoVec1[4]*y1 + iniHomoVec1[5])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					double iniMosaic_x2 = (iniHomoVec2[0]*x2 + iniHomoVec2[1]*y2 + iniHomoVec2[2])/(iniHomoVec2[6]*x2 + iniHomoVec2[7]*y2 + 1.0);
					double iniMosaic_y2 = (iniHomoVec2[3]*x2 + iniHomoVec2[4]*y2 + iniHomoVec2[5])/(iniHomoVec2[6]*x2 + iniHomoVec2[7]*y2 + 1.0);
					double penalty1 = sqrt((mosaic_x1-iniMosaic_x1)*(mosaic_x1-iniMosaic_x1)+(mosaic_y1-iniMosaic_y1)*(mosaic_y1-iniMosaic_y1));
					double penalty2 = sqrt((mosaic_x2-iniMosaic_x2)*(mosaic_x2-iniMosaic_x2)+(mosaic_y2-iniMosaic_y2)*(mosaic_y2-iniMosaic_y2));
					penalty = (penalty1 + penalty2)/2;
				}
				else
				{
					double iniMosaic_x1 = (iniHomoVec1[0]*x1 + iniHomoVec1[1]*y1 + iniHomoVec1[2])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					double iniMosaic_y1 = (iniHomoVec1[3]*x1 + iniHomoVec1[4]*y1 + iniHomoVec1[5])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					penalty = sqrt((mosaic_x1-iniMosaic_x1)*(mosaic_x1-iniMosaic_x1)+(mosaic_y1-iniMosaic_y1)*(mosaic_y1-iniMosaic_y1));
				}
				d[cnt++] = bias + _penaltyCoeffLM * penalty;
				meanError += bias;
			}
		}
	});
	meanError /= cnt;
	// cout<<"current mean-warping-bias is: "<<meanError<<endl;
}


int ImageAligner::findVisitIndex(int imgNo)
{
	int imgIndex = 0;
	for (int i = 0; i < _imgNum; i ++)
	{
		if (_visitOrder[i].imgNo == imgNo)
		{
			imgIndex = i;
		}
	}
	return imgIndex;
}


