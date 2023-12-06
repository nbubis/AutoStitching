#include "topology.h"
#define BKGRNDPIX 0

Mat_<double> TopoFinder::findTopology(bool shallLoad, bool isInOrder)
{
	_isInOrder = isInOrder;

	loadKeyFiles();
	clock_t start_time, end_time;
	// cout<<"Building similarity table ...\n";
	Mat_<int> similarMat = detectSimilarityByGuiding();
	return similarMat;
}


Mat_<int> TopoFinder::detectSimilarityOnGlobal()
{
	Mat_<int> similarMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	for (int i = 0; i < _imgNum; i++)
	{
		for (int j = 0; j < _imgNum; j++)
		{		
			similarMat(i, j) = _matcher.matches()[i][j].size();
			similarMat(j, i) = _matcher.matches()[i][j].size();
		}
	}

	return similarMat;
}


Mat_<int> TopoFinder::detectSimilarityByGuiding()
{
	bool isTimeConsecutive = _isInOrder;
	_similarityMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	_attempMap = Mat::eye(_imgNum, _imgNum, CV_16UC1);
	//! find and match main chain, meanwhile edit the similarity table
	if (isTimeConsecutive)
	{
		buildMainChain();
	}
	else
	{
		searchMainChain();
	}
	//! solve the aligning models of each image
	Mat_<double> identMatrix = Mat::eye(3,3,CV_64FC1);
	_affineMatList.push_back(identMatrix);
	Quadra bar;
	bar.imgSize = _matcher.imgSizeList()[_visitOrder0[0].imgNo];
	bar.centroid = Point2d(bar.imgSize.width/2, bar.imgSize.height/2);
	_projCoordSet.push_back(bar);
	// cout<<"Detecting potential overlaps ..."<<endl;
	for (int i = 1; i < _visitOrder0.size(); i ++)
	{
		// cout<<"No."<<i<<" finding ...";
		int curNo = _visitOrder0[i].imgNo;
		int refNo = _visitOrder0[i].refNo;
		int refIndex = findNodeIndex(refNo);
		vector<Point2d> pointSet1, pointSet2;
		clock_t st, et;
		_matcher.getMatchPoints(refNo, curNo, pointSet1, pointSet2);
	    Utils::pointTransform(_affineMatList[refIndex], pointSet1);
		//! perform initial alignment
		Mat_<double> affineMat = findFastAffine(pointSet1, pointSet2);
		_affineMatList.push_back(affineMat);
		//! record centroid of current image
		Quadra bar;
		bar.imgSize = _matcher.imgSizeList()[curNo];
		Point2d centroid(bar.imgSize.width/2.0, bar.imgSize.height/2.0);	
		bar.centroid = Utils::pointTransform(affineMat, centroid);
		_projCoordSet.push_back(bar);

		//! detect potential overlaps
		//! 1) recalculate aligning model; 2) modify centroid; 3) modify similarity table
		detectPotentialOverlap(i, pointSet1, pointSet2);
		// cout<<"-->end!"<<endl;
	}
	// drawTopoNet();
//	drawTreeLevel();
//	TsaveMosaicImage();
	return _similarityMat;
}


Mat_<double> TopoFinder::getGuidingTable()
{
	cout << "Initializing ..." << endl;
	Mat_<double> guidingTable = Mat(_imgNum, _imgNum, CV_64FC1, Scalar(0));
	Mat_<int> simiMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{		
			int num = calSimilarNum(i,j);
			simiMat(i,j) = num;
			simiMat(j,i) = num;
			double cost = 6/log(num+50.0);
			if (num == 0)
			{
				cost = -1;
			}
			guidingTable(i,j) = cost;
			guidingTable(j,i) = cost;
		}
	}
	return guidingTable;
}


Mat_<double> TopoFinder::getGuidingTableP()
{
	cout << "Initializing ..." << endl;
	Mat_<double> guidingTable = Mat(_imgNum, _imgNum, CV_64FC1, Scalar(0));
	Mat_<int> simiMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	int step = max(1,(_imgNum-1)*_imgNum/20);      //! 10%
	int n = 0;
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{		
			int num = calSimilarNum(i,j);
			simiMat(i,j) = num;
			simiMat(j,i) = num;
			double cost = 6/log(num+50.0);
			if (num == 0)
			{
				cost = -1;
			}
			guidingTable(i,j) = cost;
			guidingTable(j,i) = cost;
			n ++;
			if (n%step == 0)
			{
				cout<<10*n/step<<"% ";
			}
		}
	}

	return guidingTable;
}


void TopoFinder::buildMainChain()
{
	// cout<<"Building main chain according to the time consecutive order ..."<<endl;
	int refeNo = _imgNum/2;
	TreeNode bar(refeNo,-1,0);
	_visitOrder0.push_back(bar);
	int offset = 1;
	while (1)
	{
		int no1 = refeNo-offset, no2 = refeNo+offset;
		TreeNode bar1(no1,no1+1,0), bar2(no2,no2-1,0);
		_visitOrder0.push_back(bar1);
		_visitOrder0.push_back(bar2);
		vector<Point2d> pointSet1, pointSet2;

		for (auto pair : _matcher.matches()[no1][no1 + 1]) {
			pointSet1.push_back(pair.first);
			pointSet2.push_back(pair.second);
		}

		_similarityMat(no1, no1+1) = pointSet1.size();
		_similarityMat(no1+1, no1) = pointSet1.size();
		_attempMap(no1, no1+1) = 1;
		_attempMap(no1+1, no1) = 1;

		pointSet1.clear();
		pointSet2.clear();
		for (auto pair : _matcher.matches()[no2][no2 - 1]) {
			pointSet1.push_back(pair.first);
			pointSet2.push_back(pair.second);
		}
		_similarityMat(no2, no2-1) = pointSet1.size();
		_similarityMat(no2-1, no2) = pointSet1.size();
		_attempMap(no2, no2-1) = 1;
		_attempMap(no2-1, no2) = 1;
		offset ++;
		if (no1 == 0 || no2 == _imgNum-1)
		{
			if (no1 > 0)
			{
				_visitOrder0.push_back(TreeNode(no1-1,no1,0));
				pointSet1.clear();
				pointSet2.clear();
				for (auto pair : _matcher.matches()[no1 - 1][no1]) {
					pointSet1.push_back(pair.first);
					pointSet2.push_back(pair.second);
				}

				_similarityMat(no1-1, no1) = pointSet1.size();
				_similarityMat(no1, no1-1) = pointSet1.size();
				_attempMap(no1-1, no1) = 1;
				_attempMap(no1, no1-1) = 1;
			}
			else if (no2 < _imgNum-1)
			{
				_visitOrder0.push_back(TreeNode(no2+1,no2,0));

				pointSet1.clear();
				pointSet2.clear();
				for (auto pair : _matcher.matches()[no2 + 1][no2]) {
					pointSet1.push_back(pair.first);
					pointSet2.push_back(pair.second);
				}

				_similarityMat(no2+1, no2) = pointSet1.size();
				_similarityMat(no2, no2+1) = pointSet1.size();
				_attempMap(no2+1, no2) = 1;
				_attempMap(no2, no2+1) = 1;
			}
			break;
		}
	}
	_attempNum = _imgNum-1;
	_shotNum = _imgNum-1;
	// cout<<"Completed!"<<endl;
}


void TopoFinder::searchMainChain()
{
	//! overlapping probability of image pairs
	Mat_<double> guidingTable = getGuidingTableP();
	int iter = 0, maxIter = max(int(_imgNum*0.2), 100);

	std::vector <int> image_matches(_imgNum, 0);

	while (1)
	{
		cout <<"Searching main chain ... (attempt: " << ++iter << ")" << endl;
		Mat_<int> imgPairs= Graph::extractMSTree(guidingTable);
		int pairNo = 0;

		for (pairNo = 0; pairNo < imgPairs.rows; pairNo ++)
		{
			int no1 = imgPairs(pairNo, 0), no2 = imgPairs(pairNo,1);

			//! avoiding repeating matching which is done in last iteration
			if (_attempMap(no1,no2) != 0)
			{
				continue;
			}
			_attempMap(no1,no2) = 1;
			_attempMap(no2,no1) = 1;
			vector<Point2d> pointSet1, pointSet2;
			_attempNum ++;

			clock_t st, et;
			st = clock();
			bool yeah = true;//_matcher.featureMatcher(no1,no2,pointSet1,pointSet2);
			et = clock();
			_matchTime += (et-st);
			if (yeah)
			{
				image_matches[no1] += 1;
				image_matches[no2] += 1;

				_shotNum ++;
				_similarityMat(no1,no2) = pointSet1.size();
				_similarityMat(no2,no1) = pointSet1.size();
				guidingTable(no1,no2) = 0.0;
				guidingTable(no2,no1) = 0.0;
			}
			else
			{
				//! matching failed : cost as infinite
				guidingTable(no1,no2) = 999;
				guidingTable(no2,no1) = 999;

				if (iter == maxIter)
				{
					// cout << image_matches << endl;
					cout << "Poor image sequence! exit out." << endl;
					throw std::logic_error("no matches");
				}
				for (auto i: image_matches)
    				std::cout << i << ' ';
				cout << no1 <<" Linking "<< no2 <<" failed! < built: "<< pairNo +1 <<" edges."<<endl;
				break;
			}
		}
		if (pairNo == _imgNum-1)
		{
			for (auto i: image_matches)
    			std::cout << i << ' ';
			break;
		}
	}
	cout << "Succeed!" << endl;
	Mat_<double> costGraph = Utils::buildCostGraph(_similarityMat);
	_visitOrder0 = Graph::FloydForPath(costGraph);
}


void TopoFinder::detectPotentialOverlap(int curIndex, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2)
{
	int curRefNo = _visitOrder0[curIndex].refNo;
	int curNo = _visitOrder0[curIndex].imgNo;
	Point2d iniPos = _projCoordSet[curIndex].centroid;
	int width = _projCoordSet[curIndex].imgSize.width;
	int height = _projCoordSet[curIndex].imgSize.height;
	//! accelerate : build a KD-tree for all centroids and retrieve
	bool isGot = false;
	for (int i = 0; i < _projCoordSet.size(); i ++)
	{
		int testNo = _visitOrder0[i].imgNo;
		if (_attempMap(curNo,testNo))
		{
			continue;
		}
		Quadra testObj = _projCoordSet[i];
		double threshold = 0.5*(max(width,height) + max(testObj.imgSize.width, testObj.imgSize.height));
		double dist = cv::norm(iniPos - testObj.centroid);
		if (dist > threshold*0.8)
		{
			continue;
		}
		_attempNum ++;
		vector<Point2d> newPtSet1, newPtSet2;
		
		if (dist < threshold*0.8)
		{
			bool yeah = _matcher.matches(testNo,curNo);
			if (yeah)
			{
				_similarityMat(testNo,curNo) = newPtSet1.size();
				_similarityMat(curNo,testNo) = newPtSet1.size();
				Utils::pointTransform(_affineMatList[i], newPtSet1);
				for (int t = 0; t < newPtSet1.size(); t ++)
				{
					pointSet1.push_back(newPtSet1[t]);
					pointSet2.push_back(newPtSet2[t]);
				}
				_shotNum ++;
				isGot = true;
			}
		}
		else
		{
			if (_matcher.matches(testNo,curNo))
			{
				for (auto pair : _matcher.matches()[testNo][curNo]) {
					newPtSet1.push_back(pair.first);
					newPtSet1.push_back(pair.second);
				}
				_similarityMat(testNo,curNo) = newPtSet1.size();
				_similarityMat(curNo,testNo) = newPtSet1.size();
				Utils::pointTransform(_affineMatList[i], newPtSet1);
				for (int t = 0; t < newPtSet1.size(); t ++)
				{
					pointSet1.push_back(newPtSet1[t]);
					pointSet2.push_back(newPtSet2[t]);
				}
				_shotNum ++;
				isGot = true;
			}		
		}

	}
	if (!isGot)
	{
		return;
	}
	//! modify the affine model parameter
	_affineMatList[curIndex] = findFastAffine(pointSet1, pointSet2);
	//! modify the centroid
	_projCoordSet[curIndex].centroid = Utils::pointTransform(_affineMatList[curIndex], Point2d(width/2,height/2));
}


int TopoFinder::findNodeIndex(int imgNo)
{
	int imgIndex = 0;
	for (int i = 0; i < _imgNum; i ++)
	{
		if (_visitOrder0[i].imgNo == imgNo)
		{
			imgIndex = i;
		}
	}
	return imgIndex;
}


Mat_<double> TopoFinder::findFastAffine(vector<Point2d> pointSet1, vector<Point2d> pointSet2)
{
	int step = max(1, int(pointSet1.size()/500));
	int pointNum = pointSet1.size()/step;
	Mat_<double> affineMat(3, 3, CV_64FC1);
	Mat A(2*pointNum, 6, CV_64FC1, Scalar(0));
	Mat L(2*pointNum, 1, CV_64FC1);
	for (int i = 0; i < pointNum; i ++)
	{
		double x1 = pointSet1[i*step].x, y1 = pointSet1[i*step].y;
		double x2 = pointSet2[i*step].x, y2 = pointSet2[i*step].y;
		A.at<double>(i*2,0) = x2; A.at<double>(i*2,1) = y2; A.at<double>(i*2,2) = 1; 
		A.at<double>(i*2+1,3) = x2; A.at<double>(i*2+1,4) = y2; A.at<double>(i*2+1,5) = 1;
		L.at<double>(i*2,0) = x1;
		L.at<double>(i*2+1,0) = y1;
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	affineMat(0,0) = X(0); affineMat(0,1) = X(1); affineMat(0,2) = X(2);
	affineMat(1,0) = X(3); affineMat(1,1) = X(4); affineMat(1,2) = X(5);
	affineMat(2,0) = 0; affineMat(2,1) = 0; affineMat(2,2) = 1;
	double var = 0;
	//Utils::pointTransform(affineMat, pointSet2);
	//for (int i = 0; i < pointNum; i ++)
	//{

	//	double bias = (pointSet1[i].x-pointSet2[i].x)*(pointSet1[i].x-pointSet2[i].x) + 
 //                     (pointSet1[i].y-pointSet2[i].y)*(pointSet1[i].y-pointSet2[i].y);
	//	var += bias;
	//}
	//var /= (pointNum-6);
	return affineMat;
}


void TopoFinder::loadKeyFiles()
{
	const int targetOctave = 1; 

	for (int i = 0; i < _matcher.keyPts().size(); i++) {
		Keys bar;
		std::vector<int> subIndexList;
		bar.descriptors = _matcher.descriptors()[i];

		for (int j = 0; j < _matcher.keyPts()[i].size(); j++) {
			bar.pts.push_back(_matcher.keyPts()[i][j].pt);
			if (_matcher.keyPts()[i][j].octave == targetOctave)
			{
				subIndexList.push_back(j);
			}
		}
		_keyList.push_back(bar);
		_subKeyIndexList.push_back(subIndexList);
	}
}

int TopoFinder::calSimilarNum(int imgIndex1, int imgIndex2)
{
	vector<Point2d> orgKeyPts1 = _keyList[imgIndex1].pts;
	vector<Point2d> orgKeyPts2 = _keyList[imgIndex2].pts;
	Mat orgDescriptors1, orgDescriptors2;
	orgDescriptors1 = _keyList[imgIndex1].descriptors;
	orgDescriptors2 = _keyList[imgIndex2].descriptors;

	vector<Point2d> keyPts1, keyPts2;
	Mat descriptors1, descriptors2;

	vector<int> subSet1 = _subKeyIndexList[imgIndex1], subSet2 = _subKeyIndexList[imgIndex2];
	int realNum1 = subSet1.size(), realNum2 = subSet2.size();
	//! sample subset 1
	for (int i = 0; i < realNum1; i ++)
	{
		int no = subSet1[i];
		keyPts1.push_back(orgKeyPts1[no]);
	}
	descriptors1 = Mat(realNum1, 64, CV_32FC1);
	for (int i = 0; i < realNum1; i ++)
	{
		int no = subSet1[i];
		orgDescriptors1.row(no).copyTo(descriptors1.row(i));
	}
	//! sample subset 2
	for (int i = 0; i < realNum2; i ++)
	{
		int no = subSet2[i];
		keyPts2.push_back(orgKeyPts2[no]);
	}
	descriptors2 = Mat(realNum2, 64, CV_32FC1);
	for (int i = 0; i < realNum2; i ++)
	{
		int no = subSet2[i];
		orgDescriptors2.row(no).copyTo(descriptors2.row(i));
	}
	// Matching descriptor vectors using FLANN matcher
	vector<DMatch> m_Matches;
	FlannBasedMatcher matcher; 
	vector<vector<DMatch>> knnmatches;
	int num1 = keyPts1.size(), num2 = keyPts2.size();
	int kn = min(min(num1, num2), 5);

	try {
	    matcher.knnMatch(descriptors1, descriptors2, knnmatches, kn);   
    } catch(std::exception const& e){
    	std::cout<<"Exception: "<< e.what()<<std::endl;
	}
	double minimaDsit = 99999;
	for (int i = 0; i < knnmatches.size(); i ++)
	{
		double dist = knnmatches[i][0].distance;
		if (dist < minimaDsit)
		{
			minimaDsit = dist;
		}
	}
	double fitedThreshold = minimaDsit * 5;
	int keypointsize = knnmatches.size();
	for (int i = 0; i < keypointsize; i ++)
	{  
		const DMatch nearDist1 = knnmatches[i][0];
		const DMatch nearDist2 = knnmatches[i][1];
		double distanceRatio = nearDist1.distance / nearDist2.distance;
		if (nearDist1.distance < fitedThreshold && distanceRatio < 0.7)
		{
			m_Matches.push_back(nearDist1);
		}
	}
	int num = m_Matches.size();
	return num;
}

