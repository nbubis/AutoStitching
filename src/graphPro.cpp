#include "graphPro.h"
#define MAX_COST 9999

vector<TreeNode> Graph::DijkstraForPath(Mat_<double> graph, int rootNo)   //rootNo��ʾԴ���� 
{
	int nodeNum = graph.rows;
	Mat_<double> dist = Mat(1, nodeNum, CV_64FC1, Scalar(0));
	Mat_<int> path = Mat(1, nodeNum, CV_16UC1, Scalar(0));
	bool *visited = new bool[nodeNum];
	for(int i = 0; i < nodeNum; i ++)     //��ʼ�� 
	{
		if(graph(rootNo,i) > 0 && i != rootNo)
		{
			dist(i) = graph(rootNo,i);
			path(i) = rootNo;     //path��¼���·���ϴ�rootNo��i��ǰһ������ 
		}
		else
		{
			dist(i) = MAX_COST;    //��i����rootNoֱ�����ڣ���Ȩֵ��Ϊ����� 
			path(i) = -1;
		}
		visited[i] = false;
		path(rootNo) = rootNo;
		dist(rootNo) = 0;
	}
	visited[rootNo] = true;
	for(int i = 1; i < nodeNum; i ++)     //ѭ����չn-1�� 
	{
		int min = MAX_COST;
		int u;
		for(int j = 0; j < nodeNum; j ++)    //Ѱ��δ����չ��Ȩֵ��С�Ķ��� 
		{
			if(visited[j] == false && dist(j) < min)
			{
				min = dist(j);
				u = j;        
			}
		} 
		visited[u] = true;
		for(int k = 0; k < nodeNum; k ++)   //����dist�����ֵ��·����ֵ 
		{
			if(visited[k] == false && graph(u,k) > 0 &&
				min + graph(u,k) < dist(k))
			{
				dist(k) = min + graph(u,k);
				path(k) = u; 
			}
		}        
	}  
	delete []visited;
	path(rootNo) = -1;   //! set the parent of root node as -1
	return traverseBreadthFirst(path, rootNo);
}


vector<TreeNode> Graph::FloydForPath(Mat_<double> graph)
{
	int nodeNum = graph.rows;
	//! initialize for dist and path
	Mat_<double> dist = Mat(nodeNum, nodeNum, CV_64FC1, Scalar(0));
	Mat_<int> path = Mat(nodeNum, nodeNum, CV_16UC1, Scalar(0));
	double *graphPtr = (double*)graph.data;
	double *distPtr = (double*)dist.data;
	int *pathPtr = (int*)path.data;
	for(int i = 0; i < nodeNum; i ++)
	{
		for(int j = 0; j < nodeNum; j ++)
		{
			if(graph(i,j) > 0)
			{
				//dist(i,j) = graph(i,j);
				//path(i,j) = i;
				distPtr[i*nodeNum+j] = graphPtr[i*nodeNum+j];
				pathPtr[i*nodeNum+j] = i;
			}
			else
			{
				if(i != j)
				{
					//dist(i,j) = MAX_COST;
					//path(i,j) = -1;
					distPtr[i*nodeNum+j] = MAX_COST;
					pathPtr[i*nodeNum+j] = -1;
				}
				else
				{
					//dist(i,j) = 0;
					//path(i,j) = i;
					distPtr[i*nodeNum+j] = 0;
					pathPtr[i*nodeNum+j] = i;
				}    
			}
		}
	}
	//! perform Floyd algorithm
	for(int k = 0; k < nodeNum; k ++)                            //�м�����(ע������kΪʲôֻ���������) 
	{
		for(int i = 0; i < nodeNum; i ++)  
		{
			for(int j = 0; j < nodeNum; j ++)
			{
				//if(dist(i,k) + dist(k,j) < dist(i,j))
				//{
				//	dist(i,j) = dist(i,k) + dist(k,j);
				//	path(i,j) = path(k,j);                      //path[i][j]��¼��i��j�����·����j��ǰһ������ 
				//}
				if (distPtr[i*nodeNum+k] + distPtr[k*nodeNum+j] < distPtr[i*nodeNum+j])
				{
					distPtr[i*nodeNum+j] = distPtr[i*nodeNum+k] + distPtr[k*nodeNum+j];
					pathPtr[i*nodeNum+j] = pathPtr[k*nodeNum+j];
				}
			}
		}
	}

	//! find optimal root node
	double leastDist = 9999;
	int rootNo = 0;
	FILE *fp = fopen("costs.txt", "w");
	for (int i = 0; i < nodeNum; i ++)
	{
		double distSum = 0;
		for (int j = 0; j < nodeNum; j ++)
		{
			distSum += dist(i,j);
		}
		fprintf(fp, "%lf\n", distSum);
		if (distSum < leastDist)
		{
			leastDist = distSum;
			rootNo = i;
		}
	}
	fclose(fp);
//	appendix(dist, rootNo);
	path.row(rootNo)(rootNo) = -1;   //! set the parent of root node as -1
	return traverseBreadthFirst(path.row(rootNo), rootNo);
}


Mat_<int> Graph::extractMSTree(Mat_<double> graph)
{
	int vNum = graph.rows;
	Mat_<int> edgeList = Mat(vNum-1, 2, CV_32SC1);
	vector<double> lowcost(vNum,0);
	vector<int> adjecent(vNum,0);
	vector<bool> s(vNum);          //! label the nodes
	double *dataPtr = (double*)graph.data;
	for (int i = 0; i < vNum*vNum; i ++)
	{
		if (dataPtr[i] < 0)           //! infinite : -1, so convert back as infinite(MAX_COST)
		{
			dataPtr[i] = MAX_COST;
		}
	}
	s[0] = true;
	for (int i = 1; i < vNum; ++ i)
	{
		lowcost[i] = graph(0,i);
		adjecent[i] = 0;
		s[i] = false;
	}

	//! searching the minimum spanning tree
	for (int i = 0; i < vNum-1; ++i)    //! for other n-1 nodes
	{
		double min = MAX_COST;
		int j = 0;                      //! new node to be added
		for (int k = 1; k < vNum; ++k)
		{
			if (lowcost[k] < min && !s[k])
			{
				min = lowcost[k];
				j = k;
			}
		}
//		cout <<"Joint"<<j<<" and "<<adjecent[j]<<endl;
		//! record this edge
		edgeList(i,0) = adjecent[j];
		edgeList(i,1) = j;
		s[j] = true;                    //! label the new added node
		//! updating
		for (int k = 1; k < vNum; ++ k)
		{
			if (graph(j,k) < lowcost[k] && !s[k])
			{
				lowcost[k] = graph(j,k);
				adjecent[k] = j;
			}
		}
	}
	return edgeList;
}


vector<TreeNode> Graph::traverseBreadthFirst(Mat_<int> path, int rootNo)
{
	int nodeNum = path.cols;
	vector<TreeNode> visitOrder;
	TreeNode bar(rootNo,-1,0);
	visitOrder.push_back(bar);
	vector<int> headList;
	headList.push_back(rootNo);
	vector<int> headers, nheaders;
	headers.push_back(rootNo);
	int level = 1;
	//! T(n) = O(n^2)
	while (1)
	{
		//! searching by levels of tree
		for (int t = 0; t < headers.size(); t ++)
		{
			int headNo = headers[t];
			//! the index of node is its node no
			for (int i = 0; i < nodeNum; i ++)
			{
				if (path(i) == headNo)    //! judge parent node
				{
					nheaders.push_back(i);
					TreeNode bar(i, headNo, level);
					visitOrder.push_back(bar);
				}
			}
		}
		if (visitOrder.size() == nodeNum || headers.size() == 0)
		{
			break;
		}

		level ++;
		headers = nheaders;
		nheaders.clear();
	}

	return visitOrder;
}

