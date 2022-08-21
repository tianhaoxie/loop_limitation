//
// pointCollect.cpp
//  
//
//  Created by xietianhao on 2021-11-29.
//

#include "pointCollect.h"
#include <vector>
#include <iostream>
namespace LLPE{

pointCollect::pointCollect(const std::vector<std::vector<int>>& A):A_(A){}
pointCollect::~pointCollect(){}

bool pointCollect::isRegular(const std::vector<int>& facePoints ){
    for (int i =0; i<3 ; ++i){
        if ( A_[facePoints[i]].size() != 6)
            return false;
    }
    return true;
}


bool pointCollect::isEdge(const std::vector<int>& facePoints){
    int a=0;
    for (int i=0;i<3;++i){
        if (A_[facePoints[i]].size() != 6)
            a++;
    }
    if (a>=2){
        return true;
    }

    return false;
}

void pointCollect::computeVerticesOrder ( std::vector<int>& facePoints, int order[3], int adjacencyOrder[3]){
    order[0]=facePoints[0];
    for (int i=0 ; i<A_[order[0]].size()-1; ++i){
        if (A_[order[0]][i]==facePoints[1] ){
            if( A_[order[0]][i+1]==facePoints[2]){
                order[1]=facePoints[1];
                order[2]=facePoints[2];
                adjacencyOrder[0]=i;
                
                break;
            }
            else{
                order[1]=facePoints[2];
                order[2]=facePoints[1];
                adjacencyOrder[0]=A_[order[0]].size()-1;
                break;
            }
        }
        if (A_[order[0]][i]==facePoints[2] ){
            if (A_[order[0]][i+1]==facePoints[1]){
                order[1]=facePoints[2];
                order[2]=facePoints[1];
                adjacencyOrder[0]=i;
                break;
                
            }
            else{
                order[1]=facePoints[1];
                order[2]=facePoints[2];
                adjacencyOrder[0]=A_[order[0]].size()-1;
                break;
            }
        }
    }
    for (int i=0;i<A_[order[1]].size()-1;++i){
        if (A_[order[1]][i]==order[0] || A_[order[1]][i]==order[2]){
            if (A_[order[1]][i+1] != order[0] && A_[order[1]][i+1] != order[2] ){
                adjacencyOrder[1]=A_[order[1]].size()-1;
                break;
            }
            else{
                adjacencyOrder[1]=i;
                break;
            }
            
        }
    }
    for (int i=0;i<A_[order[2]].size()-1;++i){
        if (A_[order[2]][i]==order[0] || A_[order[2]][i]==order[1]){
            if (A_[order[2]][i+1] != order[0] && A_[order[2]][i+1] != order[1] ){
                adjacencyOrder[2]=A_[order[2]].size()-1;
                break;
            }
            else{
                adjacencyOrder[2]=i;
                break;
            }
        }
     }
    }

void pointCollect::computeVerticesOrderIrregular(std::vector<int>& facePoints, int order[3], int adjacencyOrder[3]){
    for (int i=0 ; i<3 ; ++i){
        if (A_[facePoints[i]].size() != 6){
            if (i==0)
                break;
            else
            {
                int temp= facePoints[0];
                facePoints[0]=facePoints[i];
                facePoints[i]=temp;
                break;
            }
        }
    }
    computeVerticesOrder(facePoints,order,adjacencyOrder);
}

void pointCollect::collectPointsRegular (std::vector<int>& facePoints, std::vector<int>& p ){
    if ( ! isRegular( facePoints))
    {
        std::cout << "not regular patch" << std::endl;
        return;
    }
    p.clear();
    int order[3]; // facepoints in counter-clock order
    int adjacencyOrder[3]; //The first occurence of other facepoint in adjacency list in counter-clock order
    computeVerticesOrder(facePoints,order,adjacencyOrder); // compute the relative order of 12 points(counter-clock)

    for (int i=0;i<3;++i){
        facePoints[i]=order[i];

    }
    p.resize(12);
    p[0]=A_[order[0]][(adjacencyOrder[0]+4)%6];
    p[1]=A_[order[0]][(adjacencyOrder[0]+3)%6];
    p[2]=A_[order[0]][(adjacencyOrder[0]+5)%6];
    p[3]=order[0];
    p[4]=A_[order[0]][(adjacencyOrder[0]+2)%6];
    p[5]=A_[order[1]][(adjacencyOrder[1]+3)%6];
    p[6]=order[1];
    p[7]=order[2];
    p[8]=A_[order[2]][(adjacencyOrder[2]+4)%6];
    p[9]=A_[order[1]][(adjacencyOrder[1]+4)%6];
    p[10]=A_[order[2]][(adjacencyOrder[2]+2)%6];
    p[11]=A_[order[2]][(adjacencyOrder[2]+3)%6];
}

void pointCollect::collectPointsIrregular (std::vector<int>& facePoints, std::vector<int>& p){
    if ( isRegular( facePoints))
    {
        std::cout << "not irregular patch" << std::endl;
        return;
    }
    p.clear();
    int order[3];
    int adjacencyOrder[3];
    computeVerticesOrderIrregular(facePoints,order,adjacencyOrder);
    for (int i=0;i<3;++i){
        facePoints[i]=order[i];
    }
    int N =A_[order[0]].size();
    int K =  N +6 ;
    p.resize(K);
    p[0]=order[0];
    p[1]=order[1];
    for (int i=N;i>1;--i){
        p[i]=A_[order[0]][(adjacencyOrder[0]+N-i+1)%N];
    }
    p[N+1]=A_[order[2]][(adjacencyOrder[2]+2)%6];
    p[N+2]=A_[order[1]][(adjacencyOrder[1]+4)%6];
    p[N+3]=A_[order[1]][(adjacencyOrder[1]+3)%6];
    p[N+4]=A_[order[2]][(adjacencyOrder[2]+3)%6];
    p[N+5]=A_[order[2]][(adjacencyOrder[2]+4)%6];
}
}
