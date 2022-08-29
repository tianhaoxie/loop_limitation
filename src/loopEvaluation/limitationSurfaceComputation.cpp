//
//  limitationSurfaceComputation.cpp
//  
//
//  Created by xietianhao on 2021-11-29.
//

#include "limitationSurfaceComputation.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>
using namespace std;
namespace LLPE{
limitationSurfaceComputation::limitationSurfaceComputation(pointCollect* pc, pointEvaluate* pe,const Eigen::MatrixXi& F):pc_(pc),pe_(pe),F_(F){}
limitationSurfaceComputation::~limitationSurfaceComputation(){}

//compute jacobians
void limitationSurfaceComputation::compute_J(Eigen::SparseMatrix<double>& J){
    Eigen::VectorXi completeFlag = Eigen::VectorXi::Zero(J.rows());
    std::vector<int> facePoints(F_.cols());
    Eigen::Vector3d bary[3];
    bary[0] << 1,0,0;
    bary[1] << 0,1,0;
    bary[2] << 0,0,1;
    Eigen::Vector3d temp;
    int N,K;
    Eigen::Vector3d deriv;
    for (int i =0; i<F_.rows();++i){
        std::vector<int> collectedPoints;
        Eigen::Map<Eigen::RowVectorXi>(&facePoints[0], 1, F_.cols()) = F_.row(i);
        if ( pc_ -> isRegular(facePoints)){
            pc_ -> collectPointsRegular(facePoints,collectedPoints);
            //collectedPatch.push_back(collectedPoints);
            for (int j =0;j<3; ++j){
                if (completeFlag[facePoints[j]]==1)
                    continue;
                else{
                    N = facePoints[j];
                    //compute jacobian matrix
                    Eigen::Matrix<double, 12, 1> b;
                    b=pe_ -> getb(bary[j][1],bary[j][2]);
                    for (int k=0;k<collectedPoints.size();++k){
                        K=collectedPoints[k];
                        double d=b(k);
                        J.insert(N,K)=d;
                    }
                    completeFlag[facePoints[j]]=1;
                }
            }
        }
        else{
            pc_ -> collectPointsIrregular(facePoints,collectedPoints);
            //collectedPatch.push_back(collectedPoints);
            if (pc_ -> isEdge(facePoints)){
                continue;
            }
            std::vector<Eigen::Vector3d> C;
            std::vector<Eigen::Vector3d> Cp;
            C.resize(collectedPoints.size());
            for (int j=0;j<C.size();++j){
                C[j]<<0,0,0;
            }
            for (int j =0;j<3; ++j){
                if (completeFlag[facePoints[j]]==1)
                    continue;
                else{
                    N = facePoints[j];
                    // compute jacobian matrix
                    for (int k=0;k<collectedPoints.size();++k){
                        K=collectedPoints[k];
                        C[k][0]=1;
                        C[k][1]=1;
                        C[k][2]=1;
                        pe_ -> ProjectPoints(Cp, C, collectedPoints.size()-6);
                        pe_ -> evalSurf(deriv,Cp,bary[j][1],bary[j][2],collectedPoints.size()-6);
                        J.insert(N,K)=deriv[0];
                        C[k][0]=0;
                        C[k][1]=0;
                        C[k][2]=0;
                    }
                    completeFlag[facePoints[j]]=1;
                }
            }
        }
    }
    }

//compute limited positions
void limitationSurfaceComputation::compute(const Eigen::MatrixXd& V,Eigen::MatrixXd& result){
    result.resize(V.rows(),V.cols());
    Eigen::VectorXi completeFlag = Eigen::VectorXi::Zero(V.rows());
    std::vector<int> facePoints(F_.cols());
    Eigen::Vector3d bary[3];
    bary[0] << 1,0,0;
    bary[1] << 0,1,0;
    bary[2] << 0,0,1;
    Eigen::Vector3d temp;
    
    for (int i =0; i<F_.rows();++i){

        std::vector<int> collectedPoints;
        Eigen::Map<Eigen::RowVectorXi>(&facePoints[0], 1, F_.cols()) = F_.row(i);
        if ( pc_ -> isRegular(facePoints)){
            pc_ -> collectPointsRegular(facePoints,collectedPoints);
            for (int j =0;j<3; ++j){
                if (completeFlag[facePoints[j]]==1)
                    continue;
                else{

                    temp= pe_ -> evaluateRegularPatch(V,collectedPoints,  bary[j] );
                    result(facePoints[j],0)=temp(0);
                    result(facePoints[j],1)=temp(1);
                    result(facePoints[j],2)=temp(2);
                    completeFlag[facePoints[j]]=1;
                }
            }
        }
        else{
            
            if (pc_ -> isEdge(facePoints)){
                for (int j=0;j<3;++j){
                    result(facePoints[j],0)=V(facePoints[j],0);
                    result(facePoints[j],1)=V(facePoints[j],1);
                    result(facePoints[j],2)=V(facePoints[j],2);
                    completeFlag[facePoints[j]]=1;
                }
                continue;
            }
            pc_ -> collectPointsIrregular(facePoints,collectedPoints);
            
            for (int j =0;j<3; ++j){
                if (completeFlag[facePoints[j]]==1)
                    continue;
                else{
                    
                    temp= pe_ -> evaluateIrregularPatch(V,collectedPoints,  bary[j]);
                    result(facePoints[j],0)=temp(0);
                    result(facePoints[j],1)=temp(1);
                    result(facePoints[j],2)=temp(2);
                    completeFlag[facePoints[j]]=1;
                }
            }
        }
    }
    }
    
    
}

