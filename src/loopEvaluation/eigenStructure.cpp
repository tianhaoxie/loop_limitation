//
//  eigenStructure.cpp
//  Eigen structure
//
//  Created by xie tianhao
//


#include <Eigen/Core>
#include "LoopData.h"
#include "eigenStructure.h"
#include <iostream>

eigenStructure::eigenStructure(){
    if (!load()){
        std::cout<< "Error in load eigen"<<std::endl;
    }
}
eigenStructure::~eigenStructure(){};
bool eigenStructure::load(){
    int Nmax = LoopSubdivisionData::Nmax;

    int index = 0;

    ev.resize(Nmax - 2);

    for (int i = 0; i < Nmax - 2; i++)
    {
        int N = i + 3;
        int K = N + 6;

        ev[i].eigenValues.resize(K);
        for (int j = 0; j < K; j++)
        {
            ev[i].eigenValues[j] = *(double*)&LoopSubdivisionData::data[index++];
        }
        ev[i].inverseEigenVectorsTransposed.resize(K, K);
        for (int l = 0; l < K; l++)
        {
            for (int j = 0; j < K; j++)
            {
                ev[i].inverseEigenVectorsTransposed(l, j) = *(double*)&LoopSubdivisionData::data[index++];
            }
        }
        for (int k = 0; k < 3; k++)
        {
            ev[i].Phi[k].resize(K, 12);
            for (int l = 0; l < 12; l++)
            {
                for (int j = 0; j < K; j++)
                {
                    // data contains Phi in row major
                    ev[i].Phi[k](j, l) = *(double*)&LoopSubdivisionData::data[index++];
                }
            }
        }

    }

    return true;
}
