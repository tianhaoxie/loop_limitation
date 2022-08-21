//
//  LoopData.h
//  
//
//  Created by xietianhao
//

#ifndef LOOP_EIGEN_STRUCTURE_h
#define LOOP_EIGEN_STRUCTURE_h

#include <Eigen/Core>
#include <vector>


class eigenStructure
{
public:
    
    struct EVALSTRUCT
        {
            Eigen::VectorXd eigenValues;
            Eigen::MatrixXd inverseEigenVectorsTransposed;
            Eigen::Matrix<double, Eigen::Dynamic, 12> Phi[3];
        };
    eigenStructure();
    ~eigenStructure();
    std::vector<EVALSTRUCT> ev;
    bool load();

};



#endif
