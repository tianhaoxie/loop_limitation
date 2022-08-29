//
//  limitationSurfaceComputation.h
//  
//
//  Created by xietianhao on 2021-11-29.
//

#ifndef limitationSurfaceComputation_h
#define limitationSurfaceComputation_h

#include <stdio.h>
#include "pointEvaluate.h"
#include "pointCollect.h"
#include "eigenStructure.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iomanip>
namespace LLPE {
class limitationSurfaceComputation{
public:
    limitationSurfaceComputation(pointCollect* pc, pointEvaluate* pe,const Eigen::MatrixXi& F);
    ~limitationSurfaceComputation();
    pointCollect* pc_;
    pointEvaluate* pe_;
    const Eigen::MatrixXi& F_;
    void compute_J( Eigen::SparseMatrix<double>& J);
    void compute(const Eigen::MatrixXd& V,Eigen::MatrixXd& result);
};
}
#endif /* limitationSurfaceComputation_hpp */
