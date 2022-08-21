//
//  pointEvaluate.hpp
//  
//
//  Created by xietianhao on 2021-11-29.
//

#ifndef pointEvaluate_h
#define pointEvaluate_h

#include <Eigen/Core>
#include <stdio.h>
#include "eigenStructure.h"
#include "pointCollect.h"
#include <torch/extension.h>
namespace LLPE{
class pointEvaluate{
public:
    pointEvaluate(const std::vector<eigenStructure::EVALSTRUCT>& ev);
    ~pointEvaluate();
    const std::vector<eigenStructure::EVALSTRUCT>& ev_;
    Eigen::Matrix<double, 12, 1> getb(double v, double w);
    torch::Tensor getb_tensor(double v, double w);
    double evalBasis(const Eigen::Matrix<double, 1, 12>& m, double v, double w);
    double evalBasis_tensor(const torch::Tensor& m, double v, double w);
    bool ProjectPoints(std::vector<Eigen::Vector3d>& Cp, const std::vector<Eigen::Vector3d>& C, int N);
    torch::Tensor ProjectPoints_tensor(const torch::Tensor& C, int N);
    Eigen::Vector3d evaluateRegularPatch(const Eigen::MatrixXd& V,std::vector<int>& p, Eigen::Vector3d bary);
    torch::Tensor evaluateRegularPatch_tensor(const torch::Tensor& V,std::vector<int>& p, Eigen::Vector3d bary);
    bool evalSurf(Eigen::Vector3d& Pp, const std::vector<Eigen::Vector3d>& Cp, double v, double w, int N);
    torch::Tensor evalSurf_tensor( const torch::Tensor& Cp, double v, double w, int N);
    Eigen::Vector3d evaluateIrregularPatch(const Eigen::MatrixXd& V,std::vector<int>& p, Eigen::Vector3d bary);
    torch::Tensor evaluateIrregularPatch_tensor(const torch::Tensor& V,std::vector<int>& p, Eigen::Vector3d bary);
};
}
#endif /* pointEvaluate_hpp */
