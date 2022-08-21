//
//  pointEvaluate.cpp
//  
//
//  Created by xietianhao on 2021-11-29.
//

#include "pointEvaluate.h"

#include <iostream>
namespace LLPE{

pointEvaluate::pointEvaluate(const std::vector<eigenStructure::EVALSTRUCT>& ev):ev_(ev){}
pointEvaluate::~pointEvaluate(){}
Eigen::Matrix<double, 12, 1> pointEvaluate::getb(double v, double w){
    Eigen::Matrix<double, 12, 1> b;
    
    double u = 1 - v - w;
    
    b(0,0) = u*u*u*u + 2*u*u*u*v;
    b(1,0) = u*u*u*u + 2*u*u*u*w;
    b(2,0) = u*u*u*u + 2*u*u*u*w + 6*u*u*u*v + 6*u*u*v*w + 12*u*u*v*v + 6*u*v*v*w + 6*u*v*v*v + 2*v*v*v*w + v*v*v*v;
    b(3,0) = 6*u*u*u*u + 24*u*u*u*w + 24*u*u*w*w + 8*u*w*w*w + w*w*w*w + 24*u*u*u*v + 60*u*u*v*w + 36*u*v*w*w +
    6*v*w*w*w + 24*u*u*v*v + 36*u*v*v*w + 12*v*v*w*w + 8*u*v*v*v + 6*v*v*v*w + v*v*v*v;
    b(4,0) = u*u*u*u + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + w*w*w*w + 2*u*u*u*v + 6*u*u*v*w + 6*u*v*w*w + 2*v*w*w*w;
    b(5,0) = 2*u*v*v*v + v*v*v*v;
    b(6,0) = u*u*u*u + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + w*w*w*w + 8*u*u*u*v + 36*u*u*v*w +
    36*u*v*w*w + 8*v*w*w*w + 24*u*u*v*v + 60*u*v*v*w + 24*v*v*w*w + 24*u*v*v*v + 24*v*v*v*w + 6*v*v*v*v;
    b(7,0) = u*u*u*u + 8*u*u*u*w + 24*u*u*w*w + 24*u*w*w*w + 6*w*w*w*w + 6*u*u*u*v + 36*u*u*v*w + 60*u*v*w*w +
    24*v*w*w*w + 12*u*u*v*v + 36*u*v*v*w + 24*v*v*w*w + 6*u*v*v*v + 8*v*v*v*w + v*v*v*v;
    b(8,0) = 2*u*w*w*w + w*w*w*w;
    b(9,0) = 2*v*v*v*w + v*v*v*v;
    b(10, 0) = 2*u*w*w*w + w*w*w*w + 6*u*v*w*w + 6*v*w*w*w + 6*u*v*v*w + 12*v*v*w*w + 2*u*v*v*v + 6*v*v*v*w + v*v*v*v;
    b(11, 0) = w*w*w*w + 2*v*w*w*w;
    
    b = (1.0 / 12.0) * b;
    
    return b;
}

torch::Tensor pointEvaluate::getb_tensor(double v, double w){
    torch::Tensor b = torch::zeros(12,torch::TensorOptions().device(torch::kCUDA, 0));
    double u = 1 - v - w;
    b[0] = u*u*u*u + 2*u*u*u*v;
    b[1] = u*u*u*u + 2*u*u*u*w;
    b[2] = u*u*u*u + 2*u*u*u*w + 6*u*u*u*v + 6*u*u*v*w + 12*u*u*v*v + 6*u*v*v*w + 6*u*v*v*v + 2*v*v*v*w + v*v*v*v;
    b[3] = 6*u*u*u*u + 24*u*u*u*w + 24*u*u*w*w + 8*u*w*w*w + w*w*w*w + 24*u*u*u*v + 60*u*u*v*w + 36*u*v*w*w +
    6*v*w*w*w + 24*u*u*v*v + 36*u*v*v*w + 12*v*v*w*w + 8*u*v*v*v + 6*v*v*v*w + v*v*v*v;
    b[4] = u*u*u*u + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + w*w*w*w + 2*u*u*u*v + 6*u*u*v*w + 6*u*v*w*w + 2*v*w*w*w;
    b[5] = 2*u*v*v*v + v*v*v*v;
    b[6] = u*u*u*u + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + w*w*w*w + 8*u*u*u*v + 36*u*u*v*w +
    36*u*v*w*w + 8*v*w*w*w + 24*u*u*v*v + 60*u*v*v*w + 24*v*v*w*w + 24*u*v*v*v + 24*v*v*v*w + 6*v*v*v*v;
    b[7] = u*u*u*u + 8*u*u*u*w + 24*u*u*w*w + 24*u*w*w*w + 6*w*w*w*w + 6*u*u*u*v + 36*u*u*v*w + 60*u*v*w*w +
    24*v*w*w*w + 12*u*u*v*v + 36*u*v*v*w + 24*v*v*w*w + 6*u*v*v*v + 8*v*v*v*w + v*v*v*v;
    b[8] = 2*u*w*w*w + w*w*w*w;
    b[9] = 2*v*v*v*w + v*v*v*v;
    b[10] = 2*u*w*w*w + w*w*w*w + 6*u*v*w*w + 6*v*w*w*w + 6*u*v*v*w + 12*v*v*w*w + 2*u*v*v*v + 6*v*v*v*w + v*v*v*v;
    b[11] = w*w*w*w + 2*v*w*w*w;
    
    b = (1.0 / 12.0) * b;
    
    return b;


}

double pointEvaluate::evalBasis(const Eigen::Matrix<double, 1, 12>& m, double v, double w){
    Eigen::Matrix<double, 12, 1> b = getb(v, w);
    double ret = m * b;
    return ret;
}

double pointEvaluate::evalBasis_tensor(const torch::Tensor& m, double v, double w){
    torch::Tensor b = getb_tensor(v,w);
    torch::Tensor ret = torch::matmul(m,b);
    return ret[0].item<double>();
}

bool pointEvaluate::ProjectPoints(std::vector<Eigen::Vector3d>& Cp, const std::vector<Eigen::Vector3d>& C, int N){
    Cp.resize(N+6);
    double temp;
    if(C.size()!=N+6 || Cp.size()!=N+6 )
        return false;
    
    for(int i=0;i<N+6;++i){
        for(int j=0;j<3;++j)
            Cp[i][j] = 0;
        
        for(int j=0;j<N+6;++j){
            temp=ev_[N-3].inverseEigenVectorsTransposed(j,i);
            Cp[i]+= temp * C[j];
        }
    }
    return true;
}

torch::Tensor pointEvaluate::ProjectPoints_tensor(const torch::Tensor& C, int N){

    torch::Tensor Cp = torch::zeros({N+6,3},torch::TensorOptions().device(torch::kCUDA, 0));

    double temp;
    
    for(int i=0;i<N+6;++i){
        for(int j=0;j<N+6;++j){
            temp=ev_[N-3].inverseEigenVectorsTransposed(j,i);
            Cp[i]+= temp * C[j];
        }
    }
    return Cp;
}

Eigen::Vector3d pointEvaluate::evaluateRegularPatch(const Eigen::MatrixXd& V,std::vector<int>& p, Eigen::Vector3d bary){
    Eigen::Vector3d finalr;
    if (p.size() != 12){
        std::cout << "not regular patch" << std::endl;
        return finalr;
    }
    std::vector<Eigen::Vector3d> C;
    Eigen::Vector3d temp;
    
    for (int i=0;i<12;++i){
        temp(0)=V(p[i],0);
        temp(1)=V(p[i],1);
        temp(2)=V(p[i],2);
        C.push_back(temp);
        
    }
    Eigen::Matrix<double, 3, 12> M;
    for(int i=0;i<3;++i){
        for(int j=0;j<12;++j){
            M(i,j) = C[j][i];
        }
    }
    double v= bary[1];
    double w= bary[2];
    Eigen::Matrix<double, 12, 1> b = getb(v, w);
    finalr = M * b;
    return finalr;
}

torch::Tensor pointEvaluate::evaluateRegularPatch_tensor(const torch::Tensor& V,std::vector<int>& p, Eigen::Vector3d bary){
    torch::Tensor finalr;
    
    torch::Tensor C = torch::zeros({12,3},torch::TensorOptions().device(torch::kCUDA, 0));
    
    for (int i=0;i<12;++i){
        C.index_put_({i,0},V.index({p[i],0}));
        C.index_put_({i,1},V.index({p[i],1}));
        C.index_put_({i,2},V.index({p[i],2}));
    }
    torch::Tensor M = torch::transpose(C,1,0);
    double v= bary[1];
    double w= bary[2];
    torch::Tensor b = getb_tensor(v, w);
    finalr = torch::matmul(M,b);
    return finalr;
}

bool pointEvaluate::evalSurf(Eigen::Vector3d& Pp, const std::vector<Eigen::Vector3d>& Cp, double v, double w, int N){
    if(Cp.size()!=N+6)
        return false;
    
    int m = floor(1 - log2(v+w));
    int p2 = pow(2, m-1);
    v*=p2;
    w*=p2;
    int k = 0;
    if(v>0.5){
        k=0;
        v = 2 * v-1;
        w = 2 * w;
    } else if(w>0.5){
        k=2;
        v=2*v;
        w=2*w-1;
    } else {
        k=1;
        v = 1 - 2*v;
        w = 1 - 2*w;
    }
    
    Pp << 0,0,0;
    
    for(int i=0;i<N+6;++i){
        Eigen::Matrix<double, 1, 12> ma;
        for(int c=0;c<12;++c)
            ma(0,c) = ev_[N-3].Phi[k](i,c);
        
        double e = pow(ev_[N-3].eigenValues[i], m-1) * evalBasis(ma, v, w);
       
        Pp+= e * Cp[i];
    }
    
   return true;
}

torch::Tensor pointEvaluate::evalSurf_tensor( const torch::Tensor& Cp, double v, double w, int N){

    int m = floor(1 - log2(v+w));
    int p2 = pow(2, m-1);
    v*=p2;
    w*=p2;
    int k = 0;
    if(v>0.5){
        k=0;
        v = 2 * v-1;
        w = 2 * w;
    } else if(w>0.5){
        k=2;
        v=2*v;
        w=2*w-1;
    } else {
        k=1;
        v = 1 - 2*v;
        w = 1 - 2*w;
    }
    
    torch::Tensor Pp = torch::zeros(3,torch::TensorOptions().device(torch::kCUDA, 0));
    
    for(int i=0;i<N+6;++i){
        Eigen::Matrix<double, 1, 12> ma;
        for(int c=0;c<12;++c)
            ma(0,c) = ev_[N-3].Phi[k](i,c);
        
        double e = pow(ev_[N-3].eigenValues[i], m-1) * evalBasis(ma, v, w);
       
        Pp+= e * Cp[i];
    }
    
   return Pp;
}

Eigen::Vector3d pointEvaluate::evaluateIrregularPatch(const Eigen::MatrixXd& V,std::vector<int>& p, Eigen::Vector3d bary){

    std::vector<Eigen::Vector3d> C;
    Eigen::Vector3d temp;
    Eigen::Vector3d Pp;
    int N=p.size()-6;
    for (int i=0;i<N+6;++i){
        temp(0)=V(p[i],0);
        temp(1)=V(p[i],1);
        temp(2)=V(p[i],2);
        C.push_back(temp);
        
    }
    double v=bary(1);
    double w=bary(2);
    std::vector<Eigen::Vector3d> Cp;
    if(!ProjectPoints(Cp, C, N)){
        std::cout<<"Error in project points"<<std::endl;
    }
    
    if (!evalSurf(Pp,Cp,v,w,N)){
        std::cout<<"Error in evaluate surf"<<std::endl;
    }
    return Pp;
}


torch::Tensor pointEvaluate::evaluateIrregularPatch_tensor(const torch::Tensor& V, std::vector<int>& p, Eigen::Vector3d bary){

    
    torch::Tensor Pp = torch::zeros(3,torch::TensorOptions().device(torch::kCUDA, 0));
    int N=p.size()-6;
    torch::Tensor C = torch::zeros({N+6,3},torch::TensorOptions().device(torch::kCUDA, 0));
    for (int i=0;i<N+6;++i){
        C.index_put_({i,0},V.index({p[i],0}));
        C.index_put_({i,1},V.index({p[i],1}));
        C.index_put_({i,2},V.index({p[i],2})); 
    }
    double v= bary(1);
    double w= bary(2);
    torch::Tensor Cp = ProjectPoints_tensor(C, N);
    
    
    Pp = evalSurf_tensor(Cp,v,w,N);
    return Pp;
}
}


