//
//  loop.cpp
//  
//
//  Created by tianhaoxie on 2022
//
#include <torch/extension.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <iostream>
#include <igl/readOBJ.h>
#include <igl/loop.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "loopEvaluation/pointEvaluate.h"
#include "loopEvaluation/eigenStructure.h"
#include "loopEvaluation/pointCollect.h"
#include "loopEvaluation/limitationSurfaceComputation.h"

using namespace std;
using namespace torch::indexing;

struct loop_limitation : torch::CustomClassHolder{
    
    Eigen::MatrixXd template_V,SV,LV;
    Eigen::MatrixXi template_F,SF;
    Eigen::SparseMatrix<double> J,JS,S;
    vector<vector<int>> A;
    eigenStructure* es;
    LLPE::pointCollect* pc;
    LLPE::pointEvaluate* pe;
    LLPE::limitationSurfaceComputation* lsc;
    loop_limitation(){}
    
    void read_template(const torch::Tensor& m_v,const torch::Tensor& m_f){
        torch::Tensor v_cpu;
        torch::Tensor f_cpu;
        torch::Tensor l;
        
        v_cpu = torch::_cast_Double(m_v.cpu());
        f_cpu = torch::_cast_Int(m_f.cpu());
        if (read_mesh(v_cpu,template_V,f_cpu,template_F))
            cout<<"read template success"<<endl;
        
    
    }

    bool init_J(const torch::Tensor& m_v,const torch::Tensor& m_f){
        read_template(m_v,m_f);
        igl::loop(template_V.rows(), template_F, S, SF);
        
        igl::adjacency_list(SF,A,true);
        es = new eigenStructure();
        if (!es -> load()){
            cout<< "Error in load eigen"<<endl;
            return false;
        }
        J.resize(A.size(),A.size());
        pc = new LLPE::pointCollect(A);
        pe = new LLPE::pointEvaluate(es->ev);
        lsc = new LLPE::limitationSurfaceComputation(pc, pe, SF);
        lsc -> compute_J(J);
        
        J = (J*S).topRows(template_V.rows());
        return true;
        
    }
    
    torch::Tensor get_J(bool cuda){
        torch::Tensor jacobian=torch::zeros({template_V.rows(),template_V.rows()});
        int row,col;
        double value;
        for (int i=0;i<J.outerSize();++i){
            for (Eigen::SparseMatrix<double>::InnerIterator it(J,i);it;++it){
                row=it.row();
                col=it.col();
                value=it.value();
                jacobian.index_put_({row,col},value);
            }
        }

        if (cuda)
            return jacobian.cuda();
        else
            return jacobian;
    }
    

    torch::Tensor compute_limitation(const torch::Tensor& m_v){
        torch::Tensor v_cpu;
        torch::Tensor l;
        if (m_v.is_cuda()){
            v_cpu = torch::_cast_Double(m_v.cpu());
            read_vertices(v_cpu,template_V);
            l = torch::zeros({template_V.rows(),3},torch::TensorOptions().device(torch::kCUDA, 0));
        }
        else{
            read_vertices(torch::_cast_Double(m_v),template_V);
            l = torch::zeros({template_V.rows(),3});
        }
        
        
        SV=S*template_V;
        
        lsc->compute(SV,LV);
        for (int i=0;i<template_V.rows();++i){
            l.index_put_({i,0},LV(i,0));
            l.index_put_({i,1},LV(i,1));
            l.index_put_({i,2},LV(i,2));
        }
        return l;
    }
    

    

    bool read_mesh(const torch::Tensor& m_v, Eigen::MatrixXd& V, const torch::Tensor& m_f,Eigen::MatrixXi& F){


        double* data_mv = m_v.data_ptr<double>();
        V.resize(m_v.size(0),m_v.size(1));

        for (int i=0;i<m_v.size(0);++i){
            V(i,0)= *(data_mv+3*i);
            V(i,1)= *(data_mv+3*i+1);
            V(i,2)= *(data_mv+3*i+2);
        }
        int* data_mf = m_f.data_ptr<int>();
        F.resize(m_f.size(0),m_f.size(1));
        for (int i=0;i<m_f.size(0);++i){
            F(i,0)= *(data_mf+3*i);
            F(i,1)= *(data_mf+3*i+1);
            F(i,2)= *(data_mf+3*i+2);
                
        }
        return true;
        
    }
    bool read_vertices(const torch::Tensor& m_v, Eigen::MatrixXd& V){
        double* data_mv = m_v.data_ptr<double>();
        V.resize(m_v.size(0),m_v.size(1));

        for (int i=0;i<m_v.size(0);++i){
            V(i,0)= *(data_mv+3*i);
            V(i,1)= *(data_mv+3*i+1);
            V(i,2)= *(data_mv+3*i+2);
        }
        return true;
    }

    
};

// TORCH_LIBRARY(loop, m) {

//     m.class_<loop_limitation>("loop_limitation")
//         .def (torch::init())
//         .def ("read_template", &loop_limitation::read_template)
//         .def ("get_J", &loop_limitation::get_J)
//         .def ("compute_limitation", &loop_limitation::compute_limitation)
//         .def ("compute_limitation_t", &loop_limitation::compute_limitation_t)
//         .def ("init_J", &loop_limitation::init_J)
//     ;

// }

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<loop_limitation>(m, "loop_limitation")
        .def (py::init())
        .def ("get_J", &loop_limitation::get_J)
        .def ("compute_limitation", &loop_limitation::compute_limitation)
        .def ("init_J", &loop_limitation::init_J)
    ;
}