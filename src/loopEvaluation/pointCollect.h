//
//  loopEvaluation.hpp
//  
//
//  Created by xietianhao on 2021-11-29.
//

#ifndef point_collect_h
#define point_collect_h

#include <stdio.h>
#include <vector>
namespace LLPE {
class pointCollect{
public:
    pointCollect(const std::vector<std::vector<int>>& A);
    ~pointCollect();
    const std::vector<std::vector<int>>& A_;
    bool isRegular(const std::vector<int>& facePoints);
    bool isEdge(const std::vector<int>& facePoints);
    void computeVerticesOrder ( std::vector<int>& facePoints, int order[3], int adjacencyOrder[3]);
    void computeVerticesOrderIrregular (std::vector<int>& facePoints, int order[3], int adjacencyOrder[3]);
    void collectPointsRegular (std::vector<int>& facePoints, std::vector<int>& p );
    void collectPointsIrregular (std::vector<int>& facePoints, std::vector<int>& p);
};
}

#endif /* loopEvaluation_hpp */
