#pragma once
#include <vector>
#include <Eigen/Core>
#include "obstacle/Obstacle.h"

class Workspace {
    public:
        // constructor
        Workspace() = default;
        Workspace(float width, float height);

        // methods
        int getNumObstacles();
        void addObstacle(Obstacle& obstacle);
        void inflateObstacles(float inflationRadius);


        // fields
        std::vector<Obstacle> obstacles_;
        float width_;
        float height_;
        
};