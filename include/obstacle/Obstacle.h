#pragma once

#include <vector>
#include <Eigen/Core>

class Obstacle {
    public:
        // constructor
        Obstacle() = default;
        Obstacle(std::vector<Eigen::Vector2d>& verticesCCW);
        Obstacle(float x, float y, float length);

        // methods

        // fields
        std::vector<Eigen::Vector2d> verticesCCW_;
        
};