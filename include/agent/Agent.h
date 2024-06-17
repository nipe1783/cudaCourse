#pragma once
#include <vector>
#include <Eigen/Core>

class Agent {
    public:
        // constructor
        Agent() = default;
        Agent(std::vector<Eigen::Vector2d>& verticesCCW);
        Agent(float x, float y, float length);

        // methods

        // fields
        std::vector<Eigen::Vector2d> verticesCCW_;
        
};