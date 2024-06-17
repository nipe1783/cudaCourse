#pragma once

#include "agent/Agent.h"
#include "obstacle/Obstacle.h"
#include "workspace/Workspace.h"
#include <Eigen/Core>

class Planner {
    public:
        // constructor
        Planner() = default;

        // methods
        virtual void plan(const Workspace& workspace, const Agent& agent, const Eigen::Vector2d& goal) = 0;
        std::vector<Eigen::Vector2d> generateRandomSamples();
        static __device__ Eigen::Vector2d generateRandomSampleDevice();

        // fields
        Agent agent_;
        Workspace workspace_;
        Eigen::Vector2d goal_;
        std::vector<Eigen::Vector2d> samples_;

};