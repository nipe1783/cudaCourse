#pragma once

#include "planners/Planner.cuh"

class RRT : public Planner
{
    public:
        // constructor
        RRT() = default;

        // methods
        void plan(const Workspace& workspace, const Agent& agent, const Eigen::Vector2d& goal) override;
};