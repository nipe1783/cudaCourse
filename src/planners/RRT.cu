#include "planners/RRT.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "obstacle/Obstacle.h"
#include "workspace/Workspace.h"
#include "agent/Agent.h"

void RRT::plan(const Workspace& workspace, const Agent& agent, const Eigen::Vector2d& goal){
    for (int i = 0; i < 100; i++){
        std::vector<Eigen::Vector2d> samples = generateRandomSamples();
        printf("Number of samples: %d\n", samples.size());
    }
    
}