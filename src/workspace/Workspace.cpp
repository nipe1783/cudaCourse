#include "workspace/Workspace.h"

Workspace::Workspace(float width, float height) : width_(width), height_(height) {}

void Workspace::addObstacle(Obstacle& obstacle) {
    obstacles_.push_back(obstacle);
}

int Workspace::getNumObstacles() {
    return obstacles_.size();
}

void Workspace::inflateObstacles(float inflationRadius) {
    // TODO: Implement this method
}

