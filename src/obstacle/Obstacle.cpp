#include "obstacle/Obstacle.h"

Obstacle::Obstacle(std::vector<Eigen::Vector2d>& verticesCCW) : verticesCCW_(verticesCCW) {}

Obstacle::Obstacle(float x, float y, float length) {
    verticesCCW_.push_back(Eigen::Vector2d(x, y));
    verticesCCW_.push_back(Eigen::Vector2d(x + length, y));
    verticesCCW_.push_back(Eigen::Vector2d(x + length, y + length));
    verticesCCW_.push_back(Eigen::Vector2d(x, y + length));
}