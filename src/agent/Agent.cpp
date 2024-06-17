#include "agent/Agent.h"

Agent::Agent(std::vector<Eigen::Vector2d>& verticesCCW) : verticesCCW_(verticesCCW) {}

Agent::Agent(float x, float y, float length) {
    verticesCCW_.push_back(Eigen::Vector2d(x, y));
    verticesCCW_.push_back(Eigen::Vector2d(x + length, y));
    verticesCCW_.push_back(Eigen::Vector2d(x + length, y + length));
    verticesCCW_.push_back(Eigen::Vector2d(x, y + length));
}