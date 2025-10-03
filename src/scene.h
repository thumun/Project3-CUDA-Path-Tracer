#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    bool loadFromOBJ(const std::string& fileName, Geom & geom);

public:
    Scene(std::string filename);
    int buildBVH(std::vector<Triangle>& triangles, int start, int end, int depth);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<glm::vec2> uvs; 
    std::vector<glm::vec3> verts; 
    RenderState state;

    std::vector<BVHNode> bvhNodes;
};
