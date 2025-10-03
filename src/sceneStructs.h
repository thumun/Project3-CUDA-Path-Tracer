#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    CUSTOM
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 n;
    int materialid;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int vertCount;
    int vertOffset; 

    int uvCount;
    int uvOffset; 

    int triCount;
    int triOffset;

    // for bounding box check
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    float roughness;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensDiameter;
    float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

/*
struct BVHNode
{
    glm::vec3 minBounds;
    glm::vec3 maxBounds;
    BVHNode* left;
    BVHNode* right;
    bool isLeaf;
    std::vector<Triangle> triangles;
};
*/

struct BVHNode {
    glm::vec3 minBounds;
    glm::vec3 maxBounds;
    BVHNode* leftChild;
    BVHNode* rightChild;
    std::vector<Triangle> triangles; // Empty for internal nodes
    bool isLeaf;

    BVHNode() : leftChild(nullptr), rightChild(nullptr), isLeaf(false) {}

    // Constructor for leaf nodes
    BVHNode(const std::vector<Triangle>& tris)
        : leftChild(nullptr), rightChild(nullptr), triangles(tris), isLeaf(true)
    {
        calculateBounds();
    }

    void calculateBounds() {
        if (triangles.empty()) return;

        minBounds = glm::vec3(FLT_MAX);
        maxBounds = glm::vec3(-FLT_MAX);

        for (const auto& tri : triangles) {
            minBounds = glm::min(minBounds, tri.v0);
            minBounds = glm::min(minBounds, tri.v1);
            minBounds = glm::min(minBounds, tri.v2);
            maxBounds = glm::max(maxBounds, tri.v0);
            maxBounds = glm::max(maxBounds, tri.v1);
            maxBounds = glm::max(maxBounds, tri.v2);
        }
    }
};