#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    /*if (!outside)
    {
        normal = -normal;
    }*/

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    Triangle* tris,
    bool bboxEnabled)
{
    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    // check bounding box first 
    bool isIntersect; 

    if (bboxEnabled) {
        isIntersect = intersectionBoundingBox(rt.origin, rt.direction, geom.boundsMin, geom.boundsMax);
    }
    else {
        isIntersect = true;
    }

    if (isIntersect) {
        float closestT = 1e38f;
        int closestIdx = -1;
        glm::vec3 closestBary;

        for (int i = 0; i < geom.triCount; i++) {
            glm::vec3 bary;
            float hit = glm::intersectRayTriangle(rt.origin, rt.direction,
                tris[i + geom.triOffset].v0,
                tris[i + geom.triOffset].v1,
                tris[i + geom.triOffset].v2, bary);

            if (hit && bary.z > 0.0f && bary.z < closestT) {
                closestT = bary.z;
                closestIdx = i;
                closestBary = bary;
            }
        }

        if (closestIdx != -1) {
            glm::vec3 objspaceIntersection = getPointOnRay(rt, closestT);
            intersectionPoint = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.0f));
            glm::vec3 triNormalObj = tris[closestIdx + geom.triOffset].n;
            normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(triNormalObj, 0.0f)));
            return glm::length(rt.origin - intersectionPoint);
        }
    }

    return -1;
}

__host__ __device__
bool intersectionBoundingBox(
    glm::vec3 rayOrig,
    glm::vec3 rayDir,
    glm::vec3 minB,
    glm::vec3 maxB)
{
    float tmin = -1e38f;
    float tmax = 1e38f;

    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = rayDir[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (minB[xyz] - rayOrig[xyz]) / qdxyz;
            float t2 = (maxB[xyz] - rayOrig[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
            }
            if (tb < tmax)
            {
                tmax = tb;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        return true;
    }

    return false;
}

// returns the node or -1 if no node
__host__ __device__
int meshIntersectionTest(
    Geom geom,
    Ray r,
    BVHNode* bvh,
    int nodeIndx)
{

    BVHNode node = bvh[nodeIndx];

    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    if (!intersectionBoundingBox(rt.origin, rt.direction, node.minBounds, node.maxBounds)) {
    return -1;
    }

    if (node.isLeaf) {
        return node.myIndex;
    }

    int returnLeft = meshIntersectionTest(geom, r, bvh, node.leftChild);
    if (returnLeft == -1) {
        return meshIntersectionTest(geom, r, bvh, node.rightChild);
        }
    else {
        return returnLeft;
    }
}