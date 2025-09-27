#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // want to use random in order to make sure random rays don't 
    // refract/reflect
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rndRay = u01(rng);

    if (rndRay < m.hasRefractive) {
        float cos_theta = std::fmin(glm::dot(-glm::normalize(pathSegment.ray.direction), normal), 1.0f);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
        float ior = cos_theta > 0 ? (1.0f / m.indexOfRefraction) : m.indexOfRefraction;

        //Schlick's approximation
        float r0 = (1 - ior) / (1 + ior);
        r0 = r0 * r0;
        float reflectance = r0 + (1 - r0) * std::pow((1 - cos_theta), 5);

        //total reflection
        if (rndRay < reflectance || ior * sin_theta > 1) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, ior);
        }
    }
    else if (rndRay < m.hasReflective) {
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    }
    else {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        // prevent self intersection -> move pt along normal by a tiny bit
        pathSegment.ray.origin = intersect + (normal * EPSILON);
        //pathSegment.color *= m.color;
    }
}
