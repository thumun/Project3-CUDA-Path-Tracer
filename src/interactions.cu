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

    if (m.hasRefractive > 0.0f) {
        glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);
        float cos_theta = -glm::dot(incoming, normal);
        float eta = cos_theta < 0 ? (m.indexOfRefraction/1.0f) : (1.0f / m.indexOfRefraction);
        
        // if the ray is inside the surface, invert the normal
        if (cos_theta < 0) {
            normal = -normal;
            cos_theta = -cos_theta;
        }

        glm::vec3 reflect_direction = glm::reflect(incoming, normal);
        glm::vec3 refract_direction = glm::refract(incoming, normal, eta);

        //Schlick's approximation
		float r0 = (eta - 1.0f)/(eta + 1.0f);
        r0 = r0 * r0;
        float reflectance = glm::clamp((float)(r0 + (1 - r0) * powf((1 - cos_theta), 5.0f)), 0.0f, 1.0f);

        thrust::uniform_real_distribution<float> u01(0, 1);
        float sample = u01(rng);
        if (sample < reflectance || (glm::length(refract_direction) == 0.0f)) {

            pathSegment.ray.direction = glm::normalize(reflect_direction);
            pathSegment.color *= m.specular.color;
        }
        else {
            pathSegment.ray.direction = glm::normalize(refract_direction);
            // adjust color based on the ratio of the indices of refraction
            pathSegment.color *= m.color;
        }        
    }
	else if (m.hasReflective > 0.0f)
    {
        glm::vec3 reflect_direction = glm::reflect(pathSegment.ray.direction, normal);
        glm::vec3 random_direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.direction = glm::normalize(glm::mix(reflect_direction, random_direction, m.roughness));	

		pathSegment.color *= m.specular.color;
    }
    else {
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.color *= m.color;
    }
    
    pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
}
