#define TINYOBJLOADER_IMPLEMENTATION

#include "scene.h"

#include "tiny_obj_loader.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <glm/gtx/intersect.hpp>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 0.0f;
			newMaterial.specular.color = newMaterial.color;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.specular.color = glm::vec3(1.0f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];

    bool success = true;

    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
            success = true;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            success = true;
        }
        else {
            newGeom.type = CUSTOM;
            success = loadFromOBJ(p["FILE"], newGeom);
        }

        if (success) {
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);
	if (cameraData.contains("APERTURE")) {
        camera.aperture = cameraData["APERTURE"];
	}
    else {
        camera.aperture = 0.0f;
	}

	if (cameraData.contains("FOCALDISTANCE")) {
		camera.focalDistance = cameraData["FOCALDISTANCE"];
    }
    else {
		camera.focalDistance = 5.0f;
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);



    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

bool Scene::loadFromOBJ(const std::string& fileName, Geom & geom)
{
    tinyobj::attrib_t attrib;

    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> objmaterials;
    std::string err;
    std::string warn;

    geom.vertOffset = verts.size();
    geom.uvOffset = uvs.size();
    geom.triOffset = triangles.size();

    bool success = tinyobj::LoadObj(&attrib, &shapes, &objmaterials, &warn, &err, fileName.c_str()); 

    if (!success) {
        return success;
    }

    glm::vec3 firstVert(attrib.vertices[0], attrib.vertices[1], attrib.vertices[2]);
    glm::vec3 maxBounds = firstVert;
    glm::vec3 minBounds = firstVert;

    for (int i = 0; i < attrib.vertices.size(); i+=3) {
        glm::vec3 vert = glm::vec3(attrib.vertices[i], attrib.vertices[i+1], attrib.vertices[i+2]);

        maxBounds = glm::max(maxBounds, vert);
        minBounds = glm::min(minBounds, vert);

        verts.push_back(glm::vec3(vert));
    }

    for (int i = 0; i < attrib.texcoords.size(); i+=2) {
        uvs.push_back(glm::vec2(attrib.texcoords[i], attrib.texcoords[i + 1]));
    }

    // setting up faces (triangulated)
    for (int i = 0; i < shapes.size(); i++) {
        for (int j = 0; j < shapes[i].mesh.indices.size(); j+=3) {

            tinyobj::index_t idx0 = shapes[i].mesh.indices[j];
            tinyobj::index_t idx1 = shapes[i].mesh.indices[j + 1];
            tinyobj::index_t idx2 = shapes[i].mesh.indices[j + 2];

            Triangle t{};
            t.v0 = glm::vec3(attrib.vertices[3 * idx0.vertex_index],
                             attrib.vertices[3 * idx0.vertex_index + 1],
                             attrib.vertices[3 * idx0.vertex_index + 2]);

            t.v1 = glm::vec3(attrib.vertices[3 * idx1.vertex_index],
                             attrib.vertices[3 * idx1.vertex_index + 1],
                             attrib.vertices[3 * idx1.vertex_index + 2]);

            t.v2 = glm::vec3(attrib.vertices[3 * idx2.vertex_index],
                             attrib.vertices[3 * idx2.vertex_index + 1],
                             attrib.vertices[3 * idx2.vertex_index + 2]);

            // cross prod to get normal
            t.n = glm::normalize(glm::cross(t.v1-t.v0, t.v2 - t.v1));

            // might need to change for uv's!!
            // can use shape.material_ids
            t.materialid = geom.materialid;

            triangles.push_back(t);
        }
    }

    geom.vertCount = verts.size();
    geom.uvCount = uvs.size();
    geom.triCount = triangles.size();
    geom.boundsMin = minBounds;
    geom.boundsMax = maxBounds;

    // setting up bvh
    std::vector<Triangle> geomTriangles;
    for (int i = geom.triOffset; i < geom.triOffset + geom.triCount; i++) {
        geomTriangles.push_back(triangles[i]);
    }
    // sort triangles now so can skip that in Bvh
    glm::vec3 boundsSize = geom.boundsMax - geom.boundsMin;
    int axis = (boundsSize.x >= boundsSize.y && boundsSize.x >= boundsSize.z) ? 0 :
        (boundsSize.y >= boundsSize.z) ? 1 : 2;

    // Sort by centroid along longest axis
    /*std::vector<glm::vec3> centroids;
    for (auto const& tri: geomTriangles)
    {
        centroids.push_back(glm::vec3((tri.v0 + tri.v1 + tri.v2) / 3.0f));
    }*/
    auto centroid = [](const Triangle& tri) {
        return (tri.v0 + tri.v1 + tri.v2) / 3.0f;
    };

    std::sort(geomTriangles.begin(), geomTriangles.begin() + geom.triCount,
        [axis, centroid](const Triangle& a, const Triangle& b) {
            return centroid(a)[axis] < centroid(b)[axis];
        });

    // updating og w/ sort -- do I need two diff arrays???
    // ok if mult obj support but don't have that (?)
    for (int i = 0; i < geomTriangles.size(); i++) {
        triangles[geom.triOffset + i] = geomTriangles[i];
    }

    /*BVHNode& root = BVHNode();
    bvhNodes.push_back(root);*/

    int rootIndex = buildBVH(geomTriangles, 0, geomTriangles.size(), 0);
    bvhNodes[rootIndex].numNodes = bvhNodes.size();

    return success;
}

void calculateBounds(std::vector<Triangle>& triangles, int start, int end, glm::vec3& minBounds, glm::vec3& maxBounds) {
    if (triangles.empty()) return;

    minBounds = glm::vec3(FLT_MAX);
    maxBounds = glm::vec3(-FLT_MAX);

    for (int t = start; t < end; t++) {
        auto tri = &triangles[t];

        minBounds = glm::min(minBounds, tri->v0);
        minBounds = glm::min(minBounds, tri->v1);
        minBounds = glm::min(minBounds, tri->v2);

        maxBounds = glm::max(maxBounds, tri->v0);
        maxBounds = glm::max(maxBounds, tri->v1);
        maxBounds = glm::max(maxBounds, tri->v2);
    }
}

// this works but not currently in use
int Scene::buildBVH(std::vector<Triangle>& triangles, int start, int end, int depth) {
    // to set rt & left children of leaf

    if (start >= end) {
        return -1;
    }

    // BVHNode* current = &bvhNodes[bvhNodes.size() - 1];

    int currentNodeIndex = bvhNodes.size();
    bvhNodes.push_back(BVHNode());
    bvhNodes[currentNodeIndex].myIndex = currentNodeIndex;

    calculateBounds(triangles, start, end, bvhNodes[currentNodeIndex].minBounds, bvhNodes[currentNodeIndex].maxBounds); // setting min max bounds

    // leaf
    if (end - start <= 4 || depth > 20) {
        bvhNodes[currentNodeIndex].isLeaf = true;
        bvhNodes[currentNodeIndex].leftChild = -1;
        bvhNodes[currentNodeIndex].rightChild = -1;

        // need to get the triangles that are in this leaf chunk
        bvhNodes[currentNodeIndex].triStart = start;
        bvhNodes[currentNodeIndex].triEnd = end;

        return currentNodeIndex;
    }

    bvhNodes[currentNodeIndex].isLeaf = false;

    // Split at median
    int mid = start + (end - start) / 2;
    
    bvhNodes[currentNodeIndex].leftChild = buildBVH(triangles, start, mid, depth + 1);
    bvhNodes[currentNodeIndex].rightChild = buildBVH(triangles, mid, end, depth + 1);

    return currentNodeIndex;

}