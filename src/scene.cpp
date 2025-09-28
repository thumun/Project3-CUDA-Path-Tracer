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
            float roughness = p["ROUGHNESS"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f - roughness;
            newMaterial.hasRefractive = 0.0f;
        }
        else if (p["TYPE"] == "Transmissive")
        {
            const auto& col = p["RGB"];
            float roughness = p["ROUGHNESS"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f - roughness;
            newMaterial.hasRefractive = p["TRANSMISSIVITY"];
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else {
            newGeom.type = CUSTOM;
            loadFromOBJ(p["FILE"], newGeom);
        }
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
    camera.lensDiameter = cameraData["LENSDIAMETER"];

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

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

    bool success = tinyobj::LoadObj(&attrib, &shapes, &objmaterials, &warn, &err, fileName.c_str()); 

    glm::vec3 firstVert(attrib.vertices[0], attrib.vertices[1], attrib.vertices[2]);
    glm::vec3 maxBounds = firstVert;
    glm::vec3 minBounds = firstVert;

    for (int i = 0; i < attrib.vertices.size(); i+=3) {
        glm::vec3 vert = glm::vec3(attrib.vertices[i], attrib.vertices[i+1], attrib.vertices[i+2]);

        maxBounds = glm::max(maxBounds, vert);
        minBounds = glm::max(minBounds, vert);

        geom.verts.push_back(glm::vec3(vert));
    }

    /*
    for (int i = 0; i < attrib.normals.size(); i+=3) {
        geom.normals.push_back(glm::vec3(attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2]));
    }
    */

    for (int i = 0; i < attrib.texcoords.size(); i+=2) {
        geom.uvs.push_back(glm::vec2(attrib.texcoords[i], attrib.texcoords[i + 1]));
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

            geom.triangles.push_back(t);
        }
    }

    return success;
}
