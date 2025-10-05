CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Neha Thumu
  * [LinkedIn](https://www.linkedin.com/in/neha-thumu/)
* Tested on: Windows 11 Pro, i9-13900H @ 2.60GHz 32GB, Nvidia GeForce RTX 4070

<img width="1200" height="1200" alt="cornell 2025-10-04_22-03-13z 12samp" src="https://github.com/user-attachments/assets/182ed5cf-34da-4e71-bb97-55a448c3bcff" />

### Instructions to run this project

```
1. Clone the repo and cd into the directory
2. <<mkdir build>>
3. <<cmake..>> while in the build folder
4. Open up the project in Visual Studio (Tested with Visual Studio 2022)
5. Set the configuration to release mode
6. Open project properties and set Command Arguments in Debugging based on json file you wish to run (example: ../scenes/dof.json)
```

If you wish to edit the json files, an important note is the obj relative path differs between release and debug mode so the complete path can be added instead.

### Included in the Repository 

- Cube OBJ file (from here)
- Teapot OBJ file (from here)
- list the JSON files & what they do !! 

## Project Details 

This project is a CUDA based path tracer written in C++. It is capable of rendering different types of materials (diffuse, reflective, and refractive), has OBJ support, has an AI based denoiser, a depth of field lens effect and various methods of increasing performance (bounding box culling for OBJ files and Russian Roulette path termination). 

### Supported Materials
Diffuse  |  Emissive
:-------------------------:|:-------------------------:
<img width="800" height="800" alt="diffuse_white" src="https://github.com/user-attachments/assets/1e706625-46f3-423d-b5f8-57af722344b1" /> |  <img width="800" height="800" alt="cornell 2025-10-05_00-06-09z 1181samp" src="https://github.com/user-attachments/assets/91914576-ef3b-493a-abcf-90733e709bb4" />

**Diffuse**: This type of material can take on the color of it's surroundings based on how the light reflects off of it. In order to get the matte effext, when the ray intersects with the diffuse object, it is reflected randomly using a cosine-weighted scatter function. (The diagram below contains a visual explanation of how rays are reflected.)

ADD DIAGRAM!!

**Emissive**: This is an example of how light sources behave within the pathtracer. If rays hit the light source, they stop bouncing otherwise there will be an extremely saturated render (see Trials and Tribulations for a visual example :) ). 

Reflective (0.0 Roughness)  |  Reflective (0.25 Roughness)  |  Reflective (0.75 Roughness)  |  
:-------------------------:|:-------------------------:|:-------------------------:
<img width="800" height="800" alt="specular_white" src="https://github.com/user-attachments/assets/5bc638a3-12ef-45ac-b456-ccd54cab350f" /> | <img width="800" height="800" alt="specular_smooth" src="https://github.com/user-attachments/assets/ad58b4ce-6239-4110-946c-51e730b42eea" /> | <img width="800" height="800" alt="specular_rough" src="https://github.com/user-attachments/assets/083aeb57-73c7-4f4d-a8f5-31771c81f02c" />

**Reflective**: A completely reflective material (similar to the look of a shiny mirror) is created by having the ray bounce off of the surface based on the angle of the ray and the surface normal. (Or to have a more visual explanation, the reflected ray would be the (COLOR) arrow in the diagram below.) If the roughness is increased, this creates a fuzzy or blurred effect with the reflection that results in it looking more metallic.

ADD DIAGRAM!!

Refractive  |  
:-------------------------:|
<img width="400" height="400" alt="refractive" src="https://github.com/user-attachments/assets/4f7e9bf0-7cdc-46c8-b133-aef613164e9c" /> |

**Refractive**: Finally, we have the refractive or glass-like material! This utilizes Snell's law where we want to look at the angles formed based on the refracted ray and the normal (like in the diagram below). Then we use these angles along with the index of refraction (dependent on the materials the ray is going through) to figure out how the ray is refracted through the material.

ADD DIAGRAM!!

### Anti-Aliasing 
Comparisons  |  
:-------------------------:|
<img width="2000" height="800" alt="aa" src="https://github.com/user-attachments/assets/f0e6cd0c-39ae-4adf-a872-04f8ffbafac4" /> |

**Anti-aliasing**: a toggle-able option within the GUI that prevents the rough edges between shapes by jittering the ray in order to sample the surrounding pixels. This sampling creates a smoother transition between shapes. In the image above, we can see that the no anti-aliasing image has clear lines that are not visible in the version with anti-aliasing.

### Depth of Field 
Comparisons  |  
:-------------------------:|
<img width="400" height="400" alt="cornell 2025-10-05_00-49-51z 1021samp" src="https://github.com/user-attachments/assets/00979874-3bd4-4636-83ab-3f91441f7bd0" /> |

- Depth of field !!

### OBJ Loader 
This OBJ loader utilized the [TinyOBJLoader](https://github.com/tinyobjloader/tinyobjloader/tree/release) and supports triangulated OBJs. Only one obj may be loaded in at a time. The OBJ is loaded in by being added to the JSON file and the type being 'custom'. One important note is in debug mode and release mode there are different relative paths. The materials that can be applied to the default sphere and cube can be applied to the OBJ by adding to the Material field of the JSON file. 
Diffuse  | Complete Reflection  |  Partial Reflection  |
:-------------------------:|:-------------------------:|:-------------------------:|
<img width="800" height="800" alt="cornell 2025-10-05_03-41-27z 1218samp" src="https://github.com/user-attachments/assets/8777b63f-ce31-40a7-b5b7-29943ee109ae" /> | <img width="800" height="800" alt="cornell 2025-10-05_03-46-44z 1025samp" src="https://github.com/user-attachments/assets/2ffca404-67d9-475c-bdc0-11af109a299c" /> | <img width="800" height="800" alt="cornell 2025-10-05_03-51-08z 1036samp" src="https://github.com/user-attachments/assets/1d0a1a20-d280-4ce7-9e87-fb2e0168d719" /> |

Here are some examples of the teapot OBJ being loaded in with different materials.

Complete Reflection  |  Refraction  |
:-------------------------:|:-------------------------:|
<img width="800" height="800" alt="cornell 2025-10-05_04-03-20z 456samp" src="https://github.com/user-attachments/assets/f2573049-ef99-4b42-b261-8215a4f85e4c" /> | <img width="800" height="800" alt="cornell 2025-10-05_04-27-22z 408samp" src="https://github.com/user-attachments/assets/9efd9ee0-872c-4fb9-949b-eb856e709aa0" />

Here is another example of the dragon OBJ being loaded in with the complete reflection material (0.0 roughness) and the refraction material. There is slightly more noise here due to the number of iterations before the snapshot being less than above.

#### Bounding Box Culling

This is a toggle-able option that slightly increases the efficiency of the process due to not checking if the ray intersects with the triangles in the OBJ if the ray does not hit the bounding box of the object (within the min/max bounds). This is most visible in objects with greater numbers of triangles due to the current logic looping through all of the triangles to see if there is an intersection with the ray.

ADD THE COMPARISON CHART

### Denoiser 

ADD EXPLANATION OF DENOISER

Pre-denoise  |  Post-denoise  |
:-------------------------:|:-------------------------:|
<img width="800" height="800" alt="cornell 2025-10-05_00-38-26z 157samp" src="https://github.com/user-attachments/assets/b63acd67-b4e8-4c65-a87c-9c7c64da7ac3" /> | <img width="800" height="800" alt="cornell 2025-10-05_00-38-06z 161samp" src="https://github.com/user-attachments/assets/d59fe1bc-2dc5-4745-8be5-4e1740d94547" />

An example of the denoiser at work with a sphere that has a fully reflective material. If enough time passes, the majority of the noise clears up in this situation but for the sake of this example, the snapshot was taken at approximately 60 iterations.

Pre-denoise  |  Post-denoise  |
:-------------------------:|:-------------------------:|
<img width="800" height="800" alt="cornell 2025-10-02_03-18-19z 27samp" src="https://github.com/user-attachments/assets/e2dc0e4b-af33-48c7-bf22-6e632d6c846c" /> | <img width="800" height="800" alt="cornell 2025-10-02_03-25-18z 4samp" src="https://github.com/user-attachments/assets/aafc34c9-5658-41f1-b957-a478049a20e6" />

An example of the denoiser in a more likely use case. The OBJ loading takes much longer due to the computation required for checking if there is an intersection between the rays and the triangles that compose the OBJ. As such, the noise does not clear up until approximately 600 or so iterations. This snapshot was taken at only 27 iterations!

### Russian Roulette

ADD EXPLANATION

ADD CHART

## Misc cool images
Infinity Cubes  |  Blue Skybox  |  Noir Film  |
:-------------------------:|:-------------------------:|:-------------------------:|
<img width="800" height="800" alt="cornell 2025-10-03_00-33-38z 19samp" src="https://github.com/user-attachments/assets/f1dd9ce1-9bc7-4816-940a-44521ce501d7" /> | <img width="800" height="800" alt="cornell 2025-09-29_00-03-05z 102samp" src="https://github.com/user-attachments/assets/7ad3d781-9f45-4cf9-8b6c-49b2d927f5e8" /> | <img width="800" height="800" alt="cornell 2025-10-05_13-39-36z 1871samp" src="https://github.com/user-attachments/assets/a596ec36-10c9-449d-8725-2586f248622d" /> |

## Trials and Tribulations 

During the somehwat turmulous process of creating this pathtracer (a fun challenge I will say!), there have been many curious bugs. Here are some of the more interesting ones! 

Need to stream compact :)  |  Stream compact issues pt.1  |  Stream compact issues pt.2  |
:-------------------------:|:-------------------------:|:-------------------------:|
<img width="800" height="800" alt="cornell 2025-09-20_18-01-09z 46samp" src="https://github.com/user-attachments/assets/c77da368-d5c5-46e2-adc9-5b283f2debfd" /> | <img width="800" height="800" alt="cornell 2025-09-20_20-52-15z 61samp" src="https://github.com/user-attachments/assets/8278fdae-a75f-4f7c-8f70-31ee83ee6e3a" /> | <img width="800" height="800" alt="cornell 2025-09-20_22-23-09z 9samp" src="https://github.com/user-attachments/assets/b0560175-dabf-4627-812e-2604d9daa385" /> |

Banding  |  Too close  |  Scale issue  |
:-------------------------:|:-------------------------:|:-------------------------:|
<img width="800" height="800" alt="cornell 2025-09-24_12-51-09z 61samp" src="https://github.com/user-attachments/assets/0d1a819b-2810-4b03-996e-8671abc0b6a3" /> | <img width="800" height="800" alt="cornell 2025-10-03_00-59-07z 102samp" src="https://github.com/user-attachments/assets/519733d6-7958-43dc-aa83-a35b682b673d" /> | <img width="800" height="800" alt="cornell 2025-09-29_00-31-04z 4samp" src="https://github.com/user-attachments/assets/f03ce25b-8a2e-4a80-bf10-68b5272695bb" /> |

### BVH 

This is a feature I wish I was able to figure out by the deadline but alas. I was able to successfully create my tree which essentially took used the triangles that were read in during the loadObj process and put them into leaf nodes. For the sake of not having too many levels of the trees, each leaf contains about 4 or so triangles but there can be more if the depth reaches 20. The BVH class was also able to be copied onto the GPU by structuring the tree such that instead of having pointers to the children, each node held onto the indices of the children. This way the fields consisted of ints and booleans-GPU friendly! However, the main issue came to figuring out how to check if there was an intersection with a node in my tree. The idea is simple enough: check the bounding box of the bounding box that encompasses the object, then go down the tree based on where the hit was (left child or right child of the root which has the initial bounding box). This lends itself to recursion however recursion is not the best to do on the GPU. The recommended approach is turning the recursive function into something that is iterative instead. This was an intersting challenge that I unfortunately was unable to accomplish but hope to figure out in the future! From knowledge gained via the internet, it seems that a stack pointer has to be used which is intriguing. 

I did learn a very crucial thing from my trials with BVH, it is incredibly annoying to attempt debugging a recursive function in CUDA. (This may have been prior to me looking up and realizing that it is really not recommended..) I was quite surprised at how many times the debugger pointed at random lines and I had to restart the program in order to get it to work. 

## CMake Edits / Notes 

For ease of use, I edited the CMake so there is no need to manually copy the Denoise dll's into the build foldler and and the lib files into the external folder.

## Resources Utilized 
- [Raytracing in one weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [PBR textbook](https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection)
- Slides from 4610
- BVH ([1](https://www.youtube.com/watch?v=LAxHQZ8RjQ4), [2](https://15362.courses.cs.cmu.edu/fall2025/lecture/lec-08), [3](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/))
- [Stanford Dragon OBJ](http://mrbluesummers.com/3572/downloads/stanford-dragon-model/)
- [TinyOBJLoader](https://github.com/tinyobjloader/tinyobjloader/tree/release)
- [Paul Bourke's Notes](https://paulbourke.net/miscellaneous/raytracing/)
- [AI Denoiser](https://github.com/RenderKit/oidn)
