CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Neha Thumu
  * [LinkedIn](https://www.linkedin.com/in/neha-thumu/)
* Tested on: Windows 11 Pro, i9-13900H @ 2.60GHz 32GB, Nvidia GeForce RTX 4070

![dragonStart](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-04_22-03-13z.12samp.png?raw=true)

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
![diffuse](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/diffuse_white.png?raw=true) |  ![emissive](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-05_00-06-09z.1181samp.png?raw=true) |

**Diffuse**: This type of material can take on the color of it's surroundings based on how the light reflects off of it. In order to get the matte effext, when the ray intersects with the diffuse object, it is reflected randomly using a cosine-weighted scatter function. (The diagram below contains a visual explanation of how rays are reflected.)

ADD DIAGRAM!!

**Emissive**: This is an example of how light sources behave within the pathtracer. If rays hit the light source, they stop bouncing otherwise there will be an extremely saturated render (see Trials and Tribulations for a visual example :) ). 

Reflective (0.0 Roughness)  |  Reflective (0.25 Roughness)  |  Reflective (0.75 Roughness)  |  
:-------------------------:|:-------------------------:|:-------------------------:
![reflectZero](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/specular_white.png?raw=true) | ![reflectQuarter](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/specular_smooth.png?raw=true) | ![reflectThreeQuarters](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/specular_rough.png?raw=true) |

**Reflective**: A completely reflective material (similar to the look of a shiny mirror) is created by having the ray bounce off of the surface based on the angle of the ray and the surface normal. (Or to have a more visual explanation, the reflected ray would be the (COLOR) arrow in the diagram below.) If the roughness is increased, this creates a fuzzy or blurred effect with the reflection that results in it looking more metallic.

ADD DIAGRAM!!

Refractive  |  
:-------------------------:|
<img width="400" height="400" alt="refractive" src="https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/refractive.png?raw=true" /> |


**Refractive**: Finally, we have the refractive or glass-like material! This utilizes Snell's law where we want to look at the angles formed based on the refracted ray and the normal (like in the diagram below). Then we use these angles along with the index of refraction (dependent on the materials the ray is going through) to figure out how the ray is refracted through the material.

ADD DIAGRAM!!

### Anti-Aliasing 
Comparisons  |  
:-------------------------:|
![aa](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/aa.png?raw=true) |

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
![teapotdiff](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-05_03-41-27z.1218samp.png?raw=true) | ![teapotreflectall](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-05_03-46-44z.1025samp.png?raw=true) | ![teapotreflect](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-05_03-51-08z.1036samp.png?raw=true) |

Here are some examples of the teapot OBJ being loaded in with different materials.

Complete Reflection  |  Refraction  |
:-------------------------:|:-------------------------:|
![dragonreflect](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-05_04-03-20z.456samp.png?raw=true) | ![dragonrefract](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-05_04-27-22z.408samp.png?raw=true) |

Here is another example of the dragon OBJ being loaded in with the complete reflection material (0.0 roughness) and the refraction material. There is slightly more noise here due to the number of iterations before the snapshot being less than above.

#### Bounding Box Culling

This is a toggle-able option that slightly increases the efficiency of the process due to not checking if the ray intersects with the triangles in the OBJ if the ray does not hit the bounding box of the object (within the min/max bounds). This is most visible in objects with greater numbers of triangles due to the current logic looping through all of the triangles to see if there is an intersection with the ray.

ADD THE COMPARISON CHART

### Denoiser 

ADD EXPLANATION OF DENOISER

Pre-denoise  |  Post-denoise  |
:-------------------------:|:-------------------------:|
![teapotpre](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-02_03-18-19z.27samp.png?raw=true) | ![teapotpost](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-02_03-25-18z.4samp.png?raw=true) |

An example of the denoiser at work with a sphere that has a fully reflective material. If enough time passes, the majority of the noise clears up in this situation but for the sake of this example, the snapshot was taken at approximately 60 iterations.

Pre-denoise  |  Post-denoise  |
:-------------------------:|:-------------------------:|
![teapotpre](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-02_03-18-19z.27samp.png?raw=true) | ![teapotpost](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-02_03-25-18z.4samp.png?raw=true) |

An example of the denoiser in a more likely use case. The OBJ loading takes much longer due to the computation required for checking if there is an intersection between the rays and the triangles that compose the OBJ. As such, the noise does not clear up until approximately 600 or so iterations. This snapshot was taken at only 27 iterations!

### Russian Roulette

ADD EXPLANATION

ADD CHART

## Misc cool images
Infinity Cubes  |  Blue Skybox  |  Noir Film  |
:-------------------------:|:-------------------------:|:-------------------------:|
![infinity](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-03_00-33-38z.19samp.png?raw=true) | ![skybox](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-09-29_00-03-05z.102samp.png?raw=true) | ![noir](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-05_13-39-36z.1871samp.png?raw=true) |

## Trials and Tribulations 

During the somewhat turmulous process of creating this pathtracer (a fun challenge I will say!), there have been many curious bugs. Here are some of the more interesting ones! 

Need to stream compact :)  |  Stream compact issues pt.1  |  Stream compact issues pt.2  |
:-------------------------:|:-------------------------:|:-------------------------:|
![blownout](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-09-20_17-59-29z.36samp.png?raw=true) | ![stream1](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-09-20_20-52-15z.61samp.png?raw=true) | ![stream2](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-09-20_22-23-09z.9samp.png?raw=true) |

Banding  |  Too close  |  Scale issue  |
:-------------------------:|:-------------------------:|:-------------------------:|
![banding](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-09-24_12-51-09z.61samp.png?raw=true) | ![close](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-10-03_00-59-07z.102samp.png?raw=true) | ![scale](https://github.com/thumun/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2025-09-29_00-31-04z.4samp.png?raw=true) |

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
- [Teapot OBJ](https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj)
- [Cube OBJ](https://gist.github.com/MaikKlein/0b6d6bb58772c13593d0a0add6004c1c)
- [TinyOBJLoader](https://github.com/tinyobjloader/tinyobjloader/tree/release)
- [Paul Bourke's Notes](https://paulbourke.net/miscellaneous/raytracing/)
- [AI Denoiser](https://github.com/RenderKit/oidn)
