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

This project is a CUDA based path tracer that is capable of the following: render different materials (namely diffuse and transmissive), load OBJ models (it can be a bit slow with larger models as a warning), use an AI based denoiser, have a depth of field effect and provide different ways to increase performance. 

### Diffuse Material 

<img width="800" height="800" alt="diffuse_white" src="https://github.com/user-attachments/assets/1e706625-46f3-423d-b5f8-57af722344b1" />

<img width="800" height="800" alt="cornell 2025-10-05_00-06-09z 1181samp" src="https://github.com/user-attachments/assets/91914576-ef3b-493a-abcf-90733e709bb4" />

### Reflection & Refraction
<img width="800" height="800" alt="specular_white" src="https://github.com/user-attachments/assets/5bc638a3-12ef-45ac-b456-ccd54cab350f" />
<img width="800" height="800" alt="specular_smooth" src="https://github.com/user-attachments/assets/ad58b4ce-6239-4110-946c-51e730b42eea" />
<img width="800" height="800" alt="specular_rough" src="https://github.com/user-attachments/assets/083aeb57-73c7-4f4d-a8f5-31771c81f02c" />
<img width="800" height="800" alt="refractive" src="https://github.com/user-attachments/assets/4f7e9bf0-7cdc-46c8-b133-aef613164e9c" />

### Anti-Aliasing 
<img width="800" height="800" alt="cornell 2025-10-05_00-22-14z 499samp" src="https://github.com/user-attachments/assets/1764c465-62d4-4cc2-9fa2-023384c912a8" />
<img width="800" height="800" alt="cornell 2025-10-05_00-26-46z 494samp" src="https://github.com/user-attachments/assets/a21640ec-e70f-4a20-8101-afd64d28ab2c" />

### Depth of Field 
<img width="800" height="800" alt="cornell 2025-10-05_00-49-51z 1021samp" src="https://github.com/user-attachments/assets/00979874-3bd4-4636-83ab-3f91441f7bd0" />

### OBJ Loader 
<img width="800" height="800" alt="cornell 2025-10-05_03-41-27z 1218samp" src="https://github.com/user-attachments/assets/8777b63f-ce31-40a7-b5b7-29943ee109ae" />
<img width="800" height="800" alt="cornell 2025-10-05_03-46-44z 1025samp" src="https://github.com/user-attachments/assets/2ffca404-67d9-475c-bdc0-11af109a299c" />
<img width="800" height="800" alt="cornell 2025-10-05_03-51-08z 1036samp" src="https://github.com/user-attachments/assets/1d0a1a20-d280-4ce7-9e87-fb2e0168d719" />

#### Bounding Box Culling

### Denoiser 
<img width="800" height="800" alt="cornell 2025-10-05_00-38-26z 157samp" src="https://github.com/user-attachments/assets/b63acd67-b4e8-4c65-a87c-9c7c64da7ac3" />
<img width="800" height="800" alt="cornell 2025-10-05_00-38-06z 161samp" src="https://github.com/user-attachments/assets/d59fe1bc-2dc5-4745-8be5-4e1740d94547" />

<img width="800" height="800" alt="cornell 2025-10-02_03-18-19z 27samp" src="https://github.com/user-attachments/assets/e2dc0e4b-af33-48c7-bf22-6e632d6c846c" />
<img width="800" height="800" alt="cornell 2025-10-02_03-25-18z 4samp" src="https://github.com/user-attachments/assets/aafc34c9-5658-41f1-b957-a478049a20e6" />

### Russian Roulette

## Trials and Tribulations 

During the somehwat turmulous process of creating this pathtracer (a fun challenge I will say!), there have been many curious bugs. Here are some of the more interesting ones! 

Notes about debugging/work::

- first issue (inverse black white thing) -- wanted to check to see if code working after writing bsdf & launched to find crazy output; realized it wasn't due to the bsdf code but rather b/c I didn't do stream compaction (oops) 

- second issue and an annoying one to fix: stream compaction -- had a slew of issues due to not doing stream compaction properly. Initial strat was to use remove_if which removes unnecessary data based on a predicate -> in my case, if 0 then don't need. This was a bad idea b/c then every ray that hits the light source get optimized away. I tried a weird idea of trying to keep track of rays that hit the light source but that was complicated and did not work out. Ended up restructuring to sorting the array based on a predicate (> 0). And num_paths based on part of partition that had non-light source data//data that I did not want to process. Had another bug where everything was blindingly bright! fixed this by realizing the final gather needed all the paths not just num paths.

### BVH 

This is a feature I wish I was able to figure out by the deadline but alas. I was able to successfully create my tree which essentially took used the triangles that were read in during the loadObj process and put them into leaf nodes. For the sake of not having too many levels of the trees, each leaf contains about 4 or so triangles but there can be more if the depth reaches 20. The BVH class was also able to be copied onto the GPU by structuring the tree such that instead of having pointers to the children, each node held onto the indices of the children. This way the fields consisted of ints and booleans-GPU friendly! However, the main issue came to figuring out how to check if there was an intersection with a node in my tree. The idea is simple enough: check the bounding box of the bounding box that encompasses the object, then go down the tree based on where the hit was (left child or right child of the root which has the initial bounding box). This lends itself to recursion however recursion is not the best to do on the GPU. The recommended approach is turning the recursive function into something that is iterative instead. This was an intersting challenge that I unfortunately was unable to accomplish but hope to figure out in the future! From knowledge gained via the internet, it seems that a stack pointer has to be used which is intriguing. 

I did learn a very crucial thing from my trials with BVH, it is incredibly annoying to attempt debugging a recursive function in CUDA. (This may have been prior to me looking up and realizing that it is really not recommended..) I was quite surprised at how many times the debugger pointed at random lines and I had to restart the program in order to get it to work. 

## CMake Edits / Notes 

For ease of use, I edited the CMake so there is no need to manually copy the Denoise dll's into the build foldler and and the lib files into the external folder.

## Resources Utilized 
- [Raytracing in one weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- PBR textbook
- Slides from 4610
- BVH stuff
- find links in code !!
- Stanford Dragon OBJ
