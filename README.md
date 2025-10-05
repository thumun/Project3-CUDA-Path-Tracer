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
