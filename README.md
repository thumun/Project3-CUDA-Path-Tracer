Using late day!!

CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

Notes about debugging/work::

- first issue (inverse black white thing) -- wanted to check to see if code working after writing bsdf & launched to find crazy output; realized it wasn't due to the bsdf code but rather b/c I didn't do stream compaction (oops) 

- second issue and an annoying one to fix: stream compaction -- had a slew of issues due to not doing stream compaction properly. Initial strat was to use remove_if which removes unnecessary data based on a predicate -> in my case, if 0 then don't need. This was a bad idea b/c then every ray that hits the light source get optimized away. I tried a weird idea of trying to keep track of rays that hit the light source but that was complicated and did not work out. Ended up restructuring to sorting the array based on a predicate (> 0). And num_paths based on part of partition that had non-light source data//data that I did not want to process. Had another bug where everything was blindingly bright! fixed this by realizing the final gather needed all the paths not just num paths.
