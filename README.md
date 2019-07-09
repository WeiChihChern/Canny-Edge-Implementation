# Canny-Edge-Implementation
#### (My own implementation on Canny Edge)

In Edge.h, there are two class member functions to do canny edge detection: *cannyEdge() & cannyEdge2()*

*cannyEdge2()* is an optimized version of cannyEdge() in 2D convolution process, it separates the sobel kernel into two smaller kernels for faster convolution process and reduces one for-loop.

# Latest master 6/30
- Provided a docker image: [docker pull wchern/dev:opencv410](https://cloud.docker.com/u/wchern/repository/docker/wchern/dev)
- Added Makefile (`make clean` supported) for ubuntu. Just `make` to compile.
- Added argument support for selecting benchmark parameteres (smalle or large image, number of iterations, instructions below)
- Edited `define marco` for gcc (version 7.4.0) compiler & microsoft's compiler (SIMD vectorization available on g++)
- Factorized a lot of codes
- Removed `atan()` & `sqrt(gx^2 + gy^2)` for speed boost
- Added `thread control function` to select number of threads according to image size
- vectorized some for-loops
- optimized `nonMaxSuppression()` & `hysteresis_threshold()` to reduce if statements </br>
### Update on 7/1 </br>
- Vectorized a for-loop using omp inbranch for flow control </br>
### Update on 7/1 </br>
- Revised hysteresis threshold's algorithm, using recursive DFS now with same performance

| Input size    |  Time (ms) (Avg. of 1000 runs)   | OpenMP Enable?  | Env |
| ------------- |:-------------:| -----:|----------:|
| 637 x 371     |   0.593 ms (before: 3.647 ms)     | Yes, 4 threads | gcc 7.4.0, ubuntu 18.04 (docker), -O2 optimization, omp simd   |
| 3840 x 2160   |  20.65 ms (before: 106.775 ms)  |   Yes, 8 threads |gcc 7.4.0, ubuntu 18.04 (docker), -O2 optimization, omp simd |



Parameter Usage: `./app_name -firstPara -secPara thirPara` </br>
- -firstPara = valid inputs are `-small` or `-large` </br>
- -secPara = valid input is `-iter` </br>
- -thirPara = valid input is any `positve integer` number for `iteration` </br>

![1](https://user-images.githubusercontent.com/40074617/60336360-b3c93280-99d2-11e9-92cc-212a8ee19e89.PNG)



