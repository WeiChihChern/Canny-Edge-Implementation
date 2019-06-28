# Canny-Edge-Implementation
#### (My own implementation on Canny Edge)

In Edge.h, there are two class member functions to do canny edge detection: *cannyEdge() & cannyEdge2()*

*cannyEdge2()* is an optimized version of cannyEdge() in 2D convolution process, it separates the sobel kernel into two smaller kernels for faster convolution process and reduces one for-loop.

Performance (CPU: 8700k at 4.4GHz): 

| Input size    |  Time (ms) (Avg. of 1000 runs)    | OpenMP Enable?  |
| ------------- |:-------------:| -----:|
| 637 x 371     |  3.647 ms     | Yes |
| 637 x 371     |  11.39 ms     |   No |
| 3840 x 2160   |  106.775 ms     |   Yes |
| 3840 x 2160   | 304.65ms      |    No |

## Update #2 (in branch LUT):

Done some optimizations in LUT, and added a fast atan approximation function 

| Input size    |  Time (ms) (Avg. of 1000 runs)   | OpenMP Enable?  | Env |
| ------------- |:-------------:| -----:|----------:|
| 637 x 371     |  2.469 ms     | Yes | VS Studio 2015/2019 Release mode |
| 637 x 371     |  6.675 ms     |   No |VS Studio 2015/2019 Release mode |
| 3840 x 2160   |  83.13 ms   |   Yes |VS Studio 2015/2019 Release mode |
| 3840 x 2160   | 201.24ms      |    No |VS Studio 2015/2019 Release mode |


# Update #3 (in branch LUT) 6/28
1. Added Makefile (`make clean` supported) for ubuntu. Just `make` to compile.
2. Added argument support for selecting benchmark parameteres (smalle or large image, number of iterations)
3. Edited `define marco` for gcc (version 7.4.0) compiler
4. Provided a docker image: `docker pull wchern/dev:opencv410`

| Input size    |  Time (ms) (Avg. of 1000 runs)   | OpenMP Enable?  | Env |
| ------------- |:-------------:| -----:|----------:|
| 637 x 371     |   1.493 ms     | Yes, 4 threads | gcc version 7.4.0, ubuntu 18.04 (docker), -O3 optimization   |
| 637 x 371     |       |   No |gcc version 7.4.0, ubuntu 18.04 (docker), -O3 optimization |
| 3840 x 2160   |  46.927 ms   |   Yes, 4 threads |gcc version 7.4.0, ubuntu 18.04 (docker), -O3 optimization |
| 3840 x 2160   |      |    No |gcc version 7.4.0, ubuntu 18.04 (docker), -O3 optimization |

Parameter Usage: `./app_name -firstPara -secPara thirPara` </br>
-firstPara = valid inputs are `-small` or `-large` </br>
-secPara = valid input is `-iter` </br>
-thirPara = valid input is any `positve integer` number for `iteration` </br>

![1](https://user-images.githubusercontent.com/40074617/60336360-b3c93280-99d2-11e9-92cc-212a8ee19e89.PNG)


**Working on:**
1. Learning SIMD
2. To use vector or array for storing different caluclate values like magnitudes, gradient and even convolution result. 
