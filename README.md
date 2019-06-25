# Canny-Edge-Implementation
#### (My own implementation on Canny Edge without any acceleration library)

In Edge.h, there are two class member functions to do canny edge detection: *cannyEdge() & cannyEdge2()*

*cannyEdge2()* is an optimized version of cannyEdge() in 2D convolution process, it separates the sobel kernel into two smaller kernels for faster convolution process and reduces one for-loop.

Performance (CPU: 8700k at 4.4GHz): 

| Input size    |  Time (ms) (Avg. of 1000 runs)    | OpenMP Enable?  |
| ------------- |:-------------:| -----:|
| 637 x 371     |  3.647 ms     | Yes |
| 637 x 371     |  11.39 ms     |   No |
| 3840 x 2160   |  106.775 ms     |   Yes |
| 3840 x 2160   | 304.65ms      |    No |

# Update #2 (in branch LUT):

Done some optimizations in LUT, and added a fast atan approximation function

| Input size    |  Time (ms) (Avg. of 1000 runs)   | OpenMP Enable?  |
| ------------- |:-------------:| -----:|
| 637 x 371     |  2.469 ms     | Yes |
| 637 x 371     |  6.675 ms     |   No |
| 3840 x 2160   |  83.13 ms   |   Yes |
| 3840 x 2160   | 201.24ms      |    No |


**Working on:**

Use vector or array for storing different caluclate values like magnitudes, gradient and even convolution result. For a 1-D array, looping through it could be faster than two for-loops.
