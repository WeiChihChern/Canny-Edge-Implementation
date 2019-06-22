# Canny-Edge-Implementation
#### (My own implementation on Canny Edge without any acceleration library)

In Edge.h, there are two class member functions to do canny edge detection: *cannyEdge() & cannyEdge2()*

*cannyEdge2()* is an optimized version of cannyEdge() in 2D convolution process, it separates the sobel kernel into two smaller kernels for faster convolution process and reduces one for-loop.


Comparing to OpenCV's implementation *cv::Canny()*, *cannyEdge2()* runs around 1~3ms slower on a 371x637 image. Before tweaking my code, it took like 18ms.


##### cannyEdge2() profiling result (9ms) :arrow_down:

![myResult](https://user-images.githubusercontent.com/40074617/59934722-ca293880-947e-11e9-9d51-2b85b1784f00.PNG)

##### cv::Canny() profiling result (8ms) :arrow_down:

![opencvResult](https://user-images.githubusercontent.com/40074617/59934781-eb8a2480-947e-11e9-9216-1a5d10f03f22.PNG)

##### My OpenCV build has Intel IPP support :arrow_down:
![ipp](https://user-images.githubusercontent.com/40074617/59934883-27bd8500-947f-11e9-9387-45025a47a2a4.PNG)
