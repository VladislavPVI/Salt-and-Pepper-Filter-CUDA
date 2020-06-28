# Salt-and-Pepper-Filter-CUDA
Median filter is one of the well-known order-statistic filters due to its good performance for some specific noise types such as “Gaussian,” “random,” and “salt and pepper” noises. According to the median filter, the center pixel of a M × M neighborhood is replaced by the median value of the corresponding window. Note that noise pixels are considered to be very different from the median. Using this idea median filter can remove this type of noise problems . We use this filter to remove the noise pixels on the protein crystal images before binarization operation.

*Microsoft visual studio 19 +  CUDA Toolkit 11*

Build and Run
-------------

1. Install Microsoft Visual Studio.
2. Install CUDA Toolkit (Nvidea GPU with CUDA-support required).
3. Make new CUDA-project.
4. Enjoy.

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | Intel® Pentium® G860 |
| RAM  | 6 GB DDR3 |
| GPU  | GeForce GTX 750 Ti 2GB |
| OS   | Windows 10 64-bit  |

## Results

<img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/NoiseAngelina.jpg" width="480" height="300" /> | <img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/CPUoutAngelina.jpg" width="480" height="300" />
------------ | ------------- 
Distorted image (noise 8%) | Filtered image (CPU)

<img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/NoiseAngelina.jpg" width="480" height="300" /> | <img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/GPUoutAngelina.jpg" width="480" height="300" />
------------ | ------------- 
Distorted image (noise 8%) | Filtered image (GPU)

Average results after 100 times of runs.

|    Size     |          CPU        |         GPU       | Acceleration |
|-------------|---------------------|-------------------|--------------|
| 240 x 150   | 9 ms               | 0.1 ms            |    90      |
| 480 x 300   | 34 ms               | 0.37 ms            |    91.89      |
| 960 x 600   | 140 ms              | 1.47 ms             |    95.23      |
| 1920x1200 | 452 ms   | 5.66 ms            |    79.85      |
| 3840x2400 | 2608 ms | 20.73 ms |    125.8      |


