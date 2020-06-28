#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

#include "EBMP/EasyBMP.h"
#include <algorithm> 

//Russian characters aren't displayed.Comments in English, sorry...

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

using namespace std;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void saveImage(float* image, int height, int width, bool method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = image[i * width + j];
			pixel.Green = image[i * width + j];
			pixel.Blue = image[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}
	if (method)
		Output.WriteToFile("GPUoutAngelina.bmp");
	else
		Output.WriteToFile("CPUoutAngelina.bmp");

}

void noiseImg(float* image, int height, int width, int per) {
	BMP Output;
	Output.SetSize(width, height);

	int countOfPixels = int(height * width / 100 * per);

	while (countOfPixels > 0) {
		int i = rand() % height;
		int j = rand() % width;
		int c = rand() % 2;

		if (c == 1)
			image[i * width + j] = 255;
		else
			image[i * width + j] = 0;
		countOfPixels--;
	}


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = image[i * width + j];
			pixel.Green = image[i * width + j];
			pixel.Blue = image[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}
	Output.WriteToFile("NoiseAngelina.bmp");

}

void medianFilterCPU(float* image, float* resault, int height, int width)
{
	//mask3x3
	int m = 3;
	int n = 3;
	int mean = m * n / 2;
	int pad = m / 2;

	float* expandImageArray = (float*)calloc((height + 2 * pad) * (width + 2 * pad), sizeof(float));

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			expandImageArray[(j + pad) * (width + 2 * pad) + i + pad] = image[j * width + i];
		}
	}

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			float* window = (float*)calloc(m * n, sizeof(float));

			for (int k = 0; k < m; k++) {
				for (int t = 0; t < n; t++) {
					window[k * n + t] = expandImageArray[j * (width + 2 * pad) + i + k * (width + 2 * pad) + t];
				}
			}

			bool swapped = true;
			int t = 0;
			int tmp;

			while (swapped) {
				swapped = false;
				t++;
				for (int i = 0; i < m * n - t; i++) {
					if (window[i] > window[i + 1]) {
						tmp = window[i];
						window[i] = window[i + 1];
						window[i + 1] = tmp;
						swapped = true;
					}
				}
			}

			//sort(window, window + m * n);

			resault[j * width + i] = window[mean];
		}
	}

}

__global__ void myFilter(float* output, int imageWidth, int imageHeight) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// mask 3x3
	float window[9];
	int m = 3;
	int n = 3;
	int mean = m * n / 2;
	int pad = m / 2;

	for (int i = -pad; i <= pad; i++) {
		for (int j = -pad; j <= pad; j++) {
			window[(i + pad) * n + j + pad] = tex2D(texRef, col + j, row + i);
		}
	}

	bool swapped = true;
	int t = 0;
	int tmp;

	while (swapped) {
		swapped = false;
		t++;
		for (int i = 0; i < m * n - t; i++) {
			if (window[i] > window[i + 1]) {
				tmp = window[i];
				window[i] = window[i + 1];
				window[i + 1] = tmp;
				swapped = true;
			}
		}
	}

	output[row * imageWidth + col] = window[mean];
}



int main(void)
{
	int nIter = 100;
	BMP Image;
	Image.ReadFromFile("angelina.bmp");
	int height = Image.TellHeight();
	int width = Image.TellWidth();

	float* imageArray = (float*)calloc(height * width, sizeof(float));
	float* outputCPU = (float*)calloc(height * width, sizeof(float));
	float* outputGPU = (float*)calloc(height * width, sizeof(float));
	float* outputDevice;


	for (int j = 0; j < Image.TellHeight(); j++) {
		for (int i = 0; i < Image.TellWidth(); i++) {
			imageArray[j * width + i] = Image(i, j)->Red;
		}
	}

	noiseImg(imageArray, height, width, 8);

	unsigned int start_time = clock();

	for (int j = 0; j < nIter; j++) {
		medianFilterCPU(imageArray, outputCPU, height, width);
	}

	unsigned int elapsedTime = clock() - start_time;
	float msecPerMatrixMulCpu = elapsedTime / nIter;

	cout << "CPU time: " << msecPerMatrixMulCpu << endl;

	int device_count = 0;
	cudaGetDeviceCount(&device_count);

	if (device_count == 0)
		cout << "Sorry! You dont have CudaDevice" << endl;
	else {
		cout << "CudaDevice found! Device count: " << device_count << endl;

		// Allocate CUDA array in device memory

		//Returns a channel descriptor with format f and number of bits of each component x, y, z, and w
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaArray* cu_arr;

		checkCudaErrors(cudaMallocArray(&cu_arr, &channelDesc, width, height));
		checkCudaErrors(cudaMemcpyToArray(cu_arr, 0, 0, imageArray, height * width * sizeof(float), cudaMemcpyHostToDevice));		// set texture parameters
		texRef.addressMode[0] = cudaAddressModeClamp;
		texRef.addressMode[1] = cudaAddressModeClamp;
		texRef.filterMode = cudaFilterModePoint;


		// Bind the array to the texture
		cudaBindTextureToArray(texRef, cu_arr, channelDesc);

		checkCudaErrors(cudaMalloc(&outputDevice, height * width * sizeof(float)));

		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		cudaEvent_t start;
		cudaEvent_t stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// start record
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int j = 0; j < nIter; j++) {
			myFilter << <blocksPerGrid, threadsPerBlock >> > (outputDevice, width, height);
		}

		// stop record
		checkCudaErrors(cudaEventRecord(stop, 0));

		// wait end of event
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		float msecPerMatrixMul = msecTotal / nIter;

		cout << "GPU time: " << msecPerMatrixMul << endl;

		cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy(outputGPU, outputDevice, height * width * sizeof(float), cudaMemcpyDeviceToHost));

		cudaDeviceSynchronize();

		saveImage(outputGPU, height, width, true);
		saveImage(outputCPU, height, width, false);

		checkCudaErrors(cudaFreeArray(cu_arr));
		checkCudaErrors(cudaFree(outputDevice));
	}
	return 0;
}

