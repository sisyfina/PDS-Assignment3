#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nlm.h"

float *readcsv(int *n, int *m, char *file_path){
​
    FILE *matFile;
    matFile = fopen(file_path, "r");
    if (matFile == NULL){
        printf("Could not open file %s",file_path);
        exit(-1);
    }
​
    float *X;
    if(strstr(file_path,"64") != NULL){
        *n = 64;
        *m = 64;
    }else if(strstr(file_path,"128") != NULL){
        *n = 128;
        *m = 128;
    }else if(strstr(file_path,"256") != NULL){
        *n = 256;
        *m = 256;
    }
    else{
      printf("Could not read file %s",file_path);
      exit(-1);
    }
​
    X = (float *)malloc((*n) * (*d) * sizeof(double));
​
    for(int i=0; i<*n; i++){
        for(int j=0; j<*d; j++){
            int got = fscanf(matFile, "%f", &X[i * (*d) + j]);
            if(got != 1){
                printf("Error reading\n");
                exit(-2);
            }
        }
    }
​
    fclose(matFile);
​
    return X;
}

__host__ float hg(int x, int y, int s){

  //float pi = 4.0 * atan(1.0);
  float result;
  float fx = (float)x;
  float fy = (float)y;
  float fs = (float)s;

  result = -(fx*fx + fy*fy)/(2*fs*fs);
  //result = exp(result)/(2*pi*fs*fs);
  result = exp(result);
}

__device__ float distance(float *pi, float *pj, float* gc, int patchSize){

  float result = 0.0;
  int center = (patchSize-1)/2;
  float sub;

  for (int i=0; i<patchSize; i++){
    for (int j=0; j<patchSize; j++){
      if (i!=center && j!=center){
        sub = fabs(pi[i*patchSize+j] - pj[i*patchSize+j]);
        sub = sub*sub;
        sub = sub*gc[i*patchSize+j];
        result = result+sub;
      }
    }
  }

  return result;

}

__device__ float Z(float* paddedI, float* gc, int r, int c, int x, int y, int patchSize, int filtSigma){

  int tmp1, tmp2;
  float z = 0.0, tmp3;
  int nr = r+patchSize-1;
  int nc = c+patchSize-1;
  int plus = (patchSize-1)/2;

  int nx = x+plus;
  int ny = y+plus;

  float *pi = (int *)malloc(patchSize*patchSize * sizeof(int *));

  tmp1 = 0;
  for (int i=nx-plus; i<=nx+plus; i++){
    tmp2 = 0;
    for (int j=ny-plus; j<=ny+plus; j++){
      pi[tmp1*patchSize+tmp2] = paddedI[i*c+j];
      tmp2++;
    }
    tmp1++;
  }

  float *pj = (int *)malloc(patchSize*patchSize * sizeof(int *));

  for (int k=0; k<r; k++){
    for (int l=0; l<c; l++){
      if (k!=x && l!=y){

        tmp1 = 0;
        for (int i=k; i<=k+2*plus; i++){
          tmp2 = 0;
          for (int j=l; j<=l+2*plus; j++){
            pj[tmp1*patchSize+tmp2] = paddedI[i*c+j];
            tmp2++;
          }
          tmp1++;
        }

        tmp3 = distance(pi, pj, gc, patchSize);
        tmp3 = -tmp3/(filtSigma*filtSigma);
        tmp3 = exp(tmp3);
        z = z + tmp3;
      }
    }
  }

  free(pi);
  free(pj);

  return z;
}

__device__ float w(float* paddedI, float* gc, int r, int c,
                   int x, int y, int x1, int y1, int patchSize, int filtSigma){

  float w, z;
  int tmp1, tmp2;
  int nr = r+patchSize-1;
  int nc = c+patchSize-1;
  int plus = (patchSize-1)/2;

  int nx = x+plus;
  int ny = y+plus;
  int nx1 = x1+plus;
  int ny1 = y1+plus;

  float *pi = (int *)malloc(patchSize*patchSize * sizeof(int *));

  tmp1 = 0;
  for (int i=nx-plus; i<=nx+plus; i++){
    tmp2 = 0;
    for (int j=ny-plus; j<=ny+plus; j++){
      pi[tmp1*patchSize+tmp2] = paddedI[i*c+j];
      tmp2++;
    }
    tmp1++;
  }

  float *pj = (int *)malloc(patchSize*patchSize * sizeof(int *));

  tmp1 = 0;
  for (int i=nx1-plus; i<=nx1+plus; i++){
    tmp2 = 0;
    for (int j=ny1-plus; j<=ny1+plus; j++){
      pj[tmp1*patchSize+tmp2] = paddedI[i*c+j];
      tmp2++;
    }
    tmp1++;
  }

  float tmp3 = distance(pi, pj, gc, patchSize);
  tmp3 = -tmp3/(filtSigma*filtSigma);
  tmp3 = exp(tmp3);

  z = Z(paddedI, gc, r, c, x, y, patchSize, filtSigma);

  w = tmp3/z;

  free(pi);
  free(pj);

  return w;
}

__global__ void kernel(float* If, float* paddedI_dev, float* gc_dev, int r, int c, int patchSize, int filtSigma) {

  float w, sum1;

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  plus = (patchSize-1)/2;

  sum1 = 0.0;
  for (int k=0; k<r; k++){
    for (int l=0; l<c; l++){
      w = w(paddedI_dev, gc_dev, r, c, i, j, k, l, patchSize, filtSigma);
      sum1 = sum1+w*paddedI_devI[(i+plus)*c+j+plus];
    }
  }
  If[i*c+j] = sum1;
}



float* nlm(float* I, int r, int c, int patchSize, int filtSigma, int patchSigma){


  int tmp;
  float z, *If;

  // filtered image
  float *If_host = (float *)malloc(r*c * sizeof(float *));
  float *If_dev;
  cudaMalloc( &If_dev, r*c * sizeof(float) );
  /*
   *   PADDING
   */

  int nr = r+patchSize-1;
  int nc = c+patchSize-1;
  int plus = (patchSize-1)/2;

  float *paddedI_host = (float *)malloc(nr*nc * sizeof(float));
  float *paddedI_dev;
  cudaMalloc( &paddedI_dev, nr*nc * sizeof(float) );

  // original
  for (int i=plus; i<r+plus; i++){
    for (int j=plus; j<c+plus; j++){
      paddedI_host[i*nc+j] = I[(i-plus)*c+j-plus];
    }
  }

  // symmetrical y-axis positive
  tmp = plus;
  for (int i=plus-1; i>=0; i--){
    for (int j=plus; j<c+plus; j++){
      paddedI_host[i*nc+j] = paddedI_host[tmp*nc+j];
      tmp++;
    }
  }

  // symmetrical y-axis negative
  tmp = r+plus-1;
  for (int i=plus+r; i<nr; i++){
    for (int j=plus; j<c+plus; j++){
      paddedI_host[i*nc+j] = paddedI_host[tmp*nc+j];
      tmp--;
    }
  }

  // symmetrical x-axis positive
  tmp = c+plus-1;
  for (int j=c+plus; j<nc; j++){
    for (int i=0; i<nr; i++){
      paddedI_host[i*nc+j] = paddedI_host[i*nc+tmp];
    }
    tmp--;
  }

  // symmetrical x-axis negative
  tmp = plus;
  for (int j=plus-1; j>=0; j--){
    for (int i=0; i<nr; i++){
      paddedI_host[i*nc+j] = paddedI_host[i*nc+tmp];
    }
    tmp++;
  }

  cudaMemcpy( paddedI_dev, paddedI_host, nr*nc * sizeof(float), cudaMemcpyHostToDevice );

  /*
   *   GAUSSIAN COEFFICIENTS
   */

  int center = (patchSize-1)/2;
  float sum = 0.0, max = 0.0;

  float *gc_host, *gc_dev;
  gc_host = (float *)malloc(patchSize*patchSize * sizeof(float));
  cudaMalloc( &gc_dev, patchSize*patchSize * sizeof(float) );

  for (int i=0; i<patchSize; i++){
    for (int j=0; j<patchSize; j++){
      gc_host[i*patchSize+j] = hg(abs(i-center), abs(j-center), patchSigma);
      sum = sum+gc_host[i*patchSize+j];
      if(max<gc_host[i*patchSize+j]) max=gc_host[i*patchSize+j];
    }
  }
  max = max/sum;

  for (int i=0; i<patchSize; i++){
    for (int j=0; j<patchSize; j++){
      gc_host[i*patchSize+j] = gc_host[i*patchSize+j]/(sum*max);
    }
  }

  cudaMemcpy( gc_dev, gc_host, patchSize*patchSize * sizeof(float), cudaMemcpyHostToDevice );

  /*
   *   If calculation
   */

  /* KERNEL */

  int numThreads = r*c;
  int numThreadsPerBlock = patchSize-1;
  int numBlocks = r;

  dim3 dimBlock( numThreadsPerBlock, numThreadsPerBlock );
  dim3 dimGrid( numThreads/dimBlock.x, numThreads/dimBlock.y );

  kernel<<<dimGrid, dimBlock>>>(If_host, paddedI_host, gc_host, r, c, int patchSize, int filtSigma);

  cudaMemcpy(fI, fI_dev, r*c*sizeof(float), cudaMemcpyDeviceToHost);

  free(paddedI_host);
  cudaFree(paddedI_dev);

  free(gc_host);
  cudaFree(gc_dev);

  return If;


}

int main(int argc, char* argv[]){

  int n, d;
  int patchSize = atoi(argv[2]);
  int filtSigma = atoi(argv[3]);
  int patchSigma = atoi(argv[4]);
  float *I = readcsv(&n, &d, argv[1]);
  float *If = nlm(I, n, d, patchSize, filtSigma, patchSigma);


}
