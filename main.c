#include <immintrin.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define MATRIX_SIZE 4096

#define NUM_THREAD 16

float matA[MATRIX_SIZE][MATRIX_SIZE];
float matB[MATRIX_SIZE][MATRIX_SIZE];
float matC[MATRIX_SIZE][MATRIX_SIZE];
int step = 0;

void* multiplicacao()
{

    __m256 vecA, vecB;

    __m256 vecC;

    int core = step++;
    for (int i = core * (MATRIX_SIZE / NUM_THREAD); i < (core + 1) * MATRIX_SIZE/ NUM_THREAD; i++)
        for (int j = 0; j < MATRIX_SIZE; j++) {
            vecA = _mm256_set1_ps(matA[i][j]);
            for (int k = 0; k < MATRIX_SIZE; k += 8) {
                vecB = _mm256_load_ps(&matB[j][k]);
                vecC = _mm256_load_ps(&matC[i][k]);
                vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                _mm256_store_ps(&matC[i][k], vecC);
            }
        }
}

int main()
{
    int i,j;
    struct timespec start, finish;
    double elapsed;

    pthread_t threads[NUM_THREAD];


    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matA[i][j] = 5.0f;
            matB[i][j] = 2.0f;
            matC[i][j] = 0.0f;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (i = 0; i < NUM_THREAD; i++) {
        pthread_create(&threads[i], NULL, multiplicacao, NULL);
    }


    for (i = 0; i < NUM_THREAD; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    //Printar apenas se for de dimensao 8
    if(MATRIX_SIZE == 8) {

        printf("Matriz A");
        for (i = 0; i < MATRIX_SIZE; i++) {
            printf("\n");
            for (j = 0; j < MATRIX_SIZE; j++)
                printf("%f ", matA[i][j]);
        }

        printf("\n\n\n");
        printf("Matriz B");
        for (i = 0; i < MATRIX_SIZE; i++) {
            printf("\n");
            for (j = 0; j < MATRIX_SIZE; j++)
                printf("%f ", matB[i][j]);
        }

        printf("\n\nMultiplicação da matrix A com B\n");
        for (i = 0; i < MATRIX_SIZE; i++) {
            printf("\n");
            for (j = 0; j < MATRIX_SIZE; j++)
                printf("%f ", matC[i][j]);
        }
    }

    printf("\n\nNumero de threads utilizadas -> %d\n", NUM_THREAD);
    printf("\nTamanho da matriz -> %d\n", MATRIX_SIZE);
    printf("\nTime of execution -> %f\n", elapsed);

}
