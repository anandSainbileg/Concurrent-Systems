//
// CSU33014 Lab 1
//

// Please examine version each of the following routines with names
// starting lab1. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics.

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".

#include <immintrin.h>
#include <stdio.h>

#include "lab1-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void lab1_routine0(float *restrict a, float *restrict b,
                   float *restrict c)
{
  for (int i = 0; i < 1024; i++)
  {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void lab1_vectorized0(float *restrict a, float *restrict b,
                      float *restrict c)
{
  __m128 a4, b4, c4;

  for (int i = 0; i < 1024; i = i + 4)
  {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float lab1_routine1(float *restrict a, float *restrict b,
                    int size)
{
  float sum = 0.0;

  for (int i = 0; i < size; i++)
  {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

// insert vectorized code for routine1 here
float lab1_vectorized1(float *restrict a, float *restrict b, int size)
{
  __m128 sum4 = _mm_setzero_ps();
  float sum_array[4];
  float sum = 0.0;
  int i = 0;
  __m128 a4, b4, prod4;
  for (; i <= size - 4; i += 4)
  {
    a4 = _mm_loadu_ps(&a[i]);
    b4 = _mm_loadu_ps(&b[i]);
    prod4 = _mm_mul_ps(a4, b4);
    sum4 = _mm_add_ps(sum4, prod4);
  }
  _mm_storeu_ps(sum_array, sum4);
  sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

  for (; i < size; i++)
  {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void lab1_routine2(float *restrict a, float *restrict b, int size)
{
  for (int i = 0; i < size; i++)
  {
    a[i] = 1 - (1.0 / (b[i] + 1.0));
  }
}

// in the following, size can have any positive value
void lab1_vectorized2(float *restrict a, float *restrict b, int size)
{
  __m128 one = _mm_set1_ps(1.0);
  __m128 b4, div, a4;
  int i = 0;
  for (; i <= size - 4; i += 4)
  {
    b4 = _mm_loadu_ps(&b[i]);
    b4 = _mm_add_ps(b4, one);
    div = _mm_div_ps(one, b4);
    a4 = _mm_sub_ps(one, div);
    _mm_storeu_ps(&a[i], a4);
  }

  for (; i < size; i++)
  {
    a[i] = 1 - (1.0 / (b[i] + 1.0));
  }
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void lab1_routine3(float *restrict a, float *restrict b, int size)
{
  for (int i = 0; i < size; i++)
  {
    if (a[i] < 0.0)
    {
      a[i] = b[i];
    }
  }
}

// in the following, size can have any positive value
void lab1_vectorized3(float *restrict a, float *restrict b, int size)
{
  // replace the following code with vectorized code
  __m128 zero = _mm_setzero_ps();
  __m128 a4, b4, mask, blended;
  int i = 0;
  for (; i <= size - 4; i += 4)
  {
    a4 = _mm_loadu_ps(&a[i]);
    b4 = _mm_loadu_ps(&b[i]);
    mask = _mm_cmplt_ps(a4, zero);
    blended = _mm_blendv_ps(a4, b4, mask);
    _mm_storeu_ps(&a[i], blended);
  }

  for (; i < size; i++)
  {
    if (a[i] < 0.0)
    {
      a[i] = b[i];
    }
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void lab1_routine4(float *restrict a, float *restrict b,
                   float *restrict c)
{
  for (int i = 0; i < 2048; i = i + 2)
  {
    a[i] = b[i] * c[i] - b[i + 1] * c[i + 1];
    a[i + 1] = b[i] * c[i + 1] + b[i + 1] * c[i];
  }
}

void lab1_vectorized4(float *restrict a, float *restrict b,
                      float *restrict c)
{
  // replace the following code with vectorized code
  __m128 b0, b1, c0, c1, real_b, real_c, imag_b, imag_c, real_part, imag_part, store1, store2;
  for (int i = 0; i < 2048; i += 8)
  {
    b0 = _mm_loadu_ps(&b[i]);
    b1 = _mm_loadu_ps(&b[i + 4]);
    c0 = _mm_loadu_ps(&c[i]);
    c1 = _mm_loadu_ps(&c[i + 4]);

    // Insert real and imaginary parts
    real_b = _mm_shuffle_ps(b0, b1, _MM_SHUFFLE(2, 0, 2, 0));
    imag_b = _mm_shuffle_ps(b0, b1, _MM_SHUFFLE(3, 1, 3, 1));
    real_c = _mm_shuffle_ps(c0, c1, _MM_SHUFFLE(2, 0, 2, 0));
    imag_c = _mm_shuffle_ps(c0, c1, _MM_SHUFFLE(3, 1, 3, 1));
    // Calculate real and imaginary parts of the product
    real_part = _mm_sub_ps(_mm_mul_ps(real_b, real_c), _mm_mul_ps(imag_b, imag_c));
    imag_part = _mm_add_ps(_mm_mul_ps(real_b, imag_c), _mm_mul_ps(imag_b, real_c));
    // Insert again real and imaginary parts to store back in a and put real and imaginary parts together
    store1 = _mm_unpacklo_ps(real_part, imag_part);
    store2 = _mm_unpackhi_ps(real_part, imag_part);

    _mm_storeu_ps(&a[i], store1);
    _mm_storeu_ps(&a[i + 4], store2);
  }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void lab1_routine5(unsigned char *restrict a,
                   unsigned char *restrict b, int size)
{
  for (int i = 0; i < size; i++)
  {
    a[i] = b[i];
  }
}

void lab1_vectorized5(unsigned char *restrict a,
                      unsigned char *restrict b, int size)
{
  __m128i temp;
  int i = 0;
  for (; i <= size - 16; i += 16)
  {
    // load and store 16 bytes
    temp = _mm_loadu_si128((__m128i *)&b[i]);
    _mm_storeu_si128((__m128i *)&a[i], temp);
  }
  for (; i < size; i++)
  {
    a[i] = b[i];
  }
}

/********************* routine 6 ***********************/

void lab1_routine6(float *restrict a, float *restrict b,
                   float *restrict c)
{
  // replace the following code with vectorized code
  a[0] = 0.0;
  for (int i = 1; i < 1023; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < 3; j++)
    {
      sum = sum + b[i + j - 1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void lab1_vectorized6(float *restrict a, float *restrict b,
                      float *restrict c)
{
  a[0] = 0.0;
  // load c and duplicate each for all four lanes
  __m128 c0 = _mm_load1_ps(&c[0]);
  __m128 c1 = _mm_load1_ps(&c[1]);
  __m128 c2 = _mm_load1_ps(&c[2]);

  __m128 b0, b1, b2, prod0, prod1, prod2, sum4;
  int i = 1;
  for (; i <= 1023 - 4; i += 4)
  {
    // load blocks of b values to align with all c
    b0 = _mm_loadu_ps(&b[i - 1]);
    b1 = _mm_loadu_ps(&b[i]);
    b2 = _mm_loadu_ps(&b[i + 1]);

    // multiply all b with c
    prod0 = _mm_mul_ps(c0, b0);
    prod1 = _mm_mul_ps(c1, b1);
    prod2 = _mm_mul_ps(c2, b2);

    sum4 = _mm_add_ps(prod0, prod1);
    sum4 = _mm_add_ps(sum4, prod2);
    _mm_storeu_ps(&a[i], sum4);
  }

  for (; i < 1023; i++)
  {
    float sum = 0.0;
    sum += b[i - 1] * c[0];
    sum += b[i] * c[1];
    sum += b[i + 1] * c[2];
    a[i] = sum;
  }

  // make sure the last element is set to 0.0 as the original routine
  a[1023] = 0.0;
}
