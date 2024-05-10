#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>


#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)                                         \
{                                                                                       \
    FOREACH(o0,GETLENGTH(output))                                                      \
        FOREACH(o1,GETLENGTH(*(output)))                                                \
            FOREACH(w0,GETLENGTH(weight))                                               \
                FOREACH(w1,GETLENGTH(*(weight)))                                        \
                    (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];   \
}

#define CONVOLUTE_FULL(input,output,weight)                                             \
{                                                                                       \
    FOREACH(i0,GETLENGTH(input))                                                        \
        FOREACH(i1,GETLENGTH(*(input)))                                                 \
            FOREACH(w0,GETLENGTH(weight))                                               \
                FOREACH(w1,GETLENGTH(*(weight)))                                        \
                    (output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];    \
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)                     \
{                                                                               \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                 \
        for (int y = 0; y < GETLENGTH(*weight); ++y)                            \
            CONVOLUTE_VALID(input[x], output[y], weight[x][y]);                 \
    FOREACH(j, GETLENGTH(output))                                               \
        FOREACH(i, GETCOUNT(output[j]))                                          \
        ((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);   \
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)    \
{                                                                               \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                 \
        for (int y = 0; y < GETLENGTH(*weight); ++y)                            \
            CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);              \
    FOREACH(i, GETCOUNT(inerror))                                               \
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);             \
    FOREACH(j, GETLENGTH(outerror))                                             \
        FOREACH(i, GETCOUNT(outerror[j]))                                       \
        bd[j] += ((double *)outerror[j])[i];                                    \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                 \
        for (int y = 0; y < GETLENGTH(*weight); ++y)                            \
            CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);                   \
}


#define SUBSAMP_MAX_FORWARD(input,output)                                                      \
{                                                                                               \
    const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));                               \
    const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));                               \
    FOREACH(i, GETLENGTH(output))                                                               \
    FOREACH(o0, GETLENGTH(*(output)))                                                           \
    FOREACH(o1, GETLENGTH(**(output)))                                                          \
    {                                                                                           \
        int x0 = 0, x1 = 0, ismax;                                                             \
        FOREACH(l0, len0)                                                                       \
            FOREACH(l1, len1)                                                                   \
        {                                                                                       \
            ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
            x0 += ismax * (l0 - x0);                                                            \
            x1 += ismax * (l1 - x1);                                                            \
        }                                                                                       \
        output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];                              \
    }                                                                                           \
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)                                            \
{                                                                                               \
    const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));                           \
    const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));                           \
    FOREACH(i, GETLENGTH(outerror))                                                             \
    FOREACH(o0, GETLENGTH(*(outerror)))                                                         \
    FOREACH(o1, GETLENGTH(**(outerror)))                                                        \
    {                                                                                           \
        int x0 = 0, x1 = 0, ismax;                                                             \
        FOREACH(l0, len0)                                                                       \
            FOREACH(l1, len1)                                                                   \
        {                                                                                       \
            ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
            x0 += ismax * (l0 - x0);                                                            \
            x1 += ismax * (l1 - x1);                                                            \
        }                                                                                       \
        inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];                          \
    }                                                                                           \
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)                           \
{                                                                                       \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                         \
        for (int y = 0; y < GETLENGTH(*weight); ++y)                                    \
            ((double *)output)[y] += ((double *)input)[x] * weight[x][y];               \
    FOREACH(j, GETLENGTH(bias))                                                         \
        ((double *)output)[j] = action(((double *)output)[j] + bias[j]);                \
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)           \
{                                                                                       \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                         \
        for (int y = 0; y < GETLENGTH(*weight); ++y)                                    \
            ((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];           \
    FOREACH(i, GETCOUNT(inerror))                                                       \
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);                      \
    FOREACH(j, GETLENGTH(outerror))                                                     \
        bd[j] += ((double *)outerror)[j];                                               \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                         \
        for (int y = 0; y < GETLENGTH(*weight); ++y)                                    \
            wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];                  \
}
