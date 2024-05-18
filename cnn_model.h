#ifndef CNN_MODEL_H_
#define CNN_MODEL_H_

#include <stdint.h>
#include "tensorflow/lite/micro/micro_interpreter.h"

// Function to run inference
TfLiteStatus RunInference(const uint8_t* input_data, uint8_t* output_data);

#endif // CNN_MODEL_H_
