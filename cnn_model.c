#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "cnn_model_tflite.h" // This is your header file with the model data
#include "cnn_model.h" // Function declarations

#define kTensorArenaSize 10240
uint8_t tensor_arena[kTensorArenaSize];

TfLiteStatus RunInference(const uint8_t* input_data, uint8_t* output_data) {
    const tflite::Model* model = ::tflite::GetModel(g_cnn_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return kTfLiteError;
    }

    tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Copy the input data to the input tensor
    for (size_t i = 0; i < input->bytes; i++) {
        input->data.uint8[i] = input_data[i];
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        return kTfLiteError;
    }

    // Copy the output data from the output tensor
    for (size_t i = 0; i < output->bytes; i++) {
        output_data[i] = output->data.uint8[i];
    }

    return kTfLiteOk;
}

// Top function for Vitis HLS
void cnn_model_top(uint8_t input_data[28*28], uint8_t output_data[10]) {
    // Run inference using the TFLite model
    RunInference(input_data, output_data);
}
