#include <iostream>
#include "cnn_model.h"

int main() {
    // Assuming input data is 28x28 grayscale image and output data is 10 classes
    const int input_size = 28 * 28;
    const int output_size = 10;
    uint8_t input_data[input_size] = { /* Your input data here */ };
    uint8_t output_data[output_size];

    // Run inference
    TfLiteStatus status = RunInference(input_data, output_data);
    if (status != kTfLiteOk) {
        std::cerr << "Inference failed!" << std::endl;
        return -1;
    }

    // Process the output data
    std::cout << "Inference successful! Output data:" << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << static_cast<int>(output_data[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}
