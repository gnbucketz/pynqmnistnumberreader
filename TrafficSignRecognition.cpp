#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define FILE_TRAIN_IMAGE    "path_to_training_images"
#define FILE_TRAIN_LABEL    "path_to_training_labels"
#define FILE_TEST_IMAGE     "path_to_test_images"
#define FILE_TEST_LABEL     "path_to_test_labels"
#define LENET_FILE          "model.dat"
#define COUNT_TRAIN         39209  // Number of training images
#define COUNT_TEST          12630  // Number of test images

#pragma HLS_TOP name=TrafficSignRecognition


// Assuming the GTSRB images are 32x32 pixels, you may need to adjust this size accordingly
#define IMAGE_SIZE          32

// Function to read data from the GTSRB dataset
int read_data(unsigned char data[][IMAGE_SIZE][IMAGE_SIZE], unsigned char label[], const int count, const char data_file[], const char label_file[]) {
    // Implement the logic to read data from the GTSRB dataset
}

// Function to train the LeNet-5 model on the GTSRB dataset
void training(LeNet5 *lenet, unsigned char *train_data, uint8 *train_label, int batch_size, int total_size) {
    // Implement training logic
}

// Function to test the LeNet-5 model on the GTSRB dataset
int testing(LeNet5 *lenet, unsigned char *test_data, uint8 *test_label, int total_size) {
    // Implement testing logic
}

// Main function to load data, train the model, and test the model
void TrafficSignRecognition() {
    unsigned char *train_data = (unsigned char *)calloc(COUNT_TRAIN, sizeof(unsigned char[IMAGE_SIZE][IMAGE_SIZE]));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    unsigned char *test_data = (unsigned char *)calloc(COUNT_TEST, sizeof(unsigned char[IMAGE_SIZE][IMAGE_SIZE]));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL)) {
        printf("Error reading training data\n");
        return;
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("Error reading test data\n");
        return;
    }

    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (!lenet) {
        printf("Memory allocation failed\n");
        return;
    }

    Initial(lenet); // Initialize the LeNet-5 model

    clock_t start = clock();

    int batches[] = { 100, 200 }; // Batch sizes for training

    for (int i = 0; i < sizeof(batches) / sizeof(batches[0]); ++i) {
        training(lenet, train_data, train_label, batches[i], COUNT_TRAIN);
    }

    int right = testing(lenet, test_data, test_label, COUNT_TEST);

    printf("Accuracy: %d/%d\n", right, COUNT_TEST);
    printf("Time taken: %f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);

    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);
}

int main() {
	TrafficSignRecognition();
    return 0;
}
