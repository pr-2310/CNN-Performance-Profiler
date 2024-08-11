# CNN Performance Estimation on GPU

## Project Overview

This project focuses on evaluating the performance impact of Convolutional Neural Network (CNN) layers on GPU resources. The primary goal is to analyze the computational time and memory footprint of a Conv2d layer within a neural network handling the MNIST dataset. The project leverages PyTorch for model implementation and NVIDIA profiling tools to assess memory usage and execution times.

## Prerequisites

To run this project, ensure you have the following installed:

- **Python 3.x**: The programming language used.
- **PyTorch**: For implementing and training the CNN.
- **NVIDIA GPU**: Required for GPU profiling.
- **NVIDIA Nsight**: For profiling GPU memory and computational performance.

## Repository Structure

- **cnn_est.py**: Python script containing the CNN implementation and performance estimation code.
- **CNN_perf_est.pdf**: Documentation detailing the project approach, results, and analysis.

## Setup and Execution Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/pr-2310/CNN-Performance-Profiler.git
cd cnn-performance-estimation
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install torch torchvision
```

### 3. Run the CNN Performance Estimation Script

Execute the Python script to perform the CNN performance estimation:

```bash
python cnn_est.py
```

### 4. Analyze GPU Performance

The script includes timed sections to measure execution time. Additionally, to profile the GPU memory usage:

1. Run the script with NVIDIA Nsight for detailed profiling:

    ```bash
    nsight-systems -t 20s python cnn_est.py
    ```

2. Use the Nsight Compute tool to generate a detailed report:

    ```bash
    ncu --target-processes all --set full python cnn_est.py
    ```

### 5. Review Results

- The script will output execution times and memory usage for different batch sizes.
- Refer to the `CNN_perf_est.pdf` for a comprehensive analysis of the results, including comparisons between theoretical estimations and measured data.

## Summary of Findings

The project demonstrates distinct behaviors in time complexity and memory usage across varying batch sizes for CNN operations. The analysis provides insights into optimizing CNN training within constrained hardware environments by balancing computational resources and memory demands.

## Future Enhancements

- Implement support for additional CNN architectures to expand the performance analysis.
- Explore optimization techniques for reducing the computational overhead in deep learning tasks.
- Extend the analysis to different datasets and hardware configurations for broader applicability.

## Acknowledgments

Special thanks to the professors and tools that guided the implementation and analysis of this project.
