import sys
import torch
import torchvision
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import pytest

print("--- TASK 6: VERIFY PYTORCH CUDA ---")
print(f"Python version: {sys.version}")
print(f"torch.version: {torch.__version__}")
print(f"torchvision.version: {torchvision.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
is_available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {is_available}")

if is_available:
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory}")

    print("\nCUDA Tensor Test:")
    try:
        cpu_tensor = torch.ones(2, 2)
        print(f"Created tensor on CPU: {cpu_tensor.device}")
        cuda_tensor = cpu_tensor.to('cuda')
        print(f"Moved tensor to CUDA: {cuda_tensor.device}")
        result = cuda_tensor * 5
        print(f"Resulting tensor: \n{result}\nDevice: {result.device}")
    except Exception as e:
        print(f"CUDA tensor operation failed: {e}")
else:
    print("CUDA is NOT available. Test failed.")
    sys.exit(1)

print("\n--- TASK 7: VERIFY OTHER PACKAGES ---")
print(f"MediaPipe version: {mp.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Pytest version: {pytest.__version__}")
