# Real-Time American Sign Language (ASL) Gesture Recognition Interface

## Architecture Specification

### Core Pipeline
1. **Webcam**: Captures real-time video feed.
2. **OpenCV**: Processes video frames (read, basic manipulation).
3. **MediaPipe Hand Landmarker**: Extracts 21 3D hand landmarks from the frame.
4. **Handedness Canonicalization**: Normalizes coordinates based on whether the detected hand is left or right.
5. **Wrist-centered Normalization**: Translates all points relative to the wrist (landmark 0).
6. **Scale Normalization**: Scales the coordinates to a standard range to achieve distance invariance.
7. **Feature Vector**: Flattens the coordinates into a 63-dimensional feature vector.
8. **PyTorch MLP**: A Multi-Layer Perceptron model processes the feature vector.
9. **Classes**: 29 outputs (A-Z + SPACE + DELETE + NOTHING).
10. **Confidence Filtering**: Discards predictions below a certain confidence threshold.
11. **Time-based FSM (Finite State Machine)**: Smooths predictions over time to avoid flickering and handles gesture confirmation.
12. **Text Buffer**: Accumulates the recognized gestures into text.
13. **OpenCV UI**: Renders the text buffer, bounding boxes, and landmarks back onto the video feed.

### Benchmark Module (Separate)
*   **Sign Language MNIST**: A separate benchmark task using a custom PyTorch Conv2D CNN on 28x28 grayscale images (24 static classes).
*   **IMPORTANT**: The Sign Language MNIST CNN must NOT be connected to the MediaPipe landmark pipeline.
