
# Anki Vector Robot Navigation and Image Captioning
![Alt Text](https://github.com/danikhan632/robot-path-finder/blob/main/media/path.gif)

## Overview

This repository contains a Python script designed to navigate an Anki Vector robot through a predefined path, capturing images at various points and generating descriptive captions based on the content of the images. The script integrates a pre-trained Vision-Text model for image captioning and utilizes the Anki Vector SDK for robot control, providing a seamless and interactive user experience.

## Prerequisites

- Anki Vector Robot with SDK installed
- Python 3.6 or higher
- PyTorch
- NumPy
- PIL (Python Imaging Library)
- Hugging Face Transformers Library

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/anki-vector-navigation-captioning.git
    cd anki-vector-navigation-captioning
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

- Ensure that the Anki Vector Robot is properly configured and connected to your development environment.
- The robot’s SDK should be installed and configured as per the official Anki Vector SDK documentation.
- Update the `data.json` file with the predefined poses for navigation. Each pose should contain X, Y coordinates and a direction angle.

## Usage

1. Run the script:

    ```bash
    python main.py
    ```

2. The script will initialize the Anki Vector Robot, capturing images and generating captions at each point in the predefined path.
3. Captions will be vocalized using the robot’s built-in text-to-speech functionality.
4. Movement details, including distances and directions, will be logged to the console for real-time monitoring.

## Features

- **Robot Navigation:** Navigate through a predefined path based on the poses specified in `data.json`.
- **Image Captioning:** Capture images using the robot’s camera and generate descriptive captions using a pre-trained Vision-Text model.
- **Voice Feedback:** Vocalize generated image captions for enhanced user interaction.
- **Error Handling:** Robust handling of `VectorTimeoutException` to ensure uninterrupted operation.
- **Real-time Logging:** Console outputs for movement details and operational status.

## Resources

- [Anki Vector SDK Documentation](https://developer.anki.com/vector/docs/index.html)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [PIL (Python Imaging Library)](https://pillow.readthedocs.io/en/stable/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
