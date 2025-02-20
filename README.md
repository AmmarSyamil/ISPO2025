# MIAW MCI Screening

MIAW MCI Screening is a web-based tool that screens for Mild Cognitive Impairment (MCI) by analyzing images from two cognitive tests:
- **Clock Drawing Test (CDT)**
- **Cube Copy Test (CCT)**

The application uses pre-trained TensorFlow models to classify the uploaded images and determine an overall screening result. The results include an overall classification (MCI Negative/Positive) based on the predictions of both tests, along with detailed probabilities and classifications available in a collapsible "Rincian" (Details) section.

## Features

- **Step-by-Step Upload Process:**  
  Users upload test images via modals that provide clear, step-by-step instructions.
  
- **Dual-Model Prediction:**  
  Two separate TensorFlow models are used—one for CDT and one for CCT—to evaluate cognitive performance.

- **Overall Screening Result:**  
  The app displays an overall result (MCI Negative or MCI Positive) based on defined criteria:
  - **MCI Negative:**  
    - CDT score is `5` and CCT score is either `3` or `2`, **or**
    - CDT score is `4` and CCT score is `3`
  - **MCI Positive:** Otherwise (with additional accumulated and average prediction probabilities).

- **Detailed Results:**  
  A collapsible section ("Rincian") shows side-by-side details (image preview, classification, and probability) for both CDT and CCT.

- **Modern UI with Animations:**  
  The interface includes smooth animations and responsive design using CSS and JavaScript.

## Requirements

- Python 3.6+
- Flask
- TensorFlow
- Pillow
- NumPy
- Werkzeug

> **Tip:** Use a virtual environment to manage dependencies.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/miaw-mci-screening.git
   cd miaw-mci-screening
2. **Set-Up Virtual environment**
   ```bash
   python -m venv venv
   > Windows
   venv\Scripts\activate
   > Linux
   source venv/bin/activate
3. **Download requirement**
   ```bash
   pip install -r requirements.txt


