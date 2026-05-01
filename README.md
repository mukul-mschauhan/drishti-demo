<img width="3072" height="3072" alt="Dhanush Logo" src="https://dhanushai.com" />

# DRISHTI-SENTINEL: Sovereign AI Fire & Smoke Detection System 

**DRISHTI-SENTINEL** is a robust, edge-deployed computer vision system engineered for the real-time detection of fire and smoke in austere environments, specifically designed for Armoured Fighting Vehicles (AFVs). 

This repository contains the evaluation User Interface (UI), designed to demonstrate the model's capabilities in a secure, tactical dashboard.

## 🛡️ Project Overview

DRISHTI-SENTINEL leverages a highly optimized YOLOv11n architecture, fine-tuned on a proprietary dataset of over 31,000 interior and exterior AFV images. It is built to operate under the stringent constraints of defense-grade edge hardware, delivering high-speed inference without relying on cloud connectivity.

### Core Capabilities:
*   **Real-Time Threat Detection:** Identifies fire and smoke instances with high precision.
*   **Edge-Optimized Inference:** Engineered for deployment on NVIDIA Jetson Orin Nano / AGX architectures using TensorRT (FP16/INT8).
*   **Tactical User Interface:** A streamlined, defense-styled dashboard for operators to monitor system health, view live inference streams, and review session telemetry.
*   **Batch & Stream Processing:** Supports single-image analysis, batch processing for forensic review, and continuous live-stream inference (hardware permitting).

## 🚀 The Evaluation Demo (Streamlit)

This specific repository hosts the DRISHTI-SENTINEL Evaluation Dashboard, built with Streamlit. It is designed to run in a controlled environment to demonstrate model accuracy and system design to evaluators.

### Features of the Demo UI:
*   **Detection Console:** Upload images or batches for immediate inference, complete with confidence scores and localized bounding boxes.
*   **Performance Metrics:** Review the model's validation curves (Precision, Recall, mAP@50).
*   **Live Telemetry:** Tracks session statistics (fire events, smoke events) and calculates average inference speed across batches.
*   **Session Log:** A complete, downloadable CSV log of all detections during the current session for post-mission analysis.

## 🛠️ System Architecture & Model Card

*   **Architecture:** YOLO11n (Ultralytics Framework)
*   **Parameters:** 2.58M
*   **GFLOPs:** 6.3
*   **Training Data:** 31,781 AFV internal/external images (Merged Dataset v4)
*   **Classes:** `0: Fire`, `1: Smoke`
*   **Hardware Target:** NVIDIA Jetson Orin Nano Super
*   **Precision:** FP32 (PyTorch Demo), FP16/INT8 (TensorRT Production)
*   **Native Inference Speed:** ~1.8ms / frame (on Jetson Orin)

## 💻 Running the Demo Locally

To run this dashboard on your local machine or an evaluation laptop:

### Prerequisites:
*   Python 3.9+
*   Git

### Setup Instructions:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/drishti-sentinel-demo.git](https://github.com/your-username/drishti-sentinel-demo.git)
    cd drishti-sentinel-demo
    ```

2.  **Create and activate a virtual environment (Recommended):**
    *   **Windows:**
        ```powershell
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **Linux/Mac:**
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    *The dashboard will automatically open in your default web browser at `http://localhost:8501`.*

## 🔒 Confidentiality Notice

This repository and its contents, including the fine-tuned model weights (`best.pt`), are the intellectual property of **Dhanush AI Innovation Pvt Ltd**. This evaluation build is provided strictly for authorized review and demonstration purposes. Do not distribute, reverse-engineer, or deploy in production environments without explicit authorization.

---
*© 2026 Dhanush AI Innovation Pvt Ltd. All rights reserved.*
