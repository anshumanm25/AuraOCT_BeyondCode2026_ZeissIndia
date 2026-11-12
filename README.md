# AuraOCT_BeyondCode2026_ZeissIndia
# AuraOCT — Adaptive Under-sampled Reconstruction & Augmentation for Optical Coherence Tomography

**ZEISS India Hackathon — BeyondCode'26 | Theme 1: AI/ML for Advanced Frugal Innovation (AFI)**

## Overview
High-end intraoperative Optical Coherence Tomography (iOCT) systems cost **$50,000 to $150,000 (₹42L to ₹1.25Cr)** due to swept-source lasers, fast galvanometer mirrors, and multi-kHz line spectrometers. **AuraOCT** reduces hardware costs by over **99%** to a **$485 (~₹40,000)** benchtop optical engine by under-sampling retinal wavefronts by 80–90% and reconstructing continuous sub-micron 3D tissue coordinates in real time using a **Physics-Informed Implicit Neural Representation (INR)** model.

## Frugal Hardware Bill of Materials ($485 / ~₹40,000)
| Component | Specification | Est. Cost |
| :--- | :--- | :--- |
| **Light Engine** | 840nm Near-Infrared Superluminescent Diode + Driver | $65 (~₹5,400) |
| **Interferometer** | 50:50 Single-Mode 2x2 Fiber-Optic Coupler | $30 (~₹2,500) |
| **Scanner Mechanics** | Non-linear Voice-Coil Actuator Mirror Driver | $25 (~₹2,100) |
| **Optical Receiver** | Transimpedance Amplifier (TIA) + Photodetector | $35 (~₹2,900) |
| **Edge Digitizer** | STM32 Microcontroller (12-bit ADC, 921,600 baud USB) | $25 (~₹2,100) |
| **Optics & Enclosure** | Fiber collimators, lenses, and 3D-printed mount | $305 (~₹25,000) |

## System Architecture & Pipeline
1. **Hardware Acquisition (`firmware/auraoct_digitizer.ino`):** STM32 samples analog interference voltages from the TIA photodetector and streams 1D A-scan packets (`START,...,END`) over USB Serial at 921,600 baud.
2. **Physics-Informed INR Engine (`AuraOCT_Phase1_INR.ipynb`):** A Fourier-feature Coordinate MLP trained with **Helmholtz Wave Equation** PDE residuals ($\nabla^2 u + k^2 u = 0$) eliminates sparse-sampling aliasing without hallucinating non-physical tissue layers.
3. **End-to-End Connector (`auraoct_pipeline_connector.py`):** Ingests live USB serial streams (or virtual HIL captures), exports the PyTorch INR model to **ONNX Runtime** for `<20 ms` (`>50 FPS`) edge inference, and launches an interactive **Streamlit** surgical dashboard.

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the End-to-End Connector & Live Surgical Dashboard
streamlit run auraoct_pipeline_connector.py
