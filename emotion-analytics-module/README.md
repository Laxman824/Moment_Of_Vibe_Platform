# Emotion Analytics Module for Moment of Vibe

A standalone emotion analytics module for real-time voice call analysis, designed to detect and score emotions (anger, joy, energy, confidence) from audio inputs using OpenSmile feature extraction and neural network classification.

## 🎯 Overview

This module is part of the Moment of Vibe (MOV) MVP architecture, specifically implementing the AI Processing Layer for emotion detection. It processes audio chunks in simulated real-time and outputs structured emotion scores with quality metrics and context-aware suggestions.

### Key Features

- **Multi-label Emotion Classification**: Detects anger, joy, energy, and confidence simultaneously
- **Audio Quality Analysis**: Computes SNR, RMS energy, clipping detection, and overall quality scores
- **Real-time Processing**: Processes 10-second audio chunks in <1.5 seconds
- **Context-Aware Suggestions**: Rule-based suggestions based on emotion patterns
- **Integration-Ready**: Outputs match ProcessingResult interface for seamless MOV integration

## 📋 Requirements

### System Requirements
- Python 3.12+
- 8GB RAM minimum
- Optional: CUDA-capable GPU for faster training

### Performance Targets
- ✅ Accuracy: >75% on validation set
- ✅ Latency: <1.5 seconds per 10-second audio chunk
- ✅ Code Coverage: >80%

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd emotion-analytics-module

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .