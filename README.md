# Efficient-VeRL-Agent: Post-Training LLMs for Reasoning & Embodied AI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![Task](https://img.shields.io/badge/Tasks-Reasoning_&_Embodied-green)

## 📖 Introduction

This project implements a scalable Post-Training (RLHF) system based on the [VeRL](https://github.com/volcengine/verl) framework. It focuses on reproducing and optimizing SOTA algorithms for two challenging domains:

1.  **DeepSearch R1 (Reasoning)**: Reproducing the **DeepSeek-R1** pipeline (GRPO) for math and logic reasoning tasks.
2.  **Alfworld (Embodied AI)**: Solving long-horizon decision-making tasks using PPO and GiGPO.

This repository serves as the implementation code for my research on **Memory-Efficient RLHF** and **Long-Horizon Credit Assignment**.

## 🚀 Key Features

### 1. Algorithm Evolution (From PPO to SOTA)
We implemented a systematic evolution of alignment algorithms:
* **PPO (Baseline)**: Standard Actor-Critic architecture optimized for consumer GPUs (4x4090).
* **GRPO (Critic-free)**: Implemented **Group Relative Policy Optimization** (DeepSeek-R1 style), removing the Value Network to reduce memory usage by **45%**.
* **GiGPO (Long-Horizon)**: Deployed **Group-in-Group** mechanism to solve sparse reward problems in horizon > 30 tasks.

### 2. DeepSearch R1 Reproduction
* **Method**: Implemented GRPO with group sampling for advantage estimation.
* **Tasks**: Focused on Math (GSM8K) and Coding (LiveCodeBench) reasoning paths.
* **Outcome**: Achieved **2x Batch Size** increase on constrained hardware (A10 Cluster/4090) compared to PPO.

### 3. Embodied Agent Optimization (Alfworld)
* **Challenge**: Reward Hacking and Mode Collapse in long-chain instructions.
* **Solution**: Integrated GiGPO with step-level advantage estimation.
* **Result**: Improved Success Rate from **62% (PPO)** to **86% (GiGPO)**.

## 🛠️ Repository Structure

* `verl/`: Modified framework core (Scheduler & Worker logic).
* `examples/`: Original implementations for R1, Search, and Alfworld.
    * `examples/r1/`: **DeepSearch R1 reproduction code**.
    * `examples/search/`: Retrieval-augmented generation setups.
* `custom_scripts/`: Ready-to-run experiments.
    * `reasoning_R1/`: Scripts for DeepSeek-R1 style training.
    * `embodied_alfworld/`: Scripts for PPO/GiGPO on Alfworld.

## 📊 Experimental Results

### Memory Efficiency (PPO vs. GRPO)
| Algorithm | Critic Network | Peak Memory (GB) | Throughput |
| :--- | :---: | :---: | :---: |
| **PPO** | ✅ | ~22.5 GB | 1.0x |
| **GRPO (R1)** | ❌ (Removed) | **~12.4 GB** (↓45%) | **1.3x** |



## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/skyblue-ustc/Efficient-VeRL-Agent.git
cd Efficient-VeRL-Agent
pip install -r requirements.txt
```
