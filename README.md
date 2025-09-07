# Car Insurance App (CodeNection Hackathon Project)

## 🚗 Overview
This project is dedicated for Industry Collaboration track
Problem Statement:
When buying or renewing car insurance online, users often mistype or enter incorrect vehicle details (e.g., plate number, car model, year of manufacture). These mistakes can delay policy approval, cause pricing errors, or even lead to invalid insurance coverage. We plan to build a smart system that detects and corrects typos or inaccurate vehicle input specifications in real time – ensuring smoother, faster, and more reliable insurance applications.
This project is built for the **CodeNection Hackathon**.  
It is a **digital-first car insurance platform** that combines typo detection, decentralized insurance data storage, and vehicle damage evaluation using machine learning.

### ✨ Key Features
- **Smart Typo Checking**  
  Detects typos in user input for car brand, model, and validity of manufacturing year, then predicts the correct entry using a machine learning model (gotutiyan/gector-roberta-base-5k", torch_dtype="auto") transformer.
  GECToR treats error correction as sequence tagging.
  This one is fine-tuned on error correction tasks, so it learns how to detect and correct grammar, spelling, and word choice errors directly.
  We can just incorporate user_inputs.csv file into this transformer and all can run pretty smooth 

- **Decentralized Insurance Data**  
  A complete, secure, blockchain-backed system to store car details, user policies, and insurance schemes — making it fully digitalized and easily accessible. Blockchain data storage ensures transparency for validation, yet providing satisfying data security.

- **AI-Powered Damage Assessment**  
  Uses **YOLOv8 image processing** to detect vehicles from accident photos and evaluate the worn-out/damage percentage automatically. Users are required to take 3 photos of the car from different angles, which are rear view, left-side-front and right-side-front.

- **User-Friendly & Secure**  
  Designed with a simple interface, reliable backend, and robust data storage.

---

## 🛠️ Tech Stack
- **Frontend:** React Native (or Flutter) for cross-platform mobile app  
- **Backend:** Node.js / Express + MongoDB (secure storage)  
- **Machine Learning:**  
  - Typo detection model (Random Forest / NLP-based spell correction)  
  - YOLOv8 for image-based vehicle damage evaluation  
- **Deployment:** Docker + Cloud (AWS / GCP / Azure)  

---

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Kai-nf/car-insurance-app.git
   cd car-insurance-app
