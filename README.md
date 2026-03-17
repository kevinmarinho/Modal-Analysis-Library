# Structural Dynamics of Beams — Experimental & Numerical Analysis

## 📌 Overview

This project focuses on the dynamic analysis of beam structures by combining **experimental measurements** with **numerical modeling using the Finite Element Method (FEM)**. It enables comparison, validation, and calibration of computational models based on real vibration data.

The workflow integrates signal processing, modal analysis, and optimization techniques to estimate dynamic parameters such as natural frequencies, damping, and mode shapes.

---

## ⚙️ Features

* 📊 Experimental data processing (acceleration & hammer signals)
* 🔁 Frequency Response Function (FRF) computation
* 🧠 Modal identification (frequencies, damping, mode shapes)
* 🧮 Finite Element modeling of beam structures
* ⏱️ Time-domain and frequency-domain simulations
* 🎯 Model validation using correlation metrics:

  * MAC (Modal Assurance Criterion)
  * TSAC (Time-domain Assurance Criterion)
  * RVAC (Frequency-domain Assurance Criterion)
* 🤖 Parameter optimization using Genetic Algorithms

---

## 📁 Project Structure

```
├── Biblioteca.py          # Core library (FEM, signal processing, metrics)
├── main.py                # Experimental vs numerical comparison
├── main_otimizador.py     # Parameter optimization (damping calibration)
├── data/                  # Experimental datasets (not included)
└── README.md
```

---

## 🔬 Methodology

### 1. Experimental Analysis

* Load acceleration and impact (hammer) data
* Apply filtering and signal conditioning
* Compute FRFs using FFT
* Identify resonance peaks and modal parameters

### 2. Numerical Modeling

* Beam modeled using FEM with multiple DOFs per node
* Assembly of global mass and stiffness matrices
* Application of boundary conditions
* Time integration using state-space formulation

### 3. Model Validation

* Compare experimental and numerical results:

  * Time response
  * Frequency response (FRF)
* Evaluate similarity using correlation metrics

### 4. Optimization

* Use Genetic Algorithms to adjust damping parameters
* Maximize agreement between experimental and numerical data

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install numpy scipy pandas matplotlib sympy geneticalgorithm2
```

### 2. Run comparison

```bash
python main.py
```

### 3. Run optimization

```bash
python main_otimizador.py
```

---

## 📈 Example Outputs

* Acceleration vs time (experimental vs numerical)
* FRF comparison plots
* Identified modal parameters
* Optimized damping coefficients

---

## 🧠 Applications

* Structural dynamics analysis
* Aerospace structures (beams, wings, components)
* Vibration testing and model updating
* Experimental modal analysis (EMA)

---

## 🛠️ Technologies Used

* Python
* NumPy / SciPy
* SymPy (symbolic FEM formulation)
* Matplotlib
* GeneticAlgorithm2

---

## 👨‍💻 Author

Kevin Cohim Hereda de Freitas Marinho

---

## 📄 License

This project is intended for academic and research purposes.
