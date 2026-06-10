# ⚛️ 1D Quantum System Solver and Visualizer

---

## 📋 Project Information

### **Course Details**
- **Subject**: Introduction to Computational Chemistry
- **Professor**: Dr. Saurav Prasad
- **Institution**: Vellore Institute of Technology (VIT Bhopal)
- **Semester**: Fall Semester 2025

### **Student Contributors**
- **Shamique Khan** (Reg No. 25BAI10187)
- **Prachi Kamboj** (Reg No. 25BAI10874)
- **Prashant Singh** (Reg No. 25BAI10980)

---

## 🚀 Online Demo

✨ **Try the interactive app live on Streamlit Community Cloud:**
[Quantum PIAB Explorer](https://quantum-piab-explorer.streamlit.app)

---

## 🎯 Overview

This project implements a **complete computational solution** for solving and visualizing the **1D Particle in a Box (PIAB)** quantum system. It bridges the gap between abstract quantum mechanics theory and practical computational chemistry applications, demonstrating how quantum confinement affects molecular electronic structure and spectroscopic properties.

### **Key Innovation**
Connects theoretical quantum mechanics to real-world chemistry by modeling π-electron behavior in conjugated organic molecules and predicting UV-Vis absorption spectra.

---

## 🌟 Features

### **Interactive Streamlit App**
- **Wave function and probability density plots**
- **Energy level diagrams**
- **Time evolution animations**
- **HOMO-LUMO molecular orbital analysis**
- **Data Export:** Export calculated data as CSV, JSON, or TXT

### **Core Quantum Mechanics Features**
1. **Energy Eigenvalue Calculator:** E_n = n² × π²ℏ²/(2mL²)
2. **Wave Function Solver:** ψ_n(x) = √(2/L) × sin(nπx/L)
3. **Probability Density Analysis:** |ψ_n(x)|² with normalization verification
4. **Regional Probability Calculation:** Numerical integration for P(x₁ ≤ x ≤ x₂)
5. **Expectation Values & Uncertainties:** Verification of Heisenberg Uncertainty Principle (Δx·Δp ≥ ℏ/2)
6. **Time Evolution:** Stationary states and superposition dynamics
7. **Molecular Orbital Connection:** π-electrons in conjugated molecules (Ethylene, β-Carotene, etc.)
8. **HOMO-LUMO Spectroscopy:** Predicted absorption wavelengths and color analysis

---

## 🔧 Installation & Setup

### **Local Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/shamiquekhan/Quantum_System_Solver_Project_streamlit.git
cd Quantum_System_Solver_Project_streamlit
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run quantum_piab_app.py
```

4. **Run the Jupyter Notebook:**
```bash
jupyter notebook Quantum_System_Solver_Project.ipynb
```

---

## 📖 Theoretical Foundation

### **The Schrödinger Equation**
Ĥψ(x) = Eψ(x) where Ĥ = -ℏ²/(2m) × d²/dx² + V(x)

### **PIAB Potential**
V(x) = 0 if 0 ≤ x ≤ L, otherwise ∞

### **Exact Solutions**
- **Energy:** E_n = n²π²ℏ²/(2mL²)
- **Wave Function:** ψ_n(x) = √(2/L) × sin(nπx/L)
- **Heisenberg Uncertainty:** Δx × Δp ≥ ℏ/2

---

## 📝 Project Structure

```
.
├── quantum_piab_app.py      # Main Streamlit application
├── quantum_piab_minimal.py  # Minimal version of the app
├── Quantum_System_Solver_Project.ipynb # Detailed Jupyter Notebook
├── requirements.txt         # Project dependencies
├── streamlit_config.toml    # Streamlit configuration
├── PROJECT_SUMMARY.md       # Detailed project summary
├── FILE_LISTING.md          # Description of all project files
├── QUICKREF.md              # Quick reference for quantum formulas
└── README.md                # This file
```

---

## 📄 License

MIT License - feel free to use and modify

**Attribution**: 
- Created by Shamique Khan (25BAI10187) & Prachi Kamboj (25BAI10874)
- Under guidance of Dr. Saurav Prasad
- VIT Bhopal, 2025
