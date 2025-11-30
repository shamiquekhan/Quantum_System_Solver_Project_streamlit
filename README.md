# Quantum Particle in a Box (PIAB) Explorer 🔬

A comprehensive interactive tool for understanding quantum mechanics through the 1D particle in a box model.

## Features

✨ **Interactive Visualizations:**
- Wave function and probability density plots
- Energy level diagrams
- Time evolution animations
- HOMO-LUMO molecular orbital analysis

📊 **Quantum Calculations:**
- Energy eigenvalues and wave functions
- Probability in spatial regions
- Expectation values (position, momentum, energy)
- Heisenberg Uncertainty Principle verification

🧪 **Real-World Applications:**
- Conjugated molecular orbital theory
- UV-Vis spectroscopy predictions
- β-Carotene color analysis (why carrots are orange!)

📥 **Data Export:**
- Export quantum state data as CSV, JSON, or TXT
- Download and analyze in your favorite tools

## Online Demo

🚀 **Try it live on Streamlit Community Cloud:**
[Quantum PIAB Explorer](https://quantum-piab-explorer.streamlit.app)

## Local Installation

### Prerequisites
- Python 3.8+
- pip

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/quantum-piab-explorer.git
cd quantum-piab-explorer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run app.py
```

4. **Open in browser:**
The app will automatically open at `http://localhost:8501`

## Project Structure

```
quantum-piab-explorer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── streamlit_config.toml # Streamlit configuration (optional)
```

## Usage

### 1. 📊 Wave Function & Probability
- Visualize wave functions and probability densities
- Compare multiple quantum states
- Check normalization

### 2. ⚡ Energy Levels
- View energy level diagrams
- See how energy scales with quantum number
- Calculate energy ratios

### 3. 🎯 Probability in Region
- Calculate probability of finding particle in specific regions
- Visual representation of integration regions
- Integration error estimates

### 4. 📐 Expectation Values
- Position and momentum statistics
- Heisenberg Uncertainty Principle verification
- Energy expectation values

### 5. ⏱️ Time Evolution
- Stationary state analysis
- Superposition state dynamics
- Quantum interference patterns

### 6. 🧪 Molecular Orbital Analysis
- HOMO-LUMO gap calculations
- UV-Vis absorption predictions
- Conjugation vs. color relationships
- Real molecular examples (ethylene, carotene, etc.)

### 7. 📥 Data Export
- Export quantum state data
- Multiple format support (CSV, JSON, TXT)
- Data statistics

## Physical Constants

The app uses **atomic units:**
- ℏ (reduced Planck constant) = 1.0
- m (particle mass) = 1.0
- e (elementary charge) = 1.0

## Theory

### Particle in a Box

The 1D particle in a box is a fundamental quantum mechanics problem where a particle is confined to move within a box of length L with infinite potential walls.

**Key equations:**

Energy eigenvalues:
```
E_n = n²π²ℏ² / (2mL²)
```

Wave functions:
```
ψ_n(x) = √(2/L) × sin(nπx/L)
```

Probability density:
```
|ψ_n(x)|² = (2/L) × sin²(nπx/L)
```

### Heisenberg Uncertainty Principle

```
Δx × Δp ≥ ℏ/2
```

### Molecular Orbital Connection

For conjugated molecules (polyenes), the PIAB model approximates π-electron behavior:
- Conjugation length → box length
- π-electrons → particle in box
- HOMO-LUMO gap → energy difference
- Absorption wavelength → color of molecule

## Deployment to Streamlit Community Cloud

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Quantum PIAB Explorer"
git branch -M main
git remote add origin https://github.com/yourusername/quantum-piab-explorer.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Community Cloud](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository
4. Choose branch: `main`
5. Set main file path: `app.py`
6. Click **Deploy!**

The app will be live at: `https://quantum-piab-explorer.streamlit.app`

## System Requirements

### Minimum
- RAM: 512 MB
- CPU: Single core
- Python: 3.8+

### Recommended
- RAM: 2+ GB
- CPU: Multi-core
- Python: 3.10+

## Performance Notes

- Calculations run in real-time (<100ms for most operations)
- Visualizations render instantly with matplotlib caching
- Streamlit provides automatic refresh on code changes

## Troubleshooting

### Issue: "Unable to deploy" on Streamlit Cloud
**Solution:** Make sure `requirements.txt` and `app.py` are in the repo root

### Issue: Plots not displaying
**Solution:** Clear cache: `streamlit cache clear`

### Issue: Slow performance
**Solution:** Reduce "Number of Points" slider to <1000

### Issue: Import errors
**Solution:** Install dependencies: `pip install -r requirements.txt`

## Educational Use

This tool is designed for:
- 🎓 Undergraduate Quantum Mechanics courses
- 📚 Physical Chemistry labs
- 🔬 Research demonstrations
- 🧠 Self-learning quantum theory

## Contributing

Contributions are welcome! Areas for enhancement:
- 3D visualization (orbital isosurfaces)
- 2D and 3D particle in a box
- Harmonic oscillator comparison
- Perturbation theory examples
- Interactive potential well design

## License

MIT License - feel free to use and modify

## Citation

If you use this tool in research or teaching, please cite:
```
Quantum PIAB Explorer (2025)
https://github.com/yourusername/quantum-piab-explorer
```

## Author

Created for quantum mechanics education

## Support

For issues, questions, or suggestions:
- GitHub Issues: [Report a bug](https://github.com/yourusername/quantum-piab-explorer/issues)
- Email: your.email@example.com

---

**Made with ❤️ for quantum learning**

*"The particle in a box is where quantum mechanics begins to make sense." - Physicist*
