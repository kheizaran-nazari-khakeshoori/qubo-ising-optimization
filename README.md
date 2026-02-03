# VRP-Ising Converter

A Python toolkit for converting Vehicle Routing Problems (VRP) and Quadratic Assignment Problems (QAP) into QUBO and Ising model formulations suitable for quantum annealing and optimization solvers.

## 📋 Overview

This repository contains implementations and tools for:
- **VRP to Ising Conversion**: Transform Vehicle Routing Problems into QUBO (Quadratic Unconstrained Binary Optimization) and then to Ising model format
- **QAP Local Search Solver**: Efficient local search algorithms for Quadratic Assignment Problems with statistical analysis

## 🚀 Features

### VRP-Ising Converter (`vrp-ising-converter.py`)
- Converts classical VRP instances into quantum-ready Ising models
- Handles multiple constraints:
  - Each customer visited exactly once
  - Vehicle capacity limits
  - Route position assignments
  - Distance minimization objective
- Generates QUBO matrices and converts them to Ising formulation (J, h parameters)
- Compatible with quantum annealers and Ising-based solvers

### QAP Solver (`QAP_N_5.py`, `QAP_N_12.py`)
- Efficient local search algorithm for Quadratic Assignment Problems
- Fast delta-swap operation for neighborhood exploration
- Statistical analysis with 10,000+ runs
- Visualization of energy distribution landscapes
- Demonstrates emergence of Gaussian distributions for larger problem sizes

## 📦 Installation

### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vrp-ising-converter.git
cd vrp-ising-converter
```

2. Create and activate virtual environment:
```bash
python3 -m venv vrp_venv
source vrp_venv/bin/activate  # On Linux/Mac
# or
vrp_venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install numpy matplotlib pulp
```

## 🔧 Usage

### VRP to Ising Conversion

```python
python vrp-ising-converter.py
```

This will:
- Define a sample VRP instance with cities, distances, and vehicle capacities
- Build the QUBO matrix with penalty constraints
- Convert to Ising model format (J matrix, h vector)
- Output the Ising parameters ready for quantum solvers

### QAP Local Search

For N=5 problem size with visualization:
```python
python QAP_N_5.py
```

For N=12 problem size showing Gaussian distribution:
```python
python QAP_N_12.py
```

Both scripts will:
- Generate random QAP instances
- Run 10,000 local search optimizations from random initial states
- Display energy distribution histograms
- Output statistical summaries (mean, std dev, best solution)

## 📊 Example Output

### VRP Converter
```
QUBO created: (32, 32)
Ising model created
J (interaction matrix): (32, 32)
h (local fields): (32,)
Constant term: -3500.0
```

### QAP Solver (N=12)
```
--- Statistics for N=12 ---
Best Energy found: 4234
Average (Mean) Energy: 4891.23
Standard Deviation: 156.47
```

## 🧮 Mathematical Background

### QUBO to Ising Conversion

The conversion follows the standard transformation:
- **QUBO**: Minimize x^T Q x, where x ∈ {0,1}^n
- **Ising**: Minimize s^T J s + h^T s, where s ∈ {-1,+1}^n

Transformation formulas:
```
J[i,j] = Q[i,j] / 4
h[i] = 0.5 * Σ Q[i,:]
constant = 0.25 * Σ Q
```

### QAP Formulation

The Quadratic Assignment Problem minimizes:
```
E = Σ_{i,j,k,l} F[i,k] * D[j,l] * x[i,j] * x[k,l]
```
where F is the flow matrix, D is the distance matrix, and x is the assignment matrix.

## 📁 Project Structure

```
vrp-ising-converter/
├── vrp-ising-converter.py    # Main VRP to Ising converter
├── QAP_N_5.py                # QAP solver for N=5
├── QAP_N_12.py               # QAP solver for N=12
├── vrp_venv/                 # Virtual environment
└── README.md                 # This file
```

## 🎯 Applications

- **Quantum Computing**: Prepare optimization problems for quantum annealers (D-Wave, etc.)
- **Combinatorial Optimization**: Classical and quantum approaches to NP-hard problems
- **Research**: Study energy landscapes and optimization difficulty
- **Benchmarking**: Test solver performance on VRP and QAP instances

## 📈 Visualization

The QAP solvers generate publication-quality histograms showing:
- Energy distribution across multiple optimization runs
- Mean energy with vertical reference line
- Clear demonstration of Central Limit Theorem for larger N

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available for academic and research purposes.

## 🔗 Related Topics

- Vehicle Routing Problem (VRP)
- Quadratic Assignment Problem (QAP)
- QUBO (Quadratic Unconstrained Binary Optimization)
- Ising Model
- Quantum Annealing
- Combinatorial Optimization
- Local Search Algorithms

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This is a research and educational project. For production use, consider additional validation and optimization techniques.
