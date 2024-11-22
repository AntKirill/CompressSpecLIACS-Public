# CompressSpecLIACS

A collaborative project between **LIACS** and **SRON** focused on the *CompresSpec* initiative. This repository contains the code and tools used to simulate trace gas measurement devices, apply optimization algorithms for filter selection, and perform the experiments described in [Antonov et al., 2024](#references).  

## Overview

This project provides:
- **Simulation**: A simulator for a trace gas measurement device.  
- **Optimization**: Implementation of evolutionary algorithms, specifically tailored for optimizing filter selection.  
- **Experiments**: Reproduction of all experiments detailed in the referenced publication.  

Code contributions:
- **LIACS**: Implementation of optimization and numerical experiments.  
- **SRON**: Development of the trace gas measurement simulation.  

## Getting Started

### Prerequisites

- Python environment (ensure compatibility with all dependencies).  

### Installation

1. Clone the repository:  
   ```bash
   git clone <repository-url>
   ```
2. Add the repository folder to your Python environment:  
   ```bash
   export PYTHONPATH="$PYTHONPATH:/path/to/repo"
   ```

### Running the Optimization

To optimize filter selection using the top-performing algorithm "UMDA-U-PLS-Dist, $d_1$" as described in [1], execute the following command:  

```bash
python optimizationV3.py -a umda2-dist \
  --folder_name Exp_umda2-dist_$(date '+%d-%m-%Y_%Hh%Mm%Ss') \
  --n_segms 16 \
  --budget 2110 \
  --robustness fixed \
  --n_reps 1000 \
  --mu_ 10 \
  --lambda_ 20 \
  --d0_method 2 \
  --d1_method kirill \
  --instance 0  
```

**Notes:**  
- The parameter `--d0_method 2` corresponds to distance metric $d_1$.  
- Use `--d0_method 3` for $d_2$ (due to historical reasons).  

## References

[1] Antonov, Kirill, et al. *"Selection of Filters for Photonic Crystal Spectrometer Using Domain-Aware Evolutionary Algorithms."* arXiv preprint arXiv:2410.13657 (2024).  
