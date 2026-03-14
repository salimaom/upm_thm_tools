# UPM-THM Tools

A Python library for 3D coupled Thermo-Hydro-Mechanical (THM) 
simulation of water flow and ice segregation in fractured rock masses
using the Unified Pipe Network Method (UPM).

## Scientific Basis
- Flow: Ren et al. (2017) - Unified Pipe Network Method
- Thermal: Chen et al. (2018) - T-H coupling in fractured rock
- Freezing: Clapeyron equation + hydraulic impedance function
- THM coupling: sequential iterative approach

## Modules
- `upm.geometry` - DFN generation and mesh
- `upm.flow` - Hydraulic UPM solver
- `upm.thermal` - Heat conduction and convection
- `upm.freezing` - Ice segregation and phase change
- `upm.mechanical` - Aperture update
- `upm.coupling` - THM coupling strategies
- `upm.excavation` - Underground opening inflow
- `upm.io` - Data readers and writers
- `upm.visualization` - 3D visualization

## Installation
```bash
conda activate science
pip install -e .
```

## Author
Salim Hammoum - PhD candidate
Polytechnique Montréal - 2026