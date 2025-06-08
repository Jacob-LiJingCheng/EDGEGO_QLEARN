# EDGEGO_QLEARN: Reinforcement Learning Enhanced EdgeGO Simulation

This repository provides a Python-based simulation framework that extends the original EdgeGO system. It incorporates Deep Q-Learning (DQN) to improve task scheduling and path planning in mobile edge computing for massive IoT environments.

## Reference Publication

R. Cong, Z. Zhao, G. Min, C. Feng, and Y. Jiang,  
"EdgeGO: A mobile resource-sharing framework for 6G edge computing in massive IoT systems,"  
IEEE Internet of Things Journal, vol. 9, no. 16, pp. 14521–14529, Aug. 2022.  
DOI: [10.1109/JIOT.2021.3065357](https://doi.org/10.1109/JIOT.2021.3065357)

### BibTeX

```bibtex
@article{cong2022,
  author    = {Rong Cong and Zhiwei Zhao and Geyong Min and Chenyuan Feng and Yuhong Jiang},
  title     = {EdgeGO: A Mobile Resource-sharing Framework for 6G Edge Computing in Massive IoT Systems},
  journal   = {IEEE Internet of Things Journal},
  volume    = {9},
  number    = {16},
  pages     = {14521--14529},
  year      = {2022},
  month     = {August},
  doi       = {10.1109/JIOT.2021.3065357}
}
```

## Project Structure

```
EDGEGO_QLEARN/
├── core/                        # Original EdgeGO logic (translated from MATLAB)
│   ├── PSTopLayer.py
│   ├── PathPlanning.py
│   ├── ComputeDDL.py
│   └── ...
│
├── utils/
│   └── replay.py               # DQN experience replay buffer
│
├── results/                    # Experiment results and figures
│   ├── *.png
│   ├── *.csv
│   └── *.npy
│
├── dqn.py                      # DQN model implementation
├── env_full.py                 # Environment simulation logic
├── config.yml                  # DQN hyperparameter configuration
│
├── Experiment_Distance.py              # Experiment 1: varying communication distances
├── Experiment_IoTScale.py              # Experiment 2: scaling IoT device count
├── Experiment_ComputationOverhead.py   # Experiment 3: varying computation loads
│
├── ResourceUtilization_*.py           # Original EdgeGO experiment scripts
└── README.md
```

## Experiments

This project includes three major experiments that compare the original IPTU strategy with a DQN-based method:

- `Experiment_Distance.py`: Varying average communication distance between devices
- `Experiment_IoTScale.py`: Increasing the number of IoT devices
- `Experiment_ComputationOverhead.py`: Increasing the computation overhead per device

Each script will output:

- Utilization statistics
- Figures (.png)
- Raw data in `.csv` and `.npy` formats  
All results are saved in the `results/` directory.

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, the required packages typically include:

- torch
- numpy
- matplotlib
- pyyaml
- scipy

You can generate your own:

```bash
pip freeze > requirements.txt
```

### 2. Configure DQN

Adjust learning rate, epsilon values, episodes, etc. in `config.yml`.

### 3. Run experiments

```bash
python Experiment_Distance.py
python Experiment_IoTScale.py
python Experiment_ComputationOverhead.py
```

## Notes

- The `core/` and `ResourceUtilization_*.py` files are translated from the original EdgeGO MATLAB simulation.
- The reinforcement learning code (`dqn.py`, `env_full.py`, `utils/replay.py`) is implemented to enable training with DQN.
- The environment supports ranking, 2-opt path exchange, and reward modeling based on time, utilization, path length, and stopping behavior.

## Appendix

The full source code is available at the following GitHub repository:  
https://github.com/Jacob-LiJingCheng/EDGEGO_QLEARN