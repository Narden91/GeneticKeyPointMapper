# Evolutionary Pose Retargeting System for Markerless Gait Analysis 🚶‍♀️➡️🤖

## Overview

This system uses evolutionary algorithms 🧬 to automatically learn how to map 3D human pose data from one representation system to another, specifically designed for markerless human movement and gait analysis applications. By using multi-objective optimization techniques (specifically NSGA-III), it finds efficient ways to translate between different skeletal structures with varying numbers of keypoints without requiring physical markers on subjects. The system also incorporates Bayesian Optimization (using Hyperopt with Tree-structured Parzen Estimators - TPE 🧐) to provide insights into optimal hyperparameters for the NSGA-III algorithm, further enhancing its performance and adaptability.

## Problem Statement ❓

In clinical gait analysis and human movement research, there's an increasing shift from traditional marker-based systems to markerless video-based approaches:
- Converting from a simple skeleton (e.g., from markerless video tracking with 33 keypoints) to a detailed biomechanical model (e.g., with 54 keypoints needed for clinical analysis) 💀➡️🦴
- Enabling analysis of movements captured with accessible camera systems 📸 instead of expensive specialized equipment
- Making gait and movement data from video recordings compatible with clinical assessment tools and biomechanical models

Manual retargeting between these different representation systems is time-consuming, requires specialized expertise, and introduces human bias. This system automates the process using machine learning 🧠 while maintaining clinical-grade accuracy.

## How It Works ⚙️

### 1. Input Data 💾
The system requires paired examples of the same movements in both source and target formats:
- Source format (KP3D): CSV files containing 3D coordinates (x,y,z) for each keypoint from markerless video tracking systems over time
- Target format (AL): CSV files containing 3D coordinates for the corresponding anatomical landmarks in a biomechanical model
- File naming convention: `SubjectID__MovementType__Format.csv` (e.g., `TDB_001_F__F-JUMP__KP3D.csv` for source and `TDB_001_F__F-JUMP__AL.csv` for target)

The system is specifically designed to work with human movement data from different gait analysis modalities. It can process various movement types including:
- Standard walking gait (F-GAIT) 🚶
- Running (F-RUNNING) 🏃‍♀️
- Jumping (F-JUMP) 🤸
- Static anatomical poses (A-POSE)🧍

A detailed configuration file (`config.yaml` 📄) also guides the system, specifying data paths, keypoint numbers, movement filters, optimization parameters, hyperparameter tuning settings, and subject information for train/test splits.

### 2. Data Preprocessing 🧼
Before training, the raw input data undergoes several preparation steps:
1.  **Loading & Parsing:** CSV files are read and structured into numerical arrays.
2.  **Pairing & Filtering:** Source and target files are paired. Sequences can be filtered by movement type.
3.  **Frame Synchronization:** Sequences are truncated to the shorter length if frame counts differ.
4.  **Missing Data Handling:** Frames with missing data might be removed or values interpolated.
5.  **Normalization (Optional):** Pose coordinates can be normalized (e.g., by subject height).
6.  **Train/Test Split:** Subjects are divided into training and testing sets.

### 3. Transformation Model 📐
The core of the system is a mathematical transformation:
```
TargetPose = (C1 × SourcePose) × S + B
```
Where:
- **C1**: Correspondence matrix
- **S**: Scaling factors
- **B**: Bias vectors
These parameters form the "genome" evolved by the system.

### 4. Hyperparameter Optimization (Optional - Hyperopt TPE) 🔮
Before the main evolutionary run, the system can optionally perform Bayesian Optimization using Hyperopt with Tree-structured Parzen Estimators (TPE) to find promising hyperparameters for NSGA-III.
1.  **Define Search Space**: NSGA-III hyperparameters (e.g., population size, crossover/mutation rates) and their ranges are defined in `config.yaml`.
2.  **Objective Function**: For each hyperparameter set, a shorter, faster NSGA-III run is performed. Its performance becomes the "loss" for TPE to minimize.
3.  **Iterative Search**: TPE iteratively suggests new configurations to find better hyperparameter regions.
4.  **Output**: The best hyperparameters found are used for the main NSGA-III run.

### 5. Evolutionary Optimization (NSGA-III) 🚀
The system uses NSGA-III to find the best transformation parameters (C1, S, B):
1. **Initialization**: Create a random population of transformation parameters.
2. **Evaluation**: Assess each transformation based on:
   - **Accuracy** (MPJPE - Mean Per Joint Position Error) 🎯
   - **Temporal Consistency** (Movement smoothness) 🌊
3. **Selection**: Keep the best-performing transformations.
4. **Variation**: Create new transformations (crossover & mutation).
5. **Iteration**: Repeat for many generations.

The multi-objective approach finds solutions that balance accuracy and smoothness.

## Project Structure 📂

```
├── main.py                     # Main execution script
├── config.yaml                 # Configuration file
├── pose_retargeting/
│   ├── __init__.py
│   ├── evolutionary_runner.py    # NSGA-III implementation
│   ├── genome_and_transform.py   # Transformation model
│   ├── pose_data_loader.py       # Data loading and preprocessing
│   └── retargeting_problem.py    # Problem definition for NSGA-III
├── bayesian_evo_opt/             # Bayesian hyperparameter optimization
│   ├── __init__.py
│   └── optimizer.py              # Hyperopt TPE implementation
├── data/                       # Pose data files (example)
│   ├── TDB_001_F/
│   │   ├── TDB_001_F__F-JUMP__KP3D.csv
│   │   └── TDB_001_F__F-JUMP__AL.csv
│   └── ...
└── reports/                    # Output files
    ├── evolution_log.txt              # Log of the optimization process
    ├── evolution_history.pkl          # Saved NSGA-III optimization history
    ├── pareto_front_plot.png          # Visualization of NSGA-III solutions
    └── bo_hyperopt_trials.csv         # (If BO run) CSV of TPE trials
```

## Usage 🛠️

### Prerequisites
- Python 3.10+ 🐍
- Required packages: pymoo, numpy, matplotlib, pandas, rich, scikit-learn, PyYAML, **hyperopt**

### Setup
1. Clone the repository: `git clone ...`
2. Install required packages: `pip install -r requirements.txt`
3. Organize your pose data in the `data/` directory.

### Configuration
Edit `config.yaml` 📄 to customize:
- Data paths and formats
- Keypoint numbers
- Movement types
- **Bayesian Optimization settings** (`bayesian_optimizer_settings`):
    - `run_bayesian_opt`: (true/false)
    - `max_evals_bo`: Number of TPE evaluations.
    - `search_space`: NSGA-III hyperparameter ranges.
    - `base_nsga3_config_for_bo_eval`: Config for short NSGA-III runs during TPE.
- **NSGA-III parameters** (`nsga3_optimizer`).
- Genome parameter bounds.

### Running the System
```bash
python main.py
```
The system will:
1. Load and preprocess data. 🔢
2. (Optional) Run Hyperopt TPE for NSGA-III hyperparameter tuning. ⚙️
3. Define the optimization problem.
4. Run NSGA-III to find optimal transformations. 🏆
5. Generate reports. 📊

## Output and Results 📊📈

- **Bayesian Optimization Report (if run):**
    - `bo_hyperopt_trials.csv`: CSV in `reports/` with TPE trial details.
    - Console logs.
- **NSGA-III Performance Reports:**
    - Visualizations (Pareto front, evolution progress - if plotting enabled).
    - `evolution_log.txt`: NSGA-III log.
    - `evolution_history.pkl`: NSGA-III history.
- **Optimized Solutions:** Transformation parameters (C1, S, B) and their scores.

### 1. Pareto Front Plot (NSGA-III)
If plotting is enabled, shows the accuracy (MPJPE) vs. temporal consistency trade-off.

### 2. Solution Parameters
For selected solutions: performance metrics and transformation matrices (C1, S, B).

### 3. Using the Results
- Select a solution.
- Apply to new data via `transform_source_to_target()`.
- Perform clinical gait analysis. 🩺

## Example Configuration

```yaml
# Example config.yaml entries
settings:
  verbose: 1
  global_random_seed: 42

pose_retargeting:
  source_num_keypoints: 33
  target_num_keypoints: 54
  # ...

bayesian_optimizer_settings:
  run_bayesian_opt: true
  max_evals_bo: 30
  # ... search_space ...
  base_nsga3_config_for_bo_eval:
    num_generations: 8

nsga3_optimizer:
  population_size: 100
  num_generations: 50
  # ...
```

## Customization 🎨
- Focus on specific movement types.
- Adjust train/test splits.
- Modify transformation model.
- Add new evaluation metrics.
- Tune algorithm parameters.

## Clinical Applications ⚕️
- Rehabilitation Assessment
- Sports Medicine ⚽
- Orthopedic Evaluation
- Neurological Assessment (e.g., Parkinson's)
- Pediatric Gait Analysis 👶
- Remote Monitoring 🏠

## Troubleshooting 🛠️🆘
- Data loading: Check file names/formats.
- Memory issues: Reduce sequences/frames, or TPE subset ratio.
- Poor results:
    - NSGA-III: Increase population/generations.
    - Bayesian Opt: Increase `max_evals_bo`, adjust `base_nsga3_config_for_bo_eval`.
- Subject variability: Ensure diverse training data.

## Future Directions 🚀
- Real-time Processing ⏱️
- Support for Additional Modalities
- Pathology-specific Models

## Authors

Emanuele Nardone / Cesare Davide Pace 

## Acknowledgements 🙏

This project uses **pymoo** (NSGA-III) and **Hyperopt** (TPE).
For a conceptual understanding of TPE: [Building a Tree-structured Parzen Estimator (from scratch, kind of)](https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478) 📖.
