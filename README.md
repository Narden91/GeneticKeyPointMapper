# Evolutionary Pose Retargeting System for Markerless Gait Analysis

## Overview

This system uses evolutionary algorithms to automatically learn how to map 3D human pose data from one representation system to another, specifically designed for markerless human movement and gait analysis applications. By using multi-objective optimization techniques (specifically NSGA-III), it finds efficient ways to translate between different skeletal structures with varying numbers of keypoints without requiring physical markers on subjects.

## Problem Statement

In clinical gait analysis and human movement research, there's an increasing shift from traditional marker-based systems to markerless video-based approaches:
- Converting from a simple skeleton (e.g., from markerless video tracking with 33 keypoints) to a detailed biomechanical model (e.g., with 54 keypoints needed for clinical analysis)
- Enabling analysis of movements captured with accessible camera systems instead of expensive specialized equipment
- Making gait and movement data from video recordings compatible with clinical assessment tools and biomechanical models

Manual retargeting between these different representation systems is time-consuming, requires specialized expertise, and introduces human bias. This system automates the process using machine learning while maintaining clinical-grade accuracy.

## How It Works

### 1. Input Data
The system requires paired examples of the same movements in both source and target formats:
- Source format (KP3D): CSV files containing 3D coordinates (x,y,z) for each keypoint from markerless video tracking systems over time
- Target format (AL): CSV files containing 3D coordinates for the corresponding anatomical landmarks in a biomechanical model
- File naming convention: `SubjectID__MovementType__Format.csv` (e.g., `TDB_001_F__F-JUMP__KP3D.csv` for source and `TDB_001_F__F-JUMP__AL.csv` for target)

The system is specifically designed to work with human movement data from different gait analysis modalities. It can process various movement types including:
- Standard walking gait (F-GAIT)
- Running (F-RUNNING)
- Jumping (F-JUMP)
- Static anatomical poses (A-POSE)

A detailed configuration file also guides the system, specifying data paths, keypoint numbers, movement filters, optimization parameters, and subject information for train/test splits.

### 2. Data Preprocessing
Before training, the raw input data undergoes several preparation steps:
1.  **Loading & Parsing:** CSV files are read and structured into numerical arrays representing sequences of 3D poses.
2.  **Pairing & Filtering:** Source and target files representing the same motion are paired. Sequences can be filtered by movement type based on configuration.
3.  **Frame Synchronization:** If paired sequences have different numbers of frames, they are typically truncated to the shorter length.
4.  **Missing Data Handling:**
    *   Frames with entirely missing data (e.g., marked by specific placeholder values) might be removed.
    *   Interpolation techniques can be used to fill in missing values within a sequence.
5.  **Normalization (Optional):** Pose coordinates can be normalized (e.g., by subject height) to make the learned transformation more robust to variations in subject size.
6.  **Train/Test Split:** Subjects are divided into training and testing sets. This step typically occurs after identifying all available data but before the main optimization loop, ensuring the model is trained on one subset of subjects and tested on another unseen subset.

### 3. Transformation Model
The core of the system is a mathematical transformation that converts source poses from markerless video tracking to biomechanically accurate target poses:

```
TargetPose = (C1 × SourcePose) × S + B
```

Where:
- **C1**: Correspondence matrix - determines which source keypoints influence which target anatomical landmarks
- **S**: Scaling factors - adjusts the magnitude of translated points to account for differences in skeletal proportions
- **B**: Bias vectors - applies position offsets to translated points to align with anatomical reference frames

Together, these parameters form the "genome" that the system evolves to find optimal transformations between the markerless tracking data and clinically relevant biomechanical landmarks.

### 4. Evolutionary Optimization (NSGA-III)
The system uses evolutionary algorithms to find the best transformation parameters:

1. **Initialization**: Create a population of random transformation parameters
2. **Evaluation**: Assess each transformation based on two objectives crucial for clinical gait analysis:
   - **Accuracy** (MPJPE - Mean Per Joint Position Error): How closely the transformed pose matches the anatomical landmarks needed for clinical analysis
   - **Temporal Consistency**: How smoothly the transformed poses flow over time, ensuring natural movement patterns without artifacts
3. **Selection**: Keep the best-performing transformations that preserve clinical features
4. **Variation**: Create new transformations through:
   - Crossover: Combining pieces of successful transformations
   - Mutation: Making small random changes to parameters
5. **Iteration**: Repeat for many generations to evolve increasingly effective transformations

The multi-objective approach (NSGA-III) finds solutions that balance anatomical accuracy and movement smoothness rather than sacrificing one for the other - essential for valid clinical assessments.

## Project Structure

```
├── main.py                  # Main execution script
├── config.yaml              # Configuration file
├── pose_retargeting/
│   ├── evolutionary_runner.py     # NSGA-III implementation
│   ├── genome_and_transform.py    # Transformation model
│   ├── pose_data_loader.py        # Data loading and preprocessing
│   └── retargeting_problem.py     # Problem definition
├── data/                    # Pose data files
│   ├── TDB_001_F/           # Data for subject 1
│   │   ├── TDB_001_F__F-JUMP__KP3D.csv  # Source pose data
│   │   └── TDB_001_F__F-JUMP__AL.csv    # Target pose data
│   └── ...
└── reports/                 # Output files
    ├── evolution_log.txt           # Log of the optimization process
    ├── evolution_history.pkl       # Saved optimization history
    └── pareto_front_plot.png       # Visualization of solution trade-offs
```

## Usage

### Prerequisites
- Python 3.7+
- Required packages: pymoo, numpy, matplotlib, pandas, rich, scikit-learn, PyYAML

### Setup
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Organize your pose data in the `data/` directory following the expected structure:
   - Source data should be keypoint data from markerless tracking systems
   - Target data should be the corresponding anatomical landmarks

### Configuration
Edit `config.yaml` to customize:
- Data paths and formats
- Number of keypoints in source (KP3D) and target (AL) representations
- Specific movement types to analyze (e.g., F-JUMP, F-GAIT, F-RUNNING)
- Optimization parameters (population size, generations, etc.)
- Genome parameter bounds to reflect biomechanical constraints

### Running the System
```bash
python main.py
```

The system will:
1. Load and preprocess the motion capture data
2. Define the optimization problem based on keypoint mapping
3. Run the NSGA-III evolutionary algorithm to find optimal transformations
4. Generate reports and visualizations of the results in the `reports/` directory

## Output and Results

The system generates several key outputs to help users understand and utilize the learned transformations:
- **Performance Reports:** Visualizations like the Pareto front and evolution progress plots, detailed log files, and a saved history of the optimization process.
- **Optimized Solutions:** The parameters for the best-performing transformations and their associated performance metrics.

### 1. Pareto Front Plot
The system produces a plot showing the trade-off between anatomical accuracy (MPJPE) and temporal consistency:
- Each point represents a different possible transformation solution
- Lower values on both axes are better for clinical applications
- Points are labeled with solution indices for reference
- The plot helps clinicians and researchers select the most appropriate transformation for their specific analysis needs

### 2. Solution Parameters
For selected solutions, the system provides:
- Performance metrics (MPJPE and temporal consistency scores)
- Transformation parameters (C1, S, B matrices) that define the mapping from markerless tracking to anatomical landmarks

### 3. Using the Results
After optimization, you can:
- Select a solution that offers your preferred balance between anatomical accuracy and movement smoothness
- Apply the transformation to new markerless tracking data using the `transform_source_to_target()` function in `genome_and_transform.py`
- Perform clinical gait analysis on the transformed data, including:
  - Joint angle calculations
  - Stride analysis
  - Kinematic assessments
  - Pathological gait pattern identification

## Example Configuration

```yaml
# Example configuration for gait analysis application
pose_retargeting:
  source_num_keypoints: 33  # Standard for many markerless video tracking systems
  target_num_keypoints: 54  # Detailed anatomical landmark model
  movement_filter: ["F-GAIT", "F-RUNNING"]  # Focus on locomotion patterns
  
  # Subject-based train/test split
  train_subject_ratio: 0.8  # Use 80% of subjects for training

nsga3_optimizer:
  population_size: 100
  num_generations: 50
  
  # Objective weights for final solution selection (Note: NSGA-III doesn't directly use weights for Pareto optimality,
  # but these could guide post-optimization selection or a weighted-sum approach if used differently)
  objective_weights:
    accuracy: 1.0             # Prioritize anatomical accuracy
    temporal_consistency: 0.5 # Still consider smoothness but with lower weight
```

## Customization

The system can be adapted for different clinical applications:
- Focus on specific movement types relevant to particular pathologies
- Adjust the train/test split to handle various patient populations
- Modify the transformation model to account for specific anatomical constraints
- Add new evaluation metrics for specialized clinical assessments
- Tune the evolutionary algorithm parameters to optimize for specific gait patterns

## Clinical Applications

This system can benefit various clinical and research applications:
- **Rehabilitation Assessment**: Track patient progress over time with affordable markerless systems
- **Sports Medicine**: Analyze athletic movements without restrictive markers
- **Orthopedic Evaluation**: Assist in pre/post-surgical assessment of gait
- **Neurological Assessment**: Quantify movement disorders like Parkinson's or cerebellar ataxia
- **Pediatric Gait Analysis**: Enable child-friendly assessment without the need for marker attachment
- **Remote Monitoring**: Allow for at-home gait assessment using consumer cameras

## Troubleshooting

Common issues:
- Data loading problems: Check file formats and naming conventions (should follow TDB_XXX_X__MOVEMENT-TYPE__FORMAT.csv)
- Memory issues: Reduce the number of sequences or frames processed
- Poor results: Increase population size or number of generations, adjust genome parameter bounds
- Subject variability: Ensure training data includes diverse subject characteristics if targeting heterogeneous populations

## Future Directions

Potential enhancements to the system:
- **Real-time Processing**: Optimize the transformation for real-time clinical feedback
- **Additional Modalities**: Extend to support other markerless tracking systems and clinical models
- **Pathology-specific Models**: Develop specialized transformations for specific movement disorders

## Authors

Emanuele Nardone/ Cesare Davide Pace

## Acknowledgements

This project uses the NSGA-III algorithm implementation from the pymoo library and is designed to advance the field of markerless human movement analysis for clinical applications.
