import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from rich.console import Console

console = Console()

def load_csv_to_3d_array(file_path, num_keypoints):
    """Loads a CSV file and reshapes it into (frames, num_keypoints, 3)."""
    try:
        df = pd.read_csv(file_path)
        # Assuming CSV columns are x0,y0,z0, x1,y1,z1, ... or similar flattened structure per frame
        # Or, if each row is a keypoint and there's a frame_id column, more complex grouping is needed.
        # For this example, assuming (num_frames, num_keypoints * 3)
        if df.shape[1] != num_keypoints * 3:
            console.print(f"[bold red]Error:[/bold red] File {file_path} has {df.shape[1]} columns, expected {num_keypoints*3} for {num_keypoints} keypoints.")
            return None
        
        data = df.values.reshape(df.shape[0], num_keypoints, 3)
        return data.astype(np.float32)
    except Exception as e:
        console.print(f"[bold red]Error loading or reshaping {file_path}:[/bold red] {e}")
        return None

def preprocess_sequence_data(sequence, normalization_range=None):
    """Applies preprocessing to a single sequence (frames, K, 3)."""
    if sequence is None:
        return None

    # 1. Normalization (example: scale all coordinates globally for this sequence)
    if normalization_range:
        original_shape = sequence.shape
        data_flat = sequence.reshape(-1, 1) # Flatten to scale all values
        scaler = MinMaxScaler(feature_range=normalization_range)
        scaled_data_flat = scaler.fit_transform(data_flat)
        sequence = scaled_data_flat.reshape(original_shape)
    
    # 2. Missing data handling (simple linear interpolation along time axis)
    for kp_idx in range(sequence.shape[1]):
        for coord_idx in range(sequence.shape[2]):
            kp_coord_series = pd.Series(sequence[:, kp_idx, coord_idx])
            sequence[:, kp_idx, coord_idx] = kp_coord_series.interpolate(method='linear', limit_direction='both').fillna(0).values
            
    # 3. Outlier detection (placeholder - more advanced methods like Savitzky-Golay can be added)
    # For now, we assume data is relatively clean after interpolation.

    return sequence.astype(np.float32)


def load_pose_data(config):
    """
    Loads source (KP3D) and target (AL) pose data based on config.
    Performs train/test split by subject.
    """
    data_base_path = config['pose_retargeting']['data_base_path']
    source_suffix = config['pose_retargeting']['source_keypoints_file_suffix']
    target_suffix = config['pose_retargeting']['target_keypoints_file_suffix']
    source_kp = config['pose_retargeting']['source_num_keypoints']
    target_kp = config['pose_retargeting']['target_num_keypoints']
    movement_filters = config['pose_retargeting'].get('movement_filter', [])
    if isinstance(movement_filters, str): # Ensure it's a list
        movement_filters = [movement_filters]

    all_subject_folders = [os.path.join(data_base_path, d) for d in os.listdir(data_base_path)
                           if os.path.isdir(os.path.join(data_base_path, d)) and d.startswith("TDB_")]
    
    subject_ids = [os.path.basename(sf) for sf in all_subject_folders]

    if not subject_ids:
        console.print("[bold red]No subject folders found in {data_base_path}.[/bold red]")
        return [], [], [], []

    # Train/test split for subjects
    train_subject_ids, test_subject_ids = [], []
    if config['pose_retargeting'].get('train_subjects') and config['pose_retargeting'].get('test_subjects'):
        train_subject_ids = config['pose_retargeting']['train_subjects']
        test_subject_ids = config['pose_retargeting']['test_subjects']
    elif config['pose_retargeting'].get('train_subject_ratio'):
        ratio = config['pose_retargeting']['train_subject_ratio']
        if len(subject_ids) > 1 :
             train_subject_ids, test_subject_ids = train_test_split(
                subject_ids, 
                train_size=ratio, 
                random_state=config['settings'].get('global_random_seed', 42)
            )
        elif len(subject_ids) == 1: # If only one subject, use for both train/test for now
            console.print("[yellow]Warning: Only one subject found. Using for both training and testing.[/yellow]")
            train_subject_ids = test_subject_ids = subject_ids
        else: # Should not happen if check above passed
            console.print("[bold red]Error: No subjects available for splitting.[/bold red]")
            return [], [], [], []
    else:
        console.print("[bold red]Error: No train/test split strategy defined for subjects in config.[/bold red]")
        return [], [], [], []

    console.print(f"Training subjects: {train_subject_ids}")
    console.print(f"Testing subjects: {test_subject_ids}")

    source_train, target_train = [], []
    source_test, target_test = [], []

    def process_subjects(subject_list, source_data_list, target_data_list):
        for subject_id in subject_list:
            subject_folder_path = os.path.join(data_base_path, subject_id)
            
            # Find all files for this subject
            all_files = glob.glob(os.path.join(subject_folder_path, "*.csv"))

            processed_sequences_for_subject = 0
            for f_path in all_files:
                filename = os.path.basename(f_path)
                
                # Apply movement filter
                if movement_filters: # If filters are defined
                    if not any(mf in filename for mf in movement_filters):
                        continue # Skip if no movement filter matches

                if filename.endswith(source_suffix):
                    # Try to find matching target file
                    base_name = filename.replace(source_suffix, "")
                    target_file_name = base_name + target_suffix
                    target_file_path = os.path.join(subject_folder_path, target_file_name)

                    if os.path.exists(target_file_path):
                        console.print(f"  Processing pair: [cyan]{filename}[/cyan] and [cyan]{target_file_name}[/cyan]")
                        source_seq = load_csv_to_3d_array(f_path, source_kp)
                        target_seq = load_csv_to_3d_array(target_file_path, target_kp)

                        if source_seq is not None and target_seq is not None:
                            # Ensure same number of frames
                            min_frames = min(source_seq.shape[0], target_seq.shape[0])
                            source_seq = source_seq[:min_frames]
                            target_seq = target_seq[:min_frames]
                            
                            if min_frames == 0:
                                console.print(f"[yellow]Skipping pair due to zero frames after alignment: {filename}[/yellow]")
                                continue

                            norm_range = config['pose_retargeting']['preprocessing'].get('normalization_range')
                            source_seq_proc = preprocess_sequence_data(source_seq, norm_range)
                            target_seq_proc = preprocess_sequence_data(target_seq, norm_range)
                            
                            if source_seq_proc is not None and target_seq_proc is not None:
                                source_data_list.append(source_seq_proc)
                                target_data_list.append(target_seq_proc)
                                processed_sequences_for_subject +=1
            if processed_sequences_for_subject == 0:
                console.print(f"[yellow]Warning: No valid sequences found for subject {subject_id} with movement filters {movement_filters}.[/yellow]")


    console.print("\n[bold green]Loading training data...[/bold green]")
    process_subjects(train_subject_ids, source_train, target_train)
    console.print("\n[bold green]Loading testing data...[/bold green]")
    process_subjects(test_subject_ids, source_test, target_test)
    
    if not source_train or not target_train:
        console.print("[bold red]Critical Error: No training data loaded. Check paths, filters, and file formats.[/bold red]")

    return source_train, target_train, source_test, target_test