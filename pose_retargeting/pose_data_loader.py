import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from rich.console import Console
import re 

console = Console()

def extract_movement_from_filename(filename: str) -> str | None:
    """
    Extracts the movement identifier from the filename.
    Example: "TDB_001_F__F-JUMP__AL.csv" -> "F-JUMP"
             "TDB_001_F__A-POSE__KP3D.csv" -> "A-POSE"
             "TDB_001_F__F_J-JACKS__AL.csv" -> "F_J-JACKS" (if that's a pattern)
    Assumes pattern: SUBJECT__MOVEMENT__TYPE.csv
    """
    match = re.search(r'^[^_]+__[^_]+__([A-Za-z0-9_-]+)__.*\.csv$', filename)
    if match:
        return match.group(1)
    

    parts = filename.split('__')
    if len(parts) >= 3: 
        return parts[1] 
    
    console.print(f"[yellow]Warning: Could not extract movement from filename: {filename} using expected patterns.[/yellow]")
    return None


def load_csv_to_3d_array(file_path, num_keypoints):
    """Loads a CSV file and reshapes it into (frames, num_keypoints, 3)."""
    try:
        df = pd.read_csv(file_path)
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

    if normalization_range is not None:
        # --- Ensure normalization_range is a tuple for MinMaxScaler ---
        if not isinstance(normalization_range, tuple):
            try:
                normalization_range = tuple(normalization_range)
                if len(normalization_range) != 2:
                    console.print(f"[yellow]Warning: normalization_range '{normalization_range}' is not a 2-element sequence. Disabling normalization for this sequence.[/yellow]")
                    normalization_range = None 
            except TypeError:
                console.print(f"[yellow]Warning: normalization_range '{normalization_range}' could not be converted to a tuple. Disabling normalization for this sequence.[/yellow]")
                normalization_range = None
        
        if normalization_range: 
            original_shape = sequence.shape
            # MinMaxScaler expects 2D input (n_samples, n_features)
    
            data_flat = sequence.reshape(-1, 1) # Flatten to scale all values as one "feature"
            
            scaler = MinMaxScaler(feature_range=normalization_range)
            
            try:
                scaled_data_flat = scaler.fit_transform(data_flat)
                sequence = scaled_data_flat.reshape(original_shape)
            except Exception as e:
                console.print(f"[yellow]Warning: Error during MinMax scaling for a sequence: {e}. Skipping normalization for this sequence.[/yellow]")
                # sequence remains unnormalized if scaling fails
    
    # Missing data handling (simple linear interpolation along time axis)
    for kp_idx in range(sequence.shape[1]):
        for coord_idx in range(sequence.shape[2]):
            kp_coord_series = pd.Series(sequence[:, kp_idx, coord_idx])
            sequence[:, kp_idx, coord_idx] = kp_coord_series.interpolate(method='linear', limit_direction='both').fillna(0).values
            
    return sequence.astype(np.float32)


def _remove_all_minus_one_frames_synced(
    seq1: np.ndarray, 
    seq2: np.ndarray, 
    filename_info: str = "", 
    verbose_level: int = 0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Removes frames where all values are -1.0, synchronized across two sequences.
    Assumes seq1 and seq2 have already been truncated to the same number of frames.
    Returns cleaned (seq1, seq2), or (None, None) if resulting sequences are empty.
    """
    if seq1.shape[0] == 0: 
        return None, None

    # A frame is considered "valid" if AT LEAST ONE of its coordinate values is NOT -1.0.
    # Conversely, a frame is "invalid" if ALL its coordinate values ARE -1.0.
    
    # Reshape to (num_frames, num_keypoints * 3) to check all values in a frame easily
    seq1_reshaped = seq1.reshape(seq1.shape[0], -1)
    seq2_reshaped = seq2.reshape(seq2.shape[0], -1)

    # Create a boolean mask for valid frames in seq1
    # True if frame is valid (i.e., np.any returns True if any element is != -1.0)
    seq1_valid_frames_mask = np.any(seq1_reshaped != -1.0, axis=1)
    
    # Create a boolean mask for valid frames in seq2
    seq2_valid_frames_mask = np.any(seq2_reshaped != -1.0, axis=1)
    
    # A frame must be valid in BOTH sequences to be kept
    common_valid_mask = seq1_valid_frames_mask & seq2_valid_frames_mask
    
    seq1_filtered = seq1[common_valid_mask]
    seq2_filtered = seq2[common_valid_mask]

    num_frames_removed = seq1.shape[0] - seq1_filtered.shape[0]
    if num_frames_removed > 0 and verbose_level > 1: 
        console.print(f"      [grey50]Removed {num_frames_removed} invalid frames (all -1s) from {filename_info}[/grey50]")

    if seq1_filtered.shape[0] == 0: 
        return None, None
        
    return seq1_filtered, seq2_filtered


def load_pose_data(config, verbose=0):
    """
    Loads source (KP3D) and target (AL) pose data based on config.
    Performs train/test split by subject and filters by movement.
    """
    pr_config = config['pose_retargeting']
    data_base_path = pr_config['data_base_path']
    source_suffix = pr_config['source_keypoints_file_suffix']
    target_suffix = pr_config['target_keypoints_file_suffix']
    source_kp = pr_config['source_num_keypoints']
    target_kp = pr_config['target_num_keypoints']
    
    movement_filters_config = pr_config.get('movement_filter')
    if movement_filters_config is None or movement_filters_config == "":
        movement_filters = [] 
    elif isinstance(movement_filters_config, str):
        movement_filters = [movement_filters_config] 
    else:
        movement_filters = movement_filters_config 

    console.print(f"Movement filters active: {movement_filters if movement_filters else 'None (loading all movements)'}")

    all_subject_folders = [os.path.join(data_base_path, d) for d in os.listdir(data_base_path)
                           if os.path.isdir(os.path.join(data_base_path, d)) and d.startswith("TDB_")]
    
    subject_ids = [os.path.basename(sf) for sf in all_subject_folders]

    if not subject_ids:
        console.print(f"[bold red]No subject folders found in {data_base_path}.[/bold red]")
        return [], [], [], []

    train_subject_ids, test_subject_ids = [], []
    if pr_config.get('train_subjects') and pr_config.get('test_subjects'):
        train_subject_ids = pr_config['train_subjects']
        test_subject_ids = pr_config['test_subjects']
    elif pr_config.get('train_subject_ratio'):
        ratio = pr_config['train_subject_ratio']
        if len(subject_ids) > 1 :
             train_subject_ids, test_subject_ids = train_test_split(
                subject_ids, 
                train_size=ratio, 
                random_state=config['settings'].get('global_random_seed', 42)
            )
        elif len(subject_ids) == 1:
            console.print("[yellow]Warning: Only one subject found. Using for both training and testing.[/yellow]")
            train_subject_ids = test_subject_ids = subject_ids
        else:
            console.print("[bold red]Error: No subjects available for splitting.[/bold red]")
            return [], [], [], []
    else:
        console.print("[bold red]Error: No train/test split strategy defined for subjects in config.[/bold red]")
        return [], [], [], []

    console.print(f"Training subjects: {train_subject_ids}") if verbose > 0 else None
    console.print(f"Testing subjects: {test_subject_ids}") if verbose > 0 else None

    source_train, target_train = [], []
    source_test, target_test = [], []

    def process_subjects(subject_list, source_data_list, target_data_list, subject_type_str):
        console.print(f"\n[bold green]Processing {subject_type_str} subjects...[/bold green]")
        for subject_id in subject_list:
            console.print(f"  Subject: [cyan]{subject_id}[/cyan]") if verbose > 0 else None
            subject_folder_path = os.path.join(data_base_path, subject_id)
            
            all_files = glob.glob(os.path.join(subject_folder_path, f"*{source_suffix}")) # Only look for source files initially
            
            processed_sequences_for_subject = 0
            for source_file_path in all_files:
                filename = os.path.basename(source_file_path)
                
                movement_type = extract_movement_from_filename(filename)
                
                if movement_filters: 
                    if movement_type is None or movement_type not in movement_filters:
                        continue 
                
                base_name = filename.replace(source_suffix, "")
                target_file_name = base_name + target_suffix
                target_file_path = os.path.join(subject_folder_path, target_file_name)

                if os.path.exists(target_file_path):
                    console.print(f"    Loading pair for movement '{movement_type}': [cyan]{filename}[/cyan] & [cyan]{target_file_name}[/cyan]") if verbose > 1 else None
                    source_seq_raw = load_csv_to_3d_array(source_file_path, source_kp)
                    target_seq_raw = load_csv_to_3d_array(target_file_path, target_kp)

                    if source_seq_raw is not None and target_seq_raw is not None:
                        # Step 1: Ensure raw sequences have the same number of frames by truncating to the shorter one.
                        if source_seq_raw.shape[0] != target_seq_raw.shape[0]:
                            console.print(f"      [yellow]Warning: Raw sequences for {filename}/{target_file_name} have different frame counts ({source_seq_raw.shape[0]} vs {target_seq_raw.shape[0]}). Truncating to shortest.[/yellow]")
                            min_initial_frames = min(source_seq_raw.shape[0], target_seq_raw.shape[0])
                            if min_initial_frames == 0:
                                console.print(f"      [yellow]Skipping pair {filename} due to zero frames initially.[/yellow]")
                                continue
                            source_seq_raw = source_seq_raw[:min_initial_frames]
                            target_seq_raw = target_seq_raw[:min_initial_frames]
                        
                        # Step 2: Remove frames that are "all -1.0" from both sequences synchronously.
                        source_seq_filtered, target_seq_filtered = _remove_all_minus_one_frames_synced(
                            source_seq_raw,
                            target_seq_raw,
                            filename_info=f"{filename}/{target_file_name}",
                            verbose_level=verbose 
                        )

                        if source_seq_filtered is None or target_seq_filtered is None:
                            console.print(f"      [yellow]Skipping pair {filename}/{target_file_name} due to zero frames after filtering 'all -1s' rows.[/yellow]")
                            continue
                        
                        # source_seq_filtered and target_seq_filtered contain valid, synchronized frames.
                        # Their lengths are guaranteed to be equal by _remove_all_minus_one_frames_synced.
                        source_seq = source_seq_filtered
                        target_seq = target_seq_filtered
                        
                        # The min_frames check is now simpler as sequences are already aligned and filtered.
                        if source_seq.shape[0] == 0: 
                            console.print(f"      [yellow]Skipping pair {filename} (final check) due to zero frames.[/yellow]")
                            continue
                        
                        norm_range = pr_config['preprocessing'].get('normalization_range')
                        source_seq_proc = preprocess_sequence_data(source_seq, norm_range)
                        target_seq_proc = preprocess_sequence_data(target_seq, norm_range)
                        
                        if source_seq_proc is not None and target_seq_proc is not None:
                            source_data_list.append(source_seq_proc)
                            target_data_list.append(target_seq_proc)
                            processed_sequences_for_subject +=1


            if processed_sequences_for_subject == 0:
                console.print(f"    [yellow]Warning: No valid sequences loaded for subject {subject_id} with current movement filters.[/yellow]")

    process_subjects(train_subject_ids, source_train, target_train, "training")
    process_subjects(test_subject_ids, source_test, target_test, "testing")
    
    if not source_train or not target_train:
        console.print("[bold yellow]Warning: No training data loaded. Check paths, subject split, movement filters, and file formats.[/bold yellow]")
        
        
    return source_train, target_train, source_test, target_test