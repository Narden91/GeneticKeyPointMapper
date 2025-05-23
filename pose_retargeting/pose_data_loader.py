import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rich.console import Console
import re 


console = Console()


class PoseSequenceData:
    """Container for pose sequence data with normalization information."""
    def __init__(self, sequence, subject_id=None, movement_type=None, 
                 is_normalized=False, normalization_factor=None):
        self.sequence = sequence  # (frames, keypoints, 3)
        self.subject_id = subject_id
        self.movement_type = movement_type
        self.is_normalized = is_normalized
        self.normalization_factor = normalization_factor  # height in meters if normalized
        
    def get_denormalized_sequence(self):
        """Return the sequence in original scale (meters)."""
        if self.is_normalized and self.normalization_factor is not None:
            return self.sequence * self.normalization_factor
        return self.sequence
    
    @property
    def shape(self):
        return self.sequence.shape


def extract_movement_from_filename(filename: str) -> str | None:
    """
    Extracts the movement identifier from the filename.
    Example: "TDB_001_F__F-JUMP__AL.csv" -> "F-JUMP"
             "TDB_001_F__A-POSE__KP3D.csv" -> "A-POSE"
             "TDB_001_F__F_J-JACKS__AL.csv" -> "F_J-JACKS" (if that's a pattern)
    Assumes pattern: SUBJECT__MOVEMENT__TYPE.csv
    """
    # Updated regex to be more flexible with movement names, looking for content between double underscores
    # Specifically, __MOVEMENT__ or __SUBMOVEMENT-MOVEMENT__
    match = re.search(r'__([A-Za-z0-9_-]+(?:-[A-Za-z0-9_]+)*)__', filename)
    if match:
        return match.group(1)
    
    # Fallback if the above doesn't catch specific patterns like single char before dash
    parts = filename.split('__')
    if len(parts) >= 3: 
        return parts[1] 
    
    console.print(f"[yellow]Warning: Could not extract movement from filename: {filename} using expected patterns.[/yellow]")
    return None


def load_csv_to_3d_array(file_path, num_keypoints):
    """Loads a CSV file and reshapes it into (frames, num_keypoints, 3)."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            console.print(f"[yellow]Warning:[/yellow] File {file_path} is empty.")
            return None
        if df.shape[1] != num_keypoints * 3:
            console.print(f"[bold red]Error:[/bold red] File {file_path} has {df.shape[1]} columns, expected {num_keypoints*3} for {num_keypoints} keypoints.")
            return None
        
        data = df.values.reshape(df.shape[0], num_keypoints, 3)
        return data.astype(np.float32)
    except Exception as e:
        console.print(f"[bold red]Error loading or reshaping {file_path}:[/bold red] {e}")
        return None
    
def load_subject_characteristics(file_path: str) -> dict[str, float] | None:
    """
    Loads subject characteristics, specifically height, from a CSV file.
    The CSV is expected to have 'Subject_code' and 'Height (cm)' columns.
    Heights are returned in meters.
    """
    heights_map_cm = {}
    try:
        df = pd.read_csv(file_path, delimiter=';')
        if 'Subject_code' not in df.columns or 'Height (cm)' not in df.columns:
            console.print(f"[bold red]Error:[/bold red] CSV file '{file_path}' is missing 'Subject_code' or 'Height (cm)' columns.")
            return None
            
        for _, row in df.iterrows():
            subject_code = str(row['Subject_code']).strip()
            try:
                height_cm = float(row['Height (cm)'])
                if height_cm > 0: 
                    heights_map_cm[subject_code] = height_cm
                else:
                    console.print(f"[yellow]Warning: Invalid height (<=0) for subject {subject_code} in {file_path}. Height not stored.[/yellow]")
            except ValueError:
                console.print(f"[yellow]Warning: Could not parse height for subject {subject_code} in {file_path} (value: '{row['Height (cm)']}'). Height not stored.[/yellow]")
        
        if not heights_map_cm:
            console.print(f"[yellow]Warning: No valid subject heights loaded from {file_path}.[/yellow]")
        else:
            console.print(f"[green]✓ Loaded height data (in cm) for {len(heights_map_cm)} subjects from {file_path}.[/green]")
        return heights_map_cm

    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Subject characteristics file not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        console.print(f"[bold red]Error:[/bold red] Subject characteristics file is empty: {file_path}")
        return None
    except Exception as e:
        console.print(f"[bold red]Error reading subject characteristics file {file_path}:[/bold red] {e}")
        return None



def preprocess_sequence_data(sequence: np.ndarray, subject_height_m: float | None = None, 
                           normalize_by_height_flag: bool = False, subject_id: str = None,
                           movement_type: str = None):
    """
    Applies preprocessing to a single sequence (frames, K, 3).
    Returns a PoseSequenceData object with normalization information.
    """
    if sequence is None:
        return None

    is_normalized = False
    normalization_factor = None

    # --- Height Normalization ---
    if normalize_by_height_flag:
        if subject_height_m is not None and subject_height_m > 0:
            sequence = sequence / subject_height_m
            is_normalized = True
            normalization_factor = subject_height_m
        elif subject_height_m is None and normalize_by_height_flag:
            pass
        elif subject_height_m <= 0 and normalize_by_height_flag:
            console.print(f"[yellow]Warning: Invalid subject height ({subject_height_m}m) provided. Skipping height normalization for this sequence.[/yellow]")
            pass 
    
    # Missing data handling (simple linear interpolation along time axis)
    # Ensure sequence is not empty before trying to access shape
    if sequence.shape[0] > 0:
        for kp_idx in range(sequence.shape[1]):
            for coord_idx in range(sequence.shape[2]):
                kp_coord_series = pd.Series(sequence[:, kp_idx, coord_idx])
                # Interpolate and fill NaNs at the beginning/end with 0, or original value if no NaNs
                interpolated_values = kp_coord_series.interpolate(method='linear', limit_direction='both')
                # Check if all values became NaN after interpolation (e.g. all input was NaN)
                if interpolated_values.isnull().all():
                    sequence[:, kp_idx, coord_idx] = 0.0 
                else:
                    sequence[:, kp_idx, coord_idx] = interpolated_values.bfill().ffill().fillna(0).values
    else:
        # If sequence is empty (0 frames), just return it as is (or handle as error)
        pass
    
    return PoseSequenceData(
        sequence=sequence.astype(np.float32),
        subject_id=subject_id,
        movement_type=movement_type,
        is_normalized=is_normalized,
        normalization_factor=normalization_factor
    )


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
    Optionally normalizes data by subject height.
    Returns lists of PoseSequenceData objects instead of raw numpy arrays.
    """
    pr_config = config['pose_retargeting']
    data_base_path = pr_config['data_base_path']
    source_suffix = pr_config['source_keypoints_file_suffix']
    target_suffix = pr_config['target_keypoints_file_suffix']
    source_kp = pr_config['source_num_keypoints']
    target_kp = pr_config['target_num_keypoints']
    
    preprocessing_config = pr_config.get('preprocessing', {})
    normalize_by_height_flag = preprocessing_config.get('normalize_by_height', False)
    subject_char_file_name = preprocessing_config.get('subject_characteristics_file', "SUBJECTS_CHARACTERISTICS.csv")

    subject_heights_cm_map = None
    if normalize_by_height_flag:
        char_file_path = os.path.join(data_base_path, subject_char_file_name)
        subject_heights_cm_map = load_subject_characteristics(char_file_path)
        if subject_heights_cm_map is None:
            console.print(f"[yellow]Warning: Failed to load subject characteristics. Height normalization will be skipped.[/yellow]")
            normalize_by_height_flag = False # Disable if loading failed

    movement_filters_config = pr_config.get('movement_filter')
    if movement_filters_config is None or movement_filters_config == "" or not movement_filters_config : # Check for empty list too
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
        # Validate that these subjects exist
        train_subject_ids = [s_id for s_id in train_subject_ids if s_id in subject_ids]
        test_subject_ids = [s_id for s_id in test_subject_ids if s_id in subject_ids]
        if not train_subject_ids or not test_subject_ids:
             console.print(f"[yellow]Warning: Some specified train/test subjects not found in discovered subject_ids.[/yellow]")
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
        else: # Should be caught by earlier check, but defensive
            console.print("[bold red]Error: No subjects available for splitting.[/bold red]")
            return [], [], [], []
    else:
        console.print("[bold red]Error: No train/test split strategy defined for subjects in config (train_subjects/test_subjects or train_subject_ratio).[/bold red]")
        return [], [], [], []

    console.print(f"Training subjects ({len(train_subject_ids)}): {train_subject_ids}") if verbose > 0 else None
    console.print(f"Testing subjects ({len(test_subject_ids)}): {test_subject_ids}") if verbose > 0 else None

    source_train, target_train = [], []
    source_test, target_test = [], []

    def process_subjects(subject_list, source_data_list, target_data_list, subject_type_str, 
                         heights_map_param, norm_by_height_flag_param):
        console.print(f"\n[bold green]Processing {subject_type_str} subjects...[/bold green]")
        for subject_id in subject_list:
            console.print(f"  Subject: [cyan]{subject_id}[/cyan]") if verbose > 0 else None
            subject_folder_path = os.path.join(data_base_path, subject_id)
            
            all_source_files = glob.glob(os.path.join(subject_folder_path, f"*{source_suffix}"))
            
            processed_sequences_for_subject = 0
            
            current_subject_height_m = None
            valid_subj_height_norm = False
            if norm_by_height_flag_param and heights_map_param:
                subject_height_cm = heights_map_param.get(subject_id)
                if subject_height_cm is not None:
                    if subject_height_cm > 0:
                        current_subject_height_m = subject_height_cm / 100.0 # Convert cm to m
                        valid_subj_height_norm = True # Mark that a valid height to apply
                    else:
                        console.print(f"    [yellow]Warning: Invalid height ({subject_height_cm} cm) for subject {subject_id} in characteristics file. Skipping height normalization.[/yellow]")
                else:
                    console.print(f"    [yellow]Warning: Height not found for subject {subject_id} in characteristics file. Skipping height normalization.[/yellow]")
            
            for source_file_path in all_source_files:
                filename = os.path.basename(source_file_path)
                movement_type = extract_movement_from_filename(filename)
                
                if movement_filters: 
                    if movement_type is None or movement_type not in movement_filters:
                        continue 
                
                base_name = filename.replace(source_suffix, "")
                target_file_name = base_name + target_suffix
                target_file_path = os.path.join(subject_folder_path, target_file_name)

                if os.path.exists(target_file_path):
                    if verbose > 1:
                         log_msg = f"    Loading pair for movement '{movement_type}': [cyan]{filename}[/cyan] & [cyan]{target_file_name}[/cyan]"
                         if norm_by_height_flag_param and current_subject_height_m:
                             log_msg += f" (height norm: {current_subject_height_m:.2f}m)"
                         console.print(log_msg)
                         
                    source_seq_raw = load_csv_to_3d_array(source_file_path, source_kp)
                    target_seq_raw = load_csv_to_3d_array(target_file_path, target_kp)

                    if source_seq_raw is not None and target_seq_raw is not None:
                        if source_seq_raw.shape[0] != target_seq_raw.shape[0]:
                            min_initial_frames = min(source_seq_raw.shape[0], target_seq_raw.shape[0])
                            if verbose > 1:
                                console.print(f"      [yellow]Warning: Raw sequences for {filename}/{target_file_name} have different frame counts ({source_seq_raw.shape[0]} vs {target_seq_raw.shape[0]}). Truncating to {min_initial_frames}.[/yellow]")
                            if min_initial_frames == 0:
                                console.print(f"      [yellow]Skipping pair {filename} due to zero frames initially after length mismatch.[/yellow]")
                                continue
                            source_seq_raw = source_seq_raw[:min_initial_frames]
                            target_seq_raw = target_seq_raw[:min_initial_frames]
                        
                        source_seq_filtered, target_seq_filtered = _remove_all_minus_one_frames_synced(
                            source_seq_raw, target_seq_raw,
                            filename_info=f"{filename}/{target_file_name}", verbose_level=verbose 
                        )

                        if source_seq_filtered is None or target_seq_filtered is None or source_seq_filtered.shape[0] == 0:
                            if verbose > 1:
                                console.print(f"      [yellow]Skipping pair {filename}/{target_file_name} due to zero frames after filtering 'all -1s' rows or other issues.[/yellow]")
                            continue
                        
                        source_seq = source_seq_filtered
                        target_seq = target_seq_filtered
                                                
                        source_seq_data = preprocess_sequence_data(
                            source_seq, 
                            subject_height_m=current_subject_height_m,
                            normalize_by_height_flag=norm_by_height_flag_param and valid_subj_height_norm,
                            subject_id=subject_id,
                            movement_type=movement_type
                        )
                        target_seq_data = preprocess_sequence_data(
                            target_seq,
                            subject_height_m=current_subject_height_m,
                            normalize_by_height_flag=norm_by_height_flag_param and valid_subj_height_norm,
                            subject_id=subject_id,
                            movement_type=movement_type
                        )
                        
                        if (source_seq_data is not None and target_seq_data is not None and 
                            source_seq_data.sequence.shape[0] > 0 and target_seq_data.sequence.shape[0] > 0):
                            source_data_list.append(source_seq_data)
                            target_data_list.append(target_seq_data)
                            processed_sequences_for_subject +=1
                        elif verbose > 1:
                            console.print(f"      [yellow]Skipping pair {filename} post-processing due to empty sequence result.[/yellow]")
            
            if norm_by_height_flag_param and valid_subj_height_norm and processed_sequences_for_subject > 0 and verbose > 0:
                console.print(f"    [grey50]Applied height normalization (height: {current_subject_height_m:.2f}m) to {processed_sequences_for_subject} sequences for subject {subject_id}.[/grey50]")

            if processed_sequences_for_subject == 0 and verbose > 0 :
                console.print(f"    [yellow]Warning: No valid sequences loaded for subject {subject_id} with current movement filters and processing steps.[/yellow]")

    process_subjects(train_subject_ids, source_train, target_train, "training", 
                     subject_heights_cm_map, normalize_by_height_flag)
    process_subjects(test_subject_ids, source_test, target_test, "testing",
                     subject_heights_cm_map, normalize_by_height_flag)
    
    if not source_train or not target_train:
        console.print("[bold yellow]Warning: No training data loaded. Check paths, subject split, movement filters, file formats, and preprocessing steps.[/bold yellow]")
        
    return source_train, target_train, source_test, target_test