import numpy as np
from pymoo.core.problem import Problem
from .genome_and_transform import Genome, transform_source_to_target


class PoseRetargetingProblem(Problem):
    def __init__(self, 
                 source_sequences_train,  # list of PoseSequenceData objects
                 target_sequences_train,  # list of PoseSequenceData objects
                 source_dim_config,       # (num_source_kp, 3)
                 target_dim_config,       # (num_target_kp, 3)
                 genome_param_bounds,     # Dict with 'C1_bounds', 'S_bounds', 'B_bounds'
                 trim_percentage=0.1,     # Percentage of worst errors to exclude (0.1 = 10%)
                 use_adaptive_scaling=True):  # Enable adaptive objective scaling
        
        self.source_sequences = source_sequences_train
        self.target_sequences = target_sequences_train
        self.source_dim = source_dim_config 
        self.target_dim = target_dim_config
        self.trim_percentage = trim_percentage
        self.use_adaptive_scaling = use_adaptive_scaling
        
        # Scaling factors (will be set adaptively or use defaults)
        self.objective_scales = None
        self._eval_count = 0
        self._store_unscaled = False
        
        num_source_kp, _ = self.source_dim
        num_target_kp, _ = self.target_dim

        # Calculate total number of variables in the flattened genome
        n_var_C1 = num_target_kp * num_source_kp
        n_var_S = num_target_kp * 3
        n_var_B = num_target_kp * 3
        self.n_var_total = n_var_C1 + n_var_S + n_var_B

        # Define lower and upper bounds for variables
        c1_low, c1_high = genome_param_bounds['C1_init_bounds']
        s_low, s_high = genome_param_bounds['S_init_bounds']
        b_low, b_high = genome_param_bounds['B_init_bounds']

        xl_list = ([c1_low] * n_var_C1) + ([s_low] * n_var_S) + ([b_low] * n_var_B)
        xu_list = ([c1_high] * n_var_C1) + ([s_high] * n_var_S) + ([b_high] * n_var_B)
        
        # We now have 6 objectives:
        # f1: Standard MPJPE (meters)
        # f2: Normalized MPJPE (dimensionless)
        # f3: Trimmed MPJPE (meters) - robust metric
        # f4: Median Per Joint Position Error (meters) - robust metric
        # f5: Temporal Consistency (meters)
        # f6: 90th Percentile Error (meters) - robust metric
        
        super().__init__(n_var=self.n_var_total, 
                         n_obj=6,  # Increased from 2 to 6
                         n_constr=0,
                         xl=np.array(xl_list, dtype=np.float32),
                         xu=np.array(xu_list, dtype=np.float32))

    def _calculate_robust_metrics(self, errors_all_frames_kps):
        """
        Calculate robust error metrics from a collection of per-keypoint, per-frame errors.
        
        Args:
            errors_all_frames_kps: list of arrays, each of shape (frames, keypoints)
        
        Returns:
            dict with various robust metrics
        """
        # Flatten all errors into a single array
        all_errors = []
        for errors in errors_all_frames_kps:
            all_errors.extend(errors.flatten())
        all_errors = np.array(all_errors)
        
        if len(all_errors) == 0:
            return {
                'mean': np.inf,
                'median': np.inf,
                'trimmed_mean': np.inf,
                'percentile_90': np.inf,
                'percentile_95': np.inf
            }
        
        # Standard mean
        mean_error = np.mean(all_errors)
        
        # Median (robust to outliers)
        median_error = np.median(all_errors)
        
        # Trimmed mean (exclude top trim_percentage)
        if self.trim_percentage > 0:
            n_samples = len(all_errors)
            n_trim = int(n_samples * self.trim_percentage)
            if n_trim > 0 and n_samples > n_trim:
                sorted_errors = np.sort(all_errors)
                trimmed_errors = sorted_errors[:-n_trim]  # Remove the highest n_trim errors
                trimmed_mean = np.mean(trimmed_errors)
            else:
                trimmed_mean = mean_error
        else:
            trimmed_mean = mean_error
        
        # Percentiles
        percentile_90 = np.percentile(all_errors, 90)
        percentile_95 = np.percentile(all_errors, 95)
        
        return {
            'mean': mean_error,
            'median': median_error,
            'trimmed_mean': trimmed_mean,
            'percentile_90': percentile_90,
            'percentile_95': percentile_95
        }

    def set_adaptive_scales(self, initial_objectives):
        """
        Set scaling factors based on initial population statistics.
        This ensures all objectives are on similar scales for effective multi-objective optimization.
        """
        if initial_objectives.shape[0] < 10:  # Need enough samples
            # Use default scales if not enough samples
            self.objective_scales = np.array([1.0, 0.5, 1.0, 1.0, 0.01, 1.5])
            return
        
        # Calculate robust statistics for each objective
        scales = []
        for obj_idx in range(initial_objectives.shape[1]):
            obj_values = initial_objectives[:, obj_idx]
            # Remove infinities
            finite_values = obj_values[np.isfinite(obj_values)]
            if len(finite_values) > 0:
                # Use 75th percentile as scale to be robust to outliers
                scale = np.percentile(finite_values, 75)
                # Ensure scale is not too small
                scale = max(scale, 1e-6)
            else:
                # Default scale if all values are infinite
                scale = 1.0
            scales.append(scale)
        
        self.objective_scales = np.array(scales)
        print(f"Adaptive scales set: {self.objective_scales}")
        print(f"Scale details: MPJPE={scales[0]:.4f}, Norm={scales[1]:.4f}, Trimmed={scales[2]:.4f}, "
              f"Median={scales[3]:.4f}, Temporal={scales[4]:.6f}, 90th%ile={scales[5]:.4f}")

    def _evaluate(self, x_batch, out, *args, **kwargs):
        # x_batch is a 2D numpy array where each row is a flattened genome
        batch_size = x_batch.shape[0]
        
        # Initialize objective arrays
        f1_mpjpe_meters = np.zeros(batch_size)
        f2_mpjpe_normalized = np.zeros(batch_size)
        f3_trimmed_mpjpe = np.zeros(batch_size)
        f4_median_pjpe = np.zeros(batch_size)
        f5_temporal = np.zeros(batch_size)
        f6_percentile_90 = np.zeros(batch_size)

        for i in range(batch_size):
            flat_genome = x_batch[i, :]
            genome = Genome.from_flat_representation(flat_genome, self.source_dim, self.target_dim)
            
            # Collect errors for different metrics
            errors_normalized_all = []  # For normalized space
            errors_meters_all = []      # For actual meters
            temporal_errors_all = []
            
            num_valid_sequences = 0

            for source_seq_data, target_seq_data in zip(self.source_sequences, self.target_sequences):
                # Extract sequences
                source_seq = source_seq_data.sequence
                target_seq = target_seq_data.sequence
                
                if source_seq.shape[0] == 0 or target_seq.shape[0] == 0:
                    continue 

                num_frames = source_seq.shape[0]
                
                # Predict in normalized space
                predicted_target_seq_normalized = np.zeros_like(target_seq)
                for frame_idx in range(num_frames):
                    source_frame = source_seq[frame_idx, :, :]
                    predicted_target_seq_normalized[frame_idx, :, :] = transform_source_to_target(source_frame, genome)
                
                # Calculate errors in NORMALIZED space
                errors_normalized = np.linalg.norm(predicted_target_seq_normalized - target_seq, axis=2)  # (frames, keypoints)
                errors_normalized_all.append(errors_normalized)
                
                # Calculate errors in METERS (denormalize if needed)
                if source_seq_data.is_normalized and source_seq_data.normalization_factor is not None:
                    # Denormalize predictions and targets
                    predicted_target_seq_meters = predicted_target_seq_normalized * source_seq_data.normalization_factor
                    target_seq_meters = target_seq * target_seq_data.normalization_factor
                else:
                    # Already in meters
                    predicted_target_seq_meters = predicted_target_seq_normalized
                    target_seq_meters = target_seq
                
                errors_meters = np.linalg.norm(predicted_target_seq_meters - target_seq_meters, axis=2)  # (frames, keypoints)
                errors_meters_all.append(errors_meters)
                
                # Temporal consistency (in normalized space for consistency)
                if num_frames > 1:
                    delta_pred = np.diff(predicted_target_seq_normalized, axis=0)
                    delta_true = np.diff(target_seq, axis=0)
                    temporal_diff_errors = np.linalg.norm(delta_pred - delta_true, axis=2)
                    temporal_errors_all.append(temporal_diff_errors)
                
                num_valid_sequences += 1
            
            if num_valid_sequences > 0:
                # Calculate robust metrics for normalized errors
                metrics_normalized = self._calculate_robust_metrics(errors_normalized_all)
                f2_mpjpe_normalized[i] = metrics_normalized['mean']
                
                # Calculate robust metrics for errors in meters
                metrics_meters = self._calculate_robust_metrics(errors_meters_all)
                f1_mpjpe_meters[i] = metrics_meters['mean']
                f3_trimmed_mpjpe[i] = metrics_meters['trimmed_mean']
                f4_median_pjpe[i] = metrics_meters['median']
                f6_percentile_90[i] = metrics_meters['percentile_90']
                
                # Temporal consistency (average across sequences)
                if temporal_errors_all:
                    temporal_metrics = self._calculate_robust_metrics(temporal_errors_all)
                    f5_temporal[i] = temporal_metrics['mean']
                else:
                    f5_temporal[i] = 0.0  # No temporal error if single frame
            else:
                # No valid sequences
                f1_mpjpe_meters[i] = np.inf
                f2_mpjpe_normalized[i] = np.inf
                f3_trimmed_mpjpe[i] = np.inf
                f4_median_pjpe[i] = np.inf
                f5_temporal[i] = np.inf
                f6_percentile_90[i] = np.inf
        
        # Store unscaled objectives
        unscaled_objectives = np.column_stack([
            f1_mpjpe_meters,
            f2_mpjpe_normalized,
            f3_trimmed_mpjpe,
            f4_median_pjpe,
            f5_temporal,
            f6_percentile_90
        ])
        
        # Initialize adaptive scaling on first evaluation with enough samples
        if self.use_adaptive_scaling and self.objective_scales is None and self._eval_count == 0 and batch_size >= 10:
            self.set_adaptive_scales(unscaled_objectives)
        self._eval_count += 1
        
        # Apply scaling
        if self.objective_scales is not None:
            scaled_objectives = unscaled_objectives / self.objective_scales
        else:
            # Default scaling if adaptive scaling not yet initialized
            default_scales = np.array([1.0, 0.5, 1.0, 1.0, 0.01, 1.5])
            scaled_objectives = unscaled_objectives / default_scales
        
        # Ensure no NaN or Inf in scaled objectives
        scaled_objectives = np.nan_to_num(scaled_objectives, nan=1e6, posinf=1e6, neginf=-1e6)
        
        out["F"] = scaled_objectives
        
        # Store unscaled values for later retrieval
        if hasattr(self, '_store_unscaled') and self._store_unscaled:
            out["F_unscaled"] = unscaled_objectives