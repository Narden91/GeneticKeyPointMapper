import numpy as np
from pymoo.core.problem import Problem
from .genome_and_transform import Genome, transform_source_to_target

class PoseRetargetingProblem(Problem):
    def __init__(self, 
                 source_sequences_train, # list of (frames, num_source_kp, 3)
                 target_sequences_train, # list of (frames, num_target_kp, 3)
                 source_dim_config,      # (num_source_kp, 3)
                 target_dim_config,      # (num_target_kp, 3)
                 genome_param_bounds):   # Dict with 'C1_bounds', 'S_bounds', 'B_bounds'
        
        self.source_sequences = source_sequences_train
        self.target_sequences = target_sequences_train
        self.source_dim = source_dim_config 
        self.target_dim = target_dim_config
        
        num_source_kp, _ = self.source_dim
        num_target_kp, _ = self.target_dim

        # Calculate total number of variables in the flattened genome
        n_var_C1 = num_target_kp * num_source_kp
        n_var_S = num_target_kp * 3
        n_var_B = num_target_kp * 3
        self.n_var_total = n_var_C1 + n_var_S + n_var_B

        # Define lower and upper bounds for variables
        # C1 elements: e.g. [0, 1]
        # S elements: e.g. [0.5, 1.5]
        # B elements: e.g. [-0.2, 0.2]
        c1_low, c1_high = genome_param_bounds['C1_init_bounds']
        s_low, s_high = genome_param_bounds['S_init_bounds']
        b_low, b_high = genome_param_bounds['B_init_bounds']

        xl_list = ([c1_low] * n_var_C1) + ([s_low] * n_var_S) + ([b_low] * n_var_B)
        xu_list = ([c1_high] * n_var_C1) + ([s_high] * n_var_S) + ([b_high] * n_var_B)
        
        super().__init__(n_var=self.n_var_total, 
                         n_obj=2,  # Accuracy (MPJPE) and Temporal Consistency
                         n_constr=0,
                         xl=np.array(xl_list, dtype=np.float32),
                         xu=np.array(xu_list, dtype=np.float32))

    def _evaluate(self, x_batch, out, *args, **kwargs):
        # x_batch is a 2D numpy array where each row is a flattened genome
        batch_size = x_batch.shape[0]
        
        # Objectives: f1 for MPJPE, f2 for Temporal Consistency
        f1_values = np.zeros(batch_size)
        f2_values = np.zeros(batch_size)

        for i in range(batch_size):
            flat_genome = x_batch[i, :]
            genome = Genome.from_flat_representation(flat_genome, self.source_dim, self.target_dim)
            
            total_mpjpe_for_genome = 0.0
            total_temporal_error_for_genome = 0.0
            num_valid_sequences = 0

            for source_seq, target_seq in zip(self.source_sequences, self.target_sequences):
                if source_seq.shape[0] == 0 or target_seq.shape[0] == 0:
                    continue 

                num_frames = source_seq.shape[0]
                predicted_target_seq = np.zeros_like(target_seq)

                for frame_idx in range(num_frames):
                    source_frame = source_seq[frame_idx, :, :]
                    predicted_target_seq[frame_idx, :, :] = transform_source_to_target(source_frame, genome)
                
                # 1. Accuracy Objective (MPJPE) for this sequence
                # ||Q_hat_i - Q_i||^2  -> then sqrt for L2 norm, then mean
                # MPJPE: Mean Per Joint Position Error
                # L2 norm for each keypoint, then mean over keypoints, then mean over frames
                errors_per_keypoint_per_frame = np.linalg.norm(predicted_target_seq - target_seq, axis=2) # (frames, target_kp)
                mpjpe_seq = np.mean(errors_per_keypoint_per_frame) # Mean over all keypoints and all frames for this sequence
                total_mpjpe_for_genome += mpjpe_seq

                # 2. Temporal Consistency Objective for this sequence
                if num_frames > 1:
                    # Δ(Q_hat) and Δ(Q)
                    delta_pred = np.diff(predicted_target_seq, axis=0) # (frames-1, target_kp, 3)
                    delta_true = np.diff(target_seq, axis=0)       # (frames-1, target_kp, 3)
                    
                    # ||Δ(Q_hat) - Δ(Q)||^2 -> then sqrt for L2 norm, then mean
                    temporal_diff_errors = np.linalg.norm(delta_pred - delta_true, axis=2) # (frames-1, target_kp)
                    temporal_error_seq = np.mean(temporal_diff_errors) # Mean over all keypoints and all frames for this sequence
                    total_temporal_error_for_genome += temporal_error_seq
                
                num_valid_sequences += 1
            
            if num_valid_sequences > 0:
                f1_values[i] = total_mpjpe_for_genome / num_valid_sequences
                f2_values[i] = total_temporal_error_for_genome / num_valid_sequences
            else: # Should not happen if data loader provides valid sequences
                f1_values[i] = np.inf 
                f2_values[i] = np.inf
        
        out["F"] = np.column_stack([f1_values, f2_values])