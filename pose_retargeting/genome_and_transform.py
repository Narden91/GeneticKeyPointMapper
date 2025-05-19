import numpy as np


class Genome:
    def __init__(self, source_dim, target_dim, C1_data=None, S_data=None, B_data=None):
        self.source_dim = source_dim # (num_source_keypoints, 3)
        self.target_dim = target_dim # (num_target_keypoints, 3)
        
        self.num_source_kp = source_dim[0]
        self.num_target_kp = target_dim[0]

        # C1: Correspondence matrix (target_kp, source_kp)
        # Each row C1[i,:] represents weights for target_kp[i] from all source_kp
        self.C1 = C1_data if C1_data is not None else np.random.rand(self.num_target_kp, self.num_source_kp)
        # Normalize rows of C1 to sum to 1
        row_sums = self.C1.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # Avoid division by zero if a row is all zeros
        self.C1 = self.C1 / row_sums
        
        # S: Scaling factors (target_kp, 3)
        self.S = S_data if S_data is not None else np.ones((self.num_target_kp, 3))
        
        # B: Bias vectors (target_kp, 3)
        self.B = B_data if B_data is not None else np.zeros((self.num_target_kp, 3))

    def get_flat_representation(self):
        return np.concatenate([
            self.C1.flatten(),
            self.S.flatten(),
            self.B.flatten()
        ])

    @classmethod
    def from_flat_representation(cls, flat_genome, source_dim, target_dim):
        num_source_kp, _ = source_dim
        num_target_kp, _ = target_dim

        len_C1 = num_target_kp * num_source_kp
        len_S = num_target_kp * 3
        # len_B = num_target_kp * 3 # B is the rest

        C1_flat = flat_genome[:len_C1]
        S_flat = flat_genome[len_C1 : len_C1 + len_S]
        B_flat = flat_genome[len_C1 + len_S:]

        C1_data = C1_flat.reshape(num_target_kp, num_source_kp)
        # Re-normalize C1 rows after potential mutations/crossover
        row_sums = C1_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 
        C1_data = C1_data / row_sums

        S_data = S_flat.reshape(num_target_kp, 3)
        B_data = B_flat.reshape(num_target_kp, 3)
        
        return cls(source_dim, target_dim, C1_data, S_data, B_data)

def transform_source_to_target(source_pose_frame: np.ndarray, genome: Genome):
    """
    Transforms a single source pose frame to a target pose frame using the genome.
    source_pose_frame: (num_source_keypoints, 3)
    genome: Instance of Genome class
    Returns: (num_target_keypoints, 3)
    """
    # P is source_pose_frame (num_source_kp, 3)
    # C1 is (num_target_kp, num_source_kp)
    # C1 @ P results in (num_target_kp, 3)
    
    # Ensure P is (num_source_kp, 3)
    if source_pose_frame.shape != (genome.num_source_kp, 3):
        raise ValueError(f"Source pose frame shape mismatch. Expected {(genome.num_source_kp, 3)}, got {source_pose_frame.shape}")

    transformed_points = genome.C1 @ source_pose_frame  # (target_kp, source_kp) @ (source_kp, 3) -> (target_kp, 3)
    
    # Apply scaling S and bias B
    # S is (target_kp, 3), B is (target_kp, 3)
    # Q = (C1 @ P) * S + B
    predicted_target_frame = transformed_points * genome.S + genome.B
    
    # anatomical_adjustment would be applied here in a more complex version
    # Q = predicted_target_frame + anatomical_adjustment(A, transformed_points)

    return predicted_target_frame.astype(np.float32)