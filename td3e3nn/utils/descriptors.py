import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.structure import (
    SineCoulombMatrix,
    DensityFeatures
)

class HybridRLFeaturizer:
    def __init__(self):
        # 1. SCM (不Flatten)
        self.scm = SineCoulombMatrix(flatten=False)
        # 2. 宏观密度
        self.density = DensityFeatures()
        # 3. 局部拓扑
        self.voro = VoronoiNN(weight='solid_angle', tol=0.1)

        self.output_dim = None
        self._is_fitted = False

    def fit_and_get_dim(self, example_structure: Structure):
        """初始化并锁定维度"""
        print("\n[Featurizer] Fitting...")
        try:
            self.scm.fit([example_structure])
            self._is_fitted = True

            t = self.get_features(example_structure)
            self.output_dim = t.shape[1]

            print(f"✅ Featurizer Ready!")
            print(f"   Structure Atoms: {len(example_structure)}")
            print(f"   Fixed Feature Dim: {self.output_dim} (Size-Invariant)")
            return self.output_dim
        except Exception as e:
            print(f"❌ Init Failed: {e}")
            return 0

    def get_features(self, structure: Structure) -> torch.Tensor:
        if not self._is_fitted:
            self.scm.fit([structure])
            self._is_fitted = True

        # --- A. Density (Fixed: 3) ---
        try:
            feat_density = self.density.featurize(structure)
        except:
            feat_density = [0.0, 0.0, 0.0]

        # --- B. SCM Stats (Fixed: 4) ---
        try:
            scm_matrix = self.scm.featurize(structure)[0]
            eig_vals = np.linalg.eigvalsh(scm_matrix)
            feat_scm = [
                np.mean(eig_vals), np.std(eig_vals),
                np.max(eig_vals), np.min(eig_vals)
            ]
        except:
            feat_scm = [0.0] * 4

        # --- C. Voronoi Stats (Fixed: 8) ---
        n_atoms = len(structure)
        voro_vols, voro_cns = [], []

        for i in range(n_atoms):
            try:
                nn_info = self.voro.get_nn_info(structure, i)
                vol, cn = 0.0, 0.0
                if nn_info:
                    for neighbor in nn_info:
                        area = neighbor['poly_info']['area']
                        face_dist = neighbor['poly_info']['face_dist']
                        vol += (1.0/3.0) * area * face_dist
                        cn += neighbor['weight']
                voro_vols.append(vol)
                voro_cns.append(cn)
            except:
                voro_vols.append(0.0)
                voro_cns.append(0.0)

        v_vols = np.array(voro_vols)
        v_cns = np.array(voro_cns)

        feat_voro = [
            np.mean(v_vols), np.std(v_vols), np.max(v_vols), np.min(v_vols),
            np.mean(v_cns),  np.std(v_cns),  np.max(v_cns),  np.min(v_cns)
        ]

        # --- D. Concatenate (3+4+8 = 15) ---
        all_feats = np.concatenate([feat_density, feat_scm, feat_voro])

        # 清洗 NaN
        clean_feats = np.nan_to_num(all_feats.astype(np.float32), nan=0.0, posinf=100.0, neginf=-100.0)

        return torch.tensor(clean_feats).unsqueeze(0)