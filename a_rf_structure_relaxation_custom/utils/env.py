from ase.optimize import BFGS
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.lj import LennardJones
import numpy as np
import torch
import pandas as pd
from utils.calcs_func import func_for_calc
from IPython.display import clear_output
from torch_geometric.data import Data


params = {
    "radius": 5, # cut-off radius
    "max_num_nbr": 30, # maximum number of neighbors to consider
    "dmin": 0, # min for Gaussian distance
    "dmax": 5, # max for Gaussian distance
    "step": 0.2 # step for Gaussian distance
}

def get_sturct_lib(name):
    df = pd.read_csv(name)
    lib = []
    for item, nsite in zip(df["structure"],df["nsites"]):
        struct = Structure.from_str(item, fmt= "cif")
        if nsite == 1:
            struct.make_supercell([2, 1, 1])
        if nsite> 4 :
            continue
        lib.append(struct)
    return lib


def get_sturct_lib_and_calcs_gen(name, interval = None):
    df = pd.read_csv(name)
    lib,calcs = [], []
    for item, nsite in zip(df["structure"],df["nsites"]):
        struct = Structure.from_str(item, fmt= "cif")
        if nsite == 1:
            struct.make_supercell([2, 1, 1])
        calcs.append(func_for_calc(struct))
        lib.append(struct)
    if interval is None:
        return lib, calcs
    else:
        return lib[interval[0]:interval[1]], calcs[interval[0]:interval[1]]


def get_func_true(state, rmin, rmax = params["radius"]):
    state.wrap()
    state_str = AseAtomsAdaptor.get_structure(state)
    rmax_cond = False
    n = len(state_str.sites)
    for i in range(n):
        for j in range(i, n):
            ind = 1 if i==j else None
            r = state_str.get_distance(i,j,ind)
            if r < rmin:
                return False
            if r <= rmax:
                rmax_cond = True
    return rmax_cond

def correct_action(cut_state, action, r0, counter_max = 20, eps = 1e-3, order = 4):

    cond1 = get_func_true(cut_state, r0)
    if cond1:
        return(0)
    else:
        counter, a_low, a_high = 0, 0, 1
        a_mid = (a_low + a_high)/2
        while round(a_high - a_low, order) > eps:
            state_ase = cut_state.copy()
            back_trans = -a_mid*action
            state_ase.translate(back_trans)
            state_ase.wrap()
            cond1 = get_func_true(state_ase, r0)
            if cond1:
                a_high = a_mid
            else:
                a_low = a_mid
            a_mid = (a_low + a_high)/2
            counter += 1
            if counter > counter_max:
                break
    return(a_high)



class Environment:

    r"""The class for Environment that describes structure relaxation process, implemented according to the standards of the Gym Python library.

    Parameters
    ----------
    input_struct_lib : list
        A list containing input structures to be relaxed.

    calculator_lib:
        A list containing potentials for energy and force calculations, implemented as a class Calculators in the ASE library.

    reward_func: str
        Reward function type. Possible options: force, log_force, step, hybrid.

    convert_to_graph_func:
        Function used to convert pymatgen Structure into Crystal graph.

    r0: float
        Minimum possible distance between atoms. During relaxation, atoms shift backwards if the distance between them is less than r0.

    eps: float
        Force threshold, eV/A.

    stop_numb: int
        Maximum number of relaxation steps during which the Agent is allowed to perform actions that are corrected at each step, because the atoms are shifted too close to each other or too far apart.
        The minimum distance is controlled by the r0 parameter, and the maximum distance is adjusted by the params["radius"] parameter used in convert_to_graph_func.

    r_weights: list
        Weights for the hybrid reward function.
    """

    def __init__(self,
                 input_struct_lib,
                 convert_to_graph_func,
                 reward_func = "force",
                 calculator_lib = None,
                 r0:float = 0.1,
                 eps: float = 1e-6,
                 stop_numb: int = 50,
                 r_weights = None,
                ):

        self.to_graph = convert_to_graph_func

        if reward_func == "hybrid":
            assert r_weights is not None
            assert len(r_weights) == 3
            self.r_weights = r_weights
        self.eps = eps
        self.r0 = r0
        self.stop_count = 0
        self.stop_max_count = stop_numb
        self.reward_func = reward_func

        # Matching input structures and calculators
        if calculator_lib is None:
            calculator_lib = [LennardJones()]*len(input_struct_lib)
        self.input_lib = {}
        for it, struct_calc in enumerate(zip(input_struct_lib, calculator_lib)):
            print(f"\n--- 处理结构 {it} ---")
            struct, calc = struct_calc
            struct_ase = AseAtomsAdaptor.get_atoms(struct)
            struct_ase.calc = calc

            # Relaxation with BFGS so that structures in a data set of structures are in a local minimum
            relax = BFGS(struct_ase)
            relax.run(fmax=eps)
            self.input_lib[it] = [struct_ase, calc]

        self.len = it + 1
        self.num = 0
        self.current_structure = None
        self.current_ase_structure = None
        self.weights = None


    def reset(self, trans_coef, num = None, correct = True):
        r"""Reset current structure used for training of the RL Agent.

        Parameters
        ----------
        trans_coef : float
            Distortion parameter that determines how far the structure is shifted from the local minimum before the onset of the relaxation episode
        num : int
            Number of the structure from input dataset

        Returns
        ----------
        `torch_geometric.data.Data`
            Crystal graph
        """
        # Get structure for relaxation from the dataset
        self.num = num if num is not None else np.random.choice(self.len, 1, p=self.weights)[0]
        input_ase_struct, calc = self.input_lib[self.num]
        self.current_ase_structure = input_ase_struct.copy()

        # Shift of atoms from a local minimum before the onset of relaxation
        trans = trans_coef*np.random.rand(self.current_ase_structure.get_positions().shape[0], 3)
        self.current_ase_structure.translate(trans)

        # Correct the shift if it necessary
        if correct:
            a_back = correct_action(self.current_ase_structure, trans, self.r0)
            if a_back != 0:
                self.current_ase_structure.translate(-a_back*trans)

        self.current_structure = AseAtomsAdaptor.get_structure(self.current_ase_structure)
        self.current_ase_structure.calc = calc

        # Convert the structure into crystal graph
        forces = self.current_ase_structure.get_forces()
        struct_graph = self.to_graph(self.current_structure, forces)

        self.stop_count = 0
        return struct_graph

    def step(self, a, correct = True):
        r""" One step of structure relaxation

        Parameters
        ----------
        a : torch_geometric.data.Data
            Graph in which each node contains the atomic shift of the corresponding atom in the crystal graph of the current structure
        correct : bool
            Shows whether the action should be corrected if atoms are shifted too close to each other or too far apart

        Returns
        ----------
        o2 : `torch_geometric.data.Data`
            Crystal graph of the current structure at the next stage of relaxation

        r : `float`
            Immediate return

        d : `bool`
            Done flag indicating whether the relaxation is complete.

        a : `torch_geometric.data.Data`
            Graph of atomic shifts after correction

        max_f : `float`
            Maximum force acting on atoms

        s : `bool`
            Flag indicating whether relaxation should be stopped because the Agent predicts actions that should be corrected for more than stop_numb steps.

        """
        init_positions = self.current_ase_structure.get_positions()

        # Atomic shift
        self.current_ase_structure.translate(a.x.cpu())
        self.current_ase_structure.wrap()
        next_positions = self.current_ase_structure.get_positions()
        actual_action = next_positions - init_positions
        s = False

        # Correct the shift if it necessary
        if correct:
            a_back = correct_action(self.current_ase_structure, actual_action, self.r0)
            if a_back != 0:
                self.current_ase_structure.translate(-a_back*actual_action)
                a.x += -a_back*torch.tensor(actual_action)
                if round(a_back, 4) == 1:
                    self.stop_count += 1
                    if self.stop_count > self.stop_max_count:
                        s = True
                        self.stop_count = 0
                else:
                    self.stop_count = 0

        # Convert the structure in the next step into crystal graph

        self.current_structure = AseAtomsAdaptor.get_structure(self.current_ase_structure)
        forces = self.current_ase_structure.get_forces()
        o2 = self.to_graph(self.current_structure, forces)

        # Set done flag
        max_f = max((forces**2).sum(axis=1)**0.5)
        d = max_f <= self.eps

        # Calculate reward
        if self.reward_func == "log_force":
            r = -np.log10(max_f)
        if self.reward_func == "force":
            r = -max_f
        if self.reward_func == "step":
            r = d-1
        # if self.reward_func == "hybrid":
        #     r1 = -max_f
        #     r2 = -np.log10(max_f)
        #     r3 = d-1
        #     r = self.r_weights[0]*r1+ self.r_weights[1]*r2 + self.r_weights[2]*r3
        if self.reward_func == "hybrid":
            # 1. 线性项 (Linear) - 负责高位压制
            # 权重给低一点(0.05)。当 F=50 时，贡献 -2.5。
            # 它的作用是：当力很大的时候，提供一个基础的下降方向。
            r1 = -max_f

            # 2. 对数项 (Shifted Log) - 负责中低位冲刺
            # 权重给高(1.0)。目标是 0.01。
            # 当 F=50 时，log10(5000)=3.7。贡献 -3.7。
            # 当 F=1 时，log10(100)=2.0。贡献 -2.0。
            # 这里的技巧是：让 log 在 F=50 时和 linear 的贡献量级相当。
            target_val = 0.01
            # 加个极小值防止log报错
            r2 = -np.log10(max(max_f, 1e-7) / target_val)

            # 3. 步数 (Step)
            # 降低步数惩罚权重到 0.5。
            # 既然不限制步长，智能体可能需要更多步数来微调，别催太急。
            # 但给成功一个大奖励 (+10) 依然很有必要。
            if d:
                r3 = 10.0
            else:
                r3 = -1.0 # 保持为负，维持生存压力

            # 组合
            # r_weights = [0.5, 2, 1]
            r = self.r_weights[0]*r1 + self.r_weights[1]*r2 + self.r_weights[2]*r3

        return o2, r, d, a, max_f, s


    def fake_step(self):
        r""" Fake structure relaxation step, where the action is the shift of the atoms directly to the local minimum found by BFGS optimizer.

        Returns
        ----------
        o2_f : `torch_geometric.data.Data`
            Crystal graph of the current structure at the next stage of relaxation

        r_f : `float`
            Immediate return

        d_f : `bool`
            Done flag indicating whether the relaxation is complete.

        a_f : `torch_geometric.data.Data`
            Graph of atomic shifts after correction

        max_f : `float`
            Maximum force acting on atoms

        """
        prev_pos = self.current_ase_structure.get_positions()

        # Relaxation with BFGS
        prev_state = self.current_ase_structure.copy()
        prev_state.calc = self.current_ase_structure.calc
        dyn = BFGS(prev_state)
        dyn.run(fmax=self.eps, steps=100)
        clear_output(wait=True)

        # Calculate action, forces, next state
        a_f = Data(x=torch.from_numpy(prev_state.get_positions() - prev_pos))
        forces = prev_state.get_forces()
        prev_state_str = AseAtomsAdaptor.get_structure(prev_state)
        o2_f = self.to_graph(prev_state_str, forces)

        # Set done flag
        max_f = max((forces**2).sum(axis=1)**0.5)
        d_f = max_f <= self.eps

        # Calculate reward
        if self.reward_func == "log_force":
            r_f = -np.log10(max_f)
        if self.reward_func == "force":
            r_f = -max_f
        if self.reward_func == "step":
            r_f = d_f-1
        if self.reward_func == "hybrid":
            r1 = -max_f
            target_val = 0.01
            r2 = -np.log10(max(max_f, 1e-7) / target_val)
            if d_f:
                r3 = 10.0
            else:
                r3 = -1.0

            r_f = self.r_weights[0]*r1+ self.r_weights[1]*r2 + self.r_weights[2]*r3
        if self.reward_func == "energy":
            r_f = -self.current_ase_structure.get_potential_energy()

        return o2_f, r_f, d_f, a_f, max_f

