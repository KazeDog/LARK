import hashlib
import itertools
import os
from collections import defaultdict

import networkx as nx
import warnings

# # 忽略所有DeprecationWarning
# warnings.filterwarnings("ignore", category=DeprecationWarning, module='rdkit.*')

import pickle
import random
import sys
import lmdb
from loguru import logger
from unittest import TestCase

import math
import numpy as np
import pandas as pd
import torch
# from hypermol.utils.debug import profile
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, Mol
from rdkit.Chem import rdFingerprintGenerator
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


# from data_process import algos
from hypermol.data.edge_types import EDGE_TYPES_DICT
from hypermol.data.spatial_pos import floyd_warshall


def rd_chem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]

def safe_index(alist, elem):
    return alist.index(elem) if elem in alist else len(alist) - 1

def get_atom_feature_dims(list_acquired_feature_names):
    """tbd"""
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    """tbd"""
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]

class CompoundKit(object):
    # atom_vocab_dict = {
    #     "atomic_num": list(range(1, 119)) + ['misc'],
    #     "chiral_tag": rd_chem_enum_to_list(rdchem.ChiralType.values),
    #     'atom_is_in_ring': [0, 1],
    #     'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    # }
    # bond_vocab_dict = {
    #     "is_in_ring": [0, 1],
    # }
    # # float features
    # atom_float_names = ["van_der_waals_radis", 'mass']
    # # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    # atom_vocab_dict = {
    #     "atomic_num": list(range(1, 119)) + ['misc'],
    #     "chiral_tag": rd_chem_enum_to_list(rdchem.ChiralType.values),
    #     "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    #     "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
    #     "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    #     "hybridization": rd_chem_enum_to_list(rdchem.HybridizationType.values),
    #     "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
    #     "is_aromatic": [0, 1],
    #     "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
    #     'atom_is_in_ring': [0, 1],
    #     'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     # 'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     # 'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     # 'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     # 'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     # 'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     # 'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    #     # "function_group_index": list(range(0, 83)) + ['misc'],
    # }
    # bond_vocab_dict = {
    #     "bond_dir": rd_chem_enum_to_list(rdchem.BondDir.values),
    #     "bond_type": rd_chem_enum_to_list(rdchem.BondType.values),
    #     "is_in_ring": [0, 1],
    #     'bond_stereo': rd_chem_enum_to_list(rdchem.BondStereo.values),
    #     'is_conjugated': [0, 1],
    # }
    # # float features
    # atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']

    atom_context_features = {
        "chiral_tag": rd_chem_enum_to_list(rdchem.ChiralType.values),
        "is_aromatic": [0, 1],
        "atom_is_in_ring": [0, 1],
    }

    # 需要被协同Masking的原子类别特征
    atom_masked_features = {
        # 核心身份
        "atomic_num": list(range(1, 119)) + ['misc'],
        "valence_out_shell": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],

        # 电子/价态
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "num_radical_e": [0, 1, 2, 3, 4, 'misc'],
        "hybridization": rd_chem_enum_to_list(rdchem.HybridizationType.values),

        # 拓扑/价态强关联
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
    }

    # 需要被协同Masking的原子浮点数特征
    atom_float_masked_features = [
        "van_der_waals_radis",
        "partial_charge",
        "mass",
    ]

    bond_context_features = {
        "bond_type": rd_chem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
        "is_conjugated": [0, 1],
        "bond_dir": rd_chem_enum_to_list(rdchem.BondDir.values),
        "bond_stereo": rd_chem_enum_to_list(rdchem.BondStereo.values),
    }

    bond_masked_features = {}

    atom_vocab_dict = atom_context_features | atom_masked_features
    bond_vocab_dict = bond_context_features
    atom_float_names = atom_float_masked_features


    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    ### atom
    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        # elif name == 'implicit_valence':
        #     return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        # elif name == 'num_radical_e':
        #     return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        # elif name == 'valence_out_shell':
        #     return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        elif name == 'function_group_index':
            return CompoundKit.get_function_group_index(atom)
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### fingerprint

    morgan_fp_N = 1024
    morgan2048_fp_N = 2048
    rdkit_fp_N = 2048  # 许多位向量指纹的默认长度

    # =========================================================================
    # 新增: 将指纹生成器作为类属性一次性创建，以提高效率
    # =========================================================================

    # Morgan 指纹生成器 (替代 AllChem.GetMorganFingerprintAsBitVect)
    _morgan_gen_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan_fp_N)
    _morgan_gen_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan2048_fp_N)

    # RDKit/Layered 指纹生成器 (替代 AllChem.RDKFingerprint 和 AllChem.LayeredFingerprint)
    _rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=rdkit_fp_N)

    # 拓扑扭转角 (Torsion) 指纹生成器 (替代 GetHashedTopologicalTorsionFingerprintAsBitVect)
    _torsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=rdkit_fp_N)

    # 原子对 (Atom-Pair) 指纹生成器 (替代 GetHashedAtomPairFingerprintAsBitVect)
    _atom_pair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=rdkit_fp_N)

    # =========================================================================
    # 修改: 静态方法现在使用上面预先配置好的生成器
    # =========================================================================

    @staticmethod
    def get_morgan_fingerprint(mol):
        """获取 1024位的 Morgan 指纹"""
        # 核心修改点
        mfp = CompoundKit._morgan_gen_1024.GetFingerprint(mol)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol):
        """获取 2048位的 Morgan 指纹"""
        # 核心修改点
        mfp = CompoundKit._morgan_gen_2048.GetFingerprint(mol)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """获取 MACCS 指纹 (无需修改)"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()[1:]]

    @staticmethod
    def get_rdkit_fingerprint(mol):
        """获取 RDKit 指纹"""
        # 核心修改点
        fp = CompoundKit._rdkit_gen.GetFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def get_torsion_fingerprint(mol):
        """获取拓扑扭转角指纹"""
        # 核心修改点
        fp = CompoundKit._torsion_gen.GetFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def get_atom_pair_fingerprint(mol):
        """获取原子对指纹"""
        # 核心修改点
        fp = CompoundKit._atom_pair_gen.GetFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def get_layered_fingerprint(mol):
        """获取 Layered 指纹"""
        # 核心修改点
        fp = CompoundKit._rdkit_gen.GetFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)

            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),

            # 量子化学信息
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),

            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'mass': atom.GetMass(),
            # 量子化学信息
            'partial_charge': CompoundKit.check_partial_charge(atom),
        }
        return atom_names

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        # for i, atom in enumerate(mol.GetAtoms()):
        #     atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
        #         CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
        #     atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
        #         CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
        #     atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
        #         CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
        #     atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
        #         CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
        #     atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
        #         CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
        #     atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
        #         CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


# noinspection PyPep8Naming
class Compound3DKit(object):
    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    # noinspection SpellCheckingInspection
    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            # MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index_ = np.argmin([x[1] for x in res])
            energy = res[index_][1]
            conf = new_mol.GetConformer(id=int(index_))
        except Exception:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_MMFF_atom_poses_nonminimum(mol, numConfs=None, return_energy=False, percent=75):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs, randomSeed=42)
            # MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            energies = [x[1] for x in res]
            energy_threshold = np.percentile(energies, percent)
            closest_index = np.argmin(np.abs(np.array(energies) - energy_threshold))
            # index_ = np.argmin([x[1] for x in res])
            conf = new_mol.GetConformer(id=int(closest_index))
            energy = energies[closest_index]
        except Exception:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_super_edge_angles(edges, atom_poses, dir_type='HT'):

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0, ], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs

    @staticmethod
    def get_pair_distances(atom_poses):
        """get pair distance"""
        atom_number = len(atom_poses)
        pair_distances = []
        for i in range(atom_number):
            for j in range(atom_number):
                pair_distances.append(np.linalg.norm(atom_poses[i] - atom_poses[j]))
        pair_distances = np.array(pair_distances, 'float32')
        pair_distances = pair_distances.reshape(atom_number, atom_number)
        return pair_distances

    @staticmethod
    # def get_bond_angles_matrix(atom_poses, edges):
    #     """get triple angles"""
    #     edges_number = len(edges)
    #     edges_angles = np.zeros([edges_number, edges_number], 'float32')
    #     pair_vectors = np.zeros([edges_number, 3], 'float32')
    #     for i in range(edges_number):
    #         pair_vectors[i, :] = _get_vec_bond_angles(atom_poses, edges[i][0], edges[i][1])
    #     for i in range(edges_number):
    #         for j in range(edges_number):
    #             edges_angles[i, j] = _get_angle(pair_vectors[i], pair_vectors[j])
    #     return edges_angles

    def get_pair_angles(atom_poses, edges):
        """get triple angles"""
        edges_number = len(edges)
        edges_angles = np.zeros([edges_number, edges_number], 'float32')
        pair_vectors = np.zeros([edges_number, 3], 'float32')
        for i in range(edges_number):
            pair_vectors[i, :] = _get_vec_bond_angles(atom_poses, edges[i][0], edges[i][1])
        for i in range(edges_number):
            for j in range(edges_number):
                edges_angles[i, j] = _get_angle(pair_vectors[i], -pair_vectors[j])
        return edges_angles

    @staticmethod
    def get_bond_angles(atom_poses, angles_atom_index):
        """
        Calculates multiple bond angles efficiently using vectorization.
        This function replaces the combination of `_get_angle` and the looped `get_bond_angles`.

        Args:
            atom_poses (np.ndarray): Shape (N, 3), atomic coordinates.
            angles_atom_index (np.ndarray): Shape (M, 3), indices of atoms [i, j, k]
                                            where j is the central atom.

        Returns:
            np.ndarray: Shape (M,), list of bond angles in radians.
        """
        # 确保输入是NumPy数组，以便进行高级索引
        angles_atom_index = np.array(angles_atom_index, dtype=np.int64)

        # === 步骤 1: 一次性提取所有相关的原子坐标 ===
        # 这里的索引操作是向量化的核心。它会创建一个新的数组，
        # 包含了所有角度三元组 [i,j,k] 对应的坐标。
        p_i = atom_poses[angles_atom_index[:, 0]]  # 所有'i'原子的坐标，shape (M, 3)
        p_j = atom_poses[angles_atom_index[:, 1]]  # 所有'j'中心原子的坐标，shape (M, 3)
        p_k = atom_poses[angles_atom_index[:, 2]]  # 所有'k'原子的坐标，shape (M, 3)

        # === 步骤 2: 一次性计算所有的向量 ===
        # v_ji = p_i - p_j  # 对应你循环中的第一个参数
        # v_jk = p_k - p_j  # 对应你循环中的第二个参数
        v_ji = p_i - p_j
        v_jk = p_k - p_j

        # === 步骤 3: 向量化地计算点积和模长 ===
        # 使用 einsum 高效计算所有 M 个点积
        # 'ij,ij->i' 的意思是：对于两个 shape 为 (M, 3) 的数组，
        # 沿着第二个维度 (j=3) 进行元素乘积并求和，得到一个 shape 为 (M,) 的结果。
        # 这完全等价于逐行计算点积。
        dot_products = np.einsum('ij,ij->i', v_ji, v_jk)

        # 沿着第二个维度 (axis=1) 计算所有 M 个向量的模长
        norm_v_ji = np.linalg.norm(v_ji, axis=1)
        norm_v_jk = np.linalg.norm(v_jk, axis=1)

        # === 步骤 4: 向量化地处理除法和数值稳定性 ===
        # 对应你的 _get_angle 函数中的归一化和安全检查
        product_of_norms = norm_v_ji * norm_v_jk

        # 创建一个掩码，标记出那些模长乘积不为零（或大于一个很小的数）的位置
        # 这样可以避免除以零的警告和错误
        epsilon = 1e-6
        valid_mask = product_of_norms > epsilon

        # 初始化余弦值为0
        cos_angles = np.zeros_like(dot_products)

        # 只在有效的位置上执行除法
        cos_angles[valid_mask] = dot_products[valid_mask] / product_of_norms[valid_mask]

        # 使用 np.clip 来处理所有元素的浮点数精度问题，确保值在 [-1.0, 1.0] 范围内
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        # === 步骤 5: 一次性计算所有角度 ===
        angles_rad = np.arccos(cos_angles)

        return angles_rad

    @staticmethod
    def get_torsion_angles(atom_pos, edges):
        """
        计算与给定的有向边列表完全对齐的OmniMol扭转角。
        这个函数不关心边的方向是否经过排序，只要是确定的即可。

        参数:
            atom_pos (np.ndarray): 原子三维坐标，形状 (N, 3)。
            directed_edges (np.ndarray): 一个确定的、有向的边列表，形状 (num_edges, 2)。

        返回:
            np.ndarray: 一个一维数组，包含了每条有向边的扭转角值。
                        形状 (num_edges, )。
        """
        # ... (函数内部的代码与 get_omnimol_torsions_for_canonical_edges 完全相同) ...
        num_atoms = atom_pos.shape[0]
        num_edges = edges.shape[0]

        if num_atoms < 2 or num_edges == 0:
            return np.empty(0, dtype=np.float32)

        adjacency_list = defaultdict(list)
        for i, j in edges: # 注意这里假设输入已经是无向图的边
            adjacency_list[i].append(j)
            adjacency_list[j].append(i)

        pos_diff = atom_pos[np.newaxis, :, :] - atom_pos[:, np.newaxis, :]
        norms = np.linalg.norm(pos_diff, axis=-1, keepdims=True)
        direction_vectors = pos_diff / (norms + 1e-8)

        torsion_values = np.zeros(num_edges, dtype=np.float32)

        for edge_idx, (i, j) in enumerate(edges):
            i_neighbors = [k for k in adjacency_list[i] if k != j]
            j_neighbors = [k for k in adjacency_list[j] if k != i]

            if not i_neighbors:
                c_ij = np.zeros(3)
            else:
                d_ij = direction_vectors[i, j]
                d_ik_batch = direction_vectors[i, i_neighbors]
                c_ij = np.sum(np.cross(d_ij, d_ik_batch), axis=0)

            if not j_neighbors:
                c_ji = np.zeros(3)
            else:
                d_ji = direction_vectors[j, i]
                d_jk_batch = direction_vectors[j, j_neighbors]
                c_ji = np.sum(np.cross(d_ji, d_jk_batch), axis=0)

            y_comp = np.dot(np.cross(c_ij, c_ji), direction_vectors[i, j])
            x_comp = np.dot(c_ij, c_ji)
            torsion_values[edge_idx] = np.arctan2(y_comp, x_comp)

        return torsion_values

    @staticmethod
    def get_edge_poses(atom_poses, edges):
        edges_number = len(edges)

        bond_poses = []

        for i in range(edges_number):
            bond_poses.append((atom_poses[edges[i][0]] + atom_poses[edges[i][1]]) / 2)

        return bond_poses

    @staticmethod
    def get_edge_distances(atom_poses, edges):
        edges_number = len(edges)

        bond_poses = []
        bond_distances = []

        for i in range(edges_number):
            bond_poses.append((atom_poses[edges[i][0]] + atom_poses[edges[i][1]]) / 2)

        for i in range(edges_number):
            for j in range(edges_number):
                bond_distances.append(np.linalg.norm(bond_poses[i] - bond_poses[j]))

        bond_distances = np.array(bond_distances, 'float32')
        bond_distances = bond_distances.reshape(edges_number, edges_number)
        return bond_distances

    @staticmethod
    def get_atom_bond_distances(atom_poses, edges):
        atom_number = len(atom_poses)
        edges_number = len(edges)

        bond_poses = []
        atom_bond_distances = []

        for i in range(edges_number):
            bond_poses.append((atom_poses[edges[i][0]] + atom_poses[edges[i][1]]) / 2)

        for i in range(atom_number):
            for j in range(edges_number):
                atom_bond_distances.append(np.linalg.norm(atom_poses[i] - bond_poses[j]))

        atom_bond_distances = np.array(atom_bond_distances, 'float32')
        atom_bond_distances = atom_bond_distances.reshape(atom_number, edges_number)
        return atom_bond_distances

    @staticmethod
    def get_angles_list(bond_angles, angles_bond_index):
        angles_bond_index = np.array(angles_bond_index)
        if len(angles_bond_index.shape) == 0:
            return np.array([0])
        else:
            angles_list = bond_angles[angles_bond_index[:, 0], angles_bond_index[:, 1]]
            return angles_list

    @staticmethod
    def get_bond_length(pair_distances, edges):
        edges_number = len(edges)
        bond_distances = []
        for i in range(edges_number):
            bond_distances.append(pair_distances[edges[i][0]][edges[i][1]])
        bond_distances = np.array(bond_distances, 'float32')
        return bond_distances


def _get_vec(atom_poses, edge):
    return atom_poses[edge[1]] - atom_poses[edge[0]]


def _get_vec_bond_angles(atom_poses, i, j):
    return atom_poses[j] - atom_poses[i]


def _get_angle(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
    vec2 = vec2 / (norm2 + 1e-5)
    angle = np.arccos(np.dot(vec1, vec2))
    return angle


def find_angel_index(edges, atom_num):
    """
    Finds all unique angle indices [i, j, k] (j is central) efficiently.

    Args:
        edges (np.ndarray): Shape (M, 2), list of bonds.
        atom_num (int): Total number of atoms in the molecule.

    Returns:
        np.ndarray: Shape (num_angles, 3), a list of unique atom indices for angles.
    """
    # 1. Build an adjacency list
    # The keys are atom indices, values are sets of their neighbors.
    # Using a set for neighbors automatically handles duplicate edges if any.
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)  # Treat graph as undirected

    # 2. Find angles by iterating through each potential central atom
    angle_indices = []
    for center_atom_j in range(atom_num):
        neighbors = list(adj[center_atom_j])

        # If an atom has fewer than 2 neighbors, it cannot be a central atom for an angle.
        if len(neighbors) < 2:
            continue

        # 3. Generate all unique pairs of neighbors for the central atom
        # itertools.combinations handles uniqueness automatically (e.g., (i, k) is the same as (k, i))
        for neighbor_i, neighbor_k in itertools.combinations(neighbors, 2):
            # The angle is formed by atom_i - center_atom_j - atom_k
            angle_indices.append([neighbor_i, center_atom_j, neighbor_k])

    return np.array(angle_indices, dtype=np.int64)

def get_dist_bar(dist: np.ndarray, percentiles: list):
    # print(percentiles)
    try:
        result = np.percentile(dist, percentiles)
    except Exception as e:
        result = np.zeros(len(percentiles))

    return result

def get_spatial_pos(data_len, edges):
    adj = torch.zeros([data_len, data_len], dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True
    shortest_path_result, path = floyd_warshall(adj.numpy())
    spatial_pos = shortest_path_result
    # spatial_pos = torch.from_numpy(shortest_path_result).long()
    return spatial_pos

def get_edge_spatial_pos(data_len, edges):
    """
    计算药物图中每条边经过几个原子到达其他边。

    参数:
        data_len (int): 图中节点的总数。
        edges (torch.Tensor): 图的边信息，形状为 [num_edges, 2]，表示边的起点和终点。

    返回:
        edge_spatial_pos (numpy.ndarray): 一个二维矩阵，表示每条边之间经过的原子数量。
    """
    # Step 1: 构建原图的邻接矩阵
    adj = torch.zeros([data_len, data_len], dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True

    # Step 2: 构建线图的邻接矩阵
    num_edges = edges.shape[0]
    line_graph_adj = torch.zeros([num_edges, num_edges], dtype=torch.bool)

    for i in range(num_edges):
        for j in range(num_edges):
            if i != j:
                # 如果两条边共享一个公共节点，则在它们之间建立连接
                if edges[i, 0] in edges[j] or edges[i, 1] in edges[j]:
                    line_graph_adj[i, j] = True
                    line_graph_adj[j, i] = True

    # Step 3: 使用 Floyd-Warshall 算法计算线图中边之间的最短路径
    shortest_path_result, _ = floyd_warshall(line_graph_adj.numpy())

    # Step 4: 将结果转换为边之间的“原子数量”
    # 最短路径长度减去1，因为路径长度包括起点和终点
    # edge_spatial_pos = shortest_path_result - 1
    edge_spatial_pos = shortest_path_result
    edge_spatial_pos[np.isinf(edge_spatial_pos)] = -1  # 无路径的边对用 0 表示

    return edge_spatial_pos

def add_spatial_pos(data):
    data_len = len(data['atomic_num'])
    adj = torch.zeros([data_len, data_len], dtype=torch.bool)
    adj[data['edges'][:, 0], data['edges'][:, 1]] = True
    adj[data['edges'][:, 1], data['edges'][:, 0]] = True
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = shortest_path_result
    # spatial_pos = torch.from_numpy(shortest_path_result).long()
    # spatial_pos = set_up_spatial_pos(spatial_pos, up=20)
    data['spatial_pos'] = spatial_pos
    return data


def get_atom_poses(mol, skip_big_mol=True, skip_atom_num_bar=200):
    smiles = None
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        smiles = mol
    atom_n = len(mol.GetAtoms())
    if skip_big_mol and atom_n > skip_atom_num_bar:
        return None
    if atom_n <= 400:
        _, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return smiles, atom_poses

def get_bond_type_str(bond: Chem.Bond) -> str:
    a1 = bond.GetBeginAtom().GetSymbol()
    a2 = bond.GetEndAtom().GetSymbol()
    # Sort symbols to account for symmetry (e.g., C=O is the same as O=C)
    a1, a2 = sorted([a1, a2])

    # Determine bond type
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_str = f"{a1}-{a2}"
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_str = f"{a1}={a2}"
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_str = f"{a1}#{a2}"
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_str = f"{a1}~={a2}"
    else:
        bond_str = f"{a1} <{bond_type}> {a2}"
    return bond_str

def get_edge_types(mol) -> List[int]:
    """
    Args:
        mol:

    Returns:
        List of edge types. Len: edge_num
        each entry is the edge type index.
    """
    edge_types = []
    for idx, bond in enumerate(mol.GetBonds()):
        bond_str = get_bond_type_str(bond)
        edge_types.append(EDGE_TYPES_DICT.get(bond_str, EDGE_TYPES_DICT["others"]))
    return edge_types

def binning_matrix(matrix, m, range_min, range_max):
    # 计算每个 bin 的范围
    bin_width = (range_max - range_min) / m

    # 将矩阵中的元素映射到 bin 中
    bin_indices = np.floor((matrix - range_min) / bin_width).astype(int)

    # 将超出范围的值限制在范围内
    bin_indices = np.clip(bin_indices, 0, m - 1)

    return bin_indices

def compute_edge_path(
        num_atoms,
        edges,
        bond_features_map,
        bond_id_names,
        max_path_len
) :
    """
    计算并返回扁平化的 Graphormer 全局拓扑特征。
    """
    # ... (此函数的实现和我们之前讨论的一样) ...
    G = nx.Graph()
    G.add_nodes_from(range(num_atoms))
    G.add_edges_from(edges)

    try:
        paths = dict(nx.all_pairs_shortest_path(G, cutoff=max_path_len))
    except nx.NetworkXNoPath:
        paths = {}

    edge_path_dict = {
        name: np.zeros((num_atoms, num_atoms, max_path_len), dtype=np.int64)
        for name in bond_id_names
    }

    for i in range(num_atoms):
        if i not in paths: continue
        for j in range(num_atoms):
            if j not in paths[i]: continue

            path_nodes = paths[i][j]
            for k in range(len(path_nodes) - 1):
                if k >= max_path_len: break
                u, v = path_nodes[k], path_nodes[k+1]
                edge_key = tuple(sorted((u, v)))
                if edge_key in bond_features_map:
                    bond_features = bond_features_map[edge_key]
                    for name in bond_id_names:
                        edge_path_dict[name][i, j, k] = bond_features[name]

    flattened_data = {}
    for name, arr in edge_path_dict.items():
        flattened_data[f"{name}"] = arr

    return flattened_data


def get_bond_features_dict(mol, data):
    # (新增) 创建一个 map 用于快速查找边特征
    bond_features_map_for_graphormer = {}

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # i->j and j->i (如果需要双向边)
        # data['edges'].append((i, j))
        # data['edges'].append((j, i)) # GNN通常需要双向
        bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

        bond_feature_values = {}
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            # # 这一行是您原来的逻辑，保留它
            # data[name].append(bond_feature_id)
            # 同时存入新的 map
            bond_feature_values[name] = bond_feature_id

        # 为新的 map 存储特征
        bond_features_map_for_graphormer[tuple(sorted((i, j)))] = bond_feature_values

    return bond_features_map_for_graphormer

def add_edge_encoding(data):
    mol = data['mol']

    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    bond_features_map_for_graphormer = get_bond_features_dict(mol, data)

    MAX_PATH_LENGTH = 5  # 举例，可以根据您的数据集调整

    # 调用新函数来计算全局拓扑信息
    graphormer_topo = compute_edge_path(
        num_atoms=mol.GetNumAtoms(),
        edges=data['edges'],
        bond_features_map=bond_features_map_for_graphormer,
        bond_id_names=bond_id_names,
        max_path_len=MAX_PATH_LENGTH
    )
    data.update(graphormer_topo)
    return data

def get_be_matrix_with_map_info(mol_or_smiles, aromatic_mode="aromatic_1p5", strict_kekule=False):
    """
    计算给定分子（SMILES字符串或RDKit Mol对象）的BE矩阵。
    分子必须包含原子图谱序号。

    Args:
        mol_or_smiles (str or rdkit.Chem.rdchem.Mol): 包含原子图谱序号的SMILES字符串或RDKit Mol对象。

    Returns:
        tuple: (be_matrix, max_map_num, map_to_idx_dict)
               be_matrix (numpy.ndarray): 计算得到的BE矩阵。
               max_map_num (int): 分子中最大的原子图谱序号。
               map_to_idx_dict (dict): 原子图谱序号到RDKit原子索引的映射字典。
               如果输入无效或没有图谱序号，则返回 (None, 0, None)。
    """
    if aromatic_mode not in {"aromatic_1p5", "kekule"}:
        raise ValueError("aromatic_mode must be 'aromatic_1p5' or 'kekule'.")

    if isinstance(mol_or_smiles, str):
        # 从SMILES创建分子对象，sanitization=True是默认值，会计算价态等属性
        mol = Chem.MolFromSmiles(mol_or_smiles)
    elif isinstance(mol_or_smiles, Mol):
        mol = Chem.Mol(mol_or_smiles)
    else:
        raise TypeError("Input must be a SMILES string or an RDKit Mol object.")

    if not mol:
        # print(f"Warning: RDKit could not parse the input.")
        return None, 0, None

    if aromatic_mode == "kekule":
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            if strict_kekule:
                raise

    # 确保分子属性是最新的
    mol.UpdatePropertyCache(strict=False)

    mapped_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    if not mapped_atoms:
        # print("Warning: No mapped atoms found in the molecule.")
        return None, 0, None

    map_nums = [atom.GetAtomMapNum() for atom in mapped_atoms]
    max_map_num = max(map_nums)
    be_matrix = np.zeros((max_map_num, max_map_num), dtype=np.float32)
    rdkit_idx_to_map_num = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mapped_atoms}
    pt = Chem.GetPeriodicTable()

    bond_type_to_float = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
        Chem.BondType.AROMATIC: 1.5,
    }

    # 1. 填充非对角线元素 (化学键)
    for bond in mol.GetBonds():
        b_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b_idx in rdkit_idx_to_map_num and e_idx in rdkit_idx_to_map_num:
            map_i = rdkit_idx_to_map_num[b_idx]
            map_j = rdkit_idx_to_map_num[e_idx]
            bond_val = bond_type_to_float.get(bond.GetBondType(), 0.0)
            be_matrix[map_i - 1, map_j - 1] = bond_val
            be_matrix[map_j - 1, map_i - 1] = bond_val

    # 2. 填充对角线元素 (未共享价电子)
    # --- 这是修改的核心部分 ---
    for atom in mapped_atoms:
        map_num = atom.GetAtomMapNum()

        # 获取计算所需的所有参数
        atomic_num = atom.GetAtomicNum()
        total_valence_e = pt.GetNOuterElecs(atomic_num)
        explicit_valence = atom.GetExplicitValence()
        formal_charge = atom.GetFormalCharge()

        # 使用包含形式电荷的正确公式计算未共享电子数
        # Correct Formula: Lone Pair Electrons = Total Valence Electrons - Explicit Valence - Formal Charge
        num_lone_pair_e = total_valence_e - explicit_valence - formal_charge

        # 将计算结果填入矩阵
        be_matrix[map_num - 1, map_num - 1] = num_lone_pair_e

    return be_matrix, max_map_num, {atom.GetAtomMapNum(): atom.GetIdx() for atom in mapped_atoms}

def mol_to_data_pretrain(mol, smiles, pre_calculated_compose=None):
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    if len(mol.GetAtoms()) == 0:
        return None
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        if pre_calculated_compose is not None:
            mol, atom_poses = mol, pre_calculated_compose
        else:
            mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
            # mol, atom_poses = Compound3DKit.get_MMFF_atom_poses_nonminimum(mol, numConfs=50, percent=75)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)


    # atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    # bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    atom_id_names = (list(CompoundKit.atom_context_features.keys()) +
                     list(CompoundKit.atom_masked_features.keys()) +
                     CompoundKit.atom_float_masked_features)
    bond_id_names = list(CompoundKit.bond_context_features.keys())

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j
        data['edges'] += [(i, j)]
        # for name in bond_id_names:
        #     bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
        #     data[name] += [bond_feature_id]

    #### self loop
    if len(data['edges']) == 0:
        N = len(data[atom_id_names[0]])
        for i in range(N):
            data['edges'] += [(i, i)]
        for name in bond_id_names:
            bond_feature_id = get_bond_feature_dims([name])[0] - 1  # self loop: value = len - 1
            data[name] += [bond_feature_id] * N

    bond_features_map_for_graphormer = get_bond_features_dict(mol, data)

    MAX_PATH_LENGTH = 5  # 举例，可以根据您的数据集调整

    # 调用新函数来计算全局拓扑信息
    graphormer_topo = compute_edge_path(
        num_atoms=mol.GetNumAtoms(),
        edges=data['edges'],
        bond_features_map=bond_features_map_for_graphormer,
        bond_id_names=bond_id_names,
        max_path_len=MAX_PATH_LENGTH
    )
    data.update(graphormer_topo)



    # ### make ndarray and check length
    # for name in list(CompoundKit.atom_vocab_dict.keys()):
    #     data[name] = np.array(data[name], 'int64')
    # for name in CompoundKit.atom_float_names:
    #     data[name] = np.array(data[name], 'float32')
    # for name in bond_id_names:
    #     data[name] = np.array(data[name], 'int64')
    # data['edges'] = np.array(data['edges'], 'int64')

    for name in (list(CompoundKit.atom_context_features.keys()) +
                     list(CompoundKit.atom_masked_features.keys())):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_masked_features:
        data[name] = np.array(data[name], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')
    
    
    angles_atom_index = find_angel_index(data['edges'], len(atom_poses))
    data['angles_atom_index'] = angles_atom_index

    if len(angles_atom_index) == 0:
        data['angles_atom_index'] = np.array([[0, 0, 0]])

    atom_poses = np.array(atom_poses, 'float32')
    data['mol'] = mol

    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['rdkit_fp'] = np.array(CompoundKit.get_rdkit_fingerprint(mol), 'int64')
    data['torsion_fp'] = np.array(CompoundKit.get_torsion_fingerprint(mol), 'int64')
    data['atom_pair_fp'] = np.array(CompoundKit.get_atom_pair_fingerprint(mol), 'int64')
    data['layered_fp'] = np.array(CompoundKit.get_layered_fingerprint(mol), 'int64')

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['atom_distances_2d'] = get_spatial_pos(len(data['atom_pos']), data['edges'])

    data['bond_angles'] = Compound3DKit.get_bond_angles(atom_poses, data['angles_atom_index'])
    data['bond_angles_bin'] = binning_matrix(data['bond_angles'], 20, 0, math.pi)

    data['torsion_angles'] = Compound3DKit.get_torsion_angles(atom_poses, data['edges'])
    data['torsion_angles_bin'] = binning_matrix(data['torsion_angles'], 36, -math.pi, math.pi)

    be_matrix, max_map_num, map_list = get_be_matrix_with_map_info(mol)
    data['be_matrix'] = be_matrix
    data['max_map_num'] = max_map_num
    data['map_list'] = map_list

    return data

def process_pretrain_smiles(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is not None:
        data = mol_to_data_pretrain(mol, smiles)  # Assuming this function exists and is provided elsewhere
        data['smiles'] = smiles
        return data
    logger.error(f"Invalid smiles: {smiles}")
    return None

def process_smiles_parallel_streaming(
        txt_path: str,
        target_lmdb_path: str,
        num_limit=None,
        map_size=1500 * 1024 ** 3,
        num_cores=4,
        max_pending_tasks=1000,
        write_buffer_size=10000
):
    """
    从TXT文件【并行流式】处理SMILES数据，并使用基于总行数的平滑进度条。
    此版本解决了慢任务阻塞块的问题，并优化了内存使用。

    Args:
        txt_path (str): 输入的TXT文件路径。
        target_lmdb_path (str): 输出的LMDB文件路径。
        num_limit (int, optional): 最多处理的SMILES数量。
        map_size (int, optional): LMDB数据库的最大尺寸（字节）。
        num_cores (int, optional): 使用的CPU核心数。
        max_pending_tasks (int, optional): 内存中最多允许的待处理任务数。
                                          用于控制内存占用，建议设置为 num_cores 的数倍。
        write_buffer_size (int, optional): 结果缓冲区大小，达到此数量后批量写入LMDB。
    """
    parent_dir = os.path.dirname(target_lmdb_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    env = lmdb.open(target_lmdb_path, map_size=map_size, subdir=False, lock=False)
    successful_count = 0
    results_buffer = []

    print("正在计算文件总行数以设置进度条...")
    total_lines = 0
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        if num_limit is not None:
            total_lines = min(total_lines, num_limit)
        print(f"文件总行数: {total_lines:,}。")
    except Exception as e:
        print(f"无法计算总行数: {e}。进度条将不显示百分比。")
        total_lines = None  # 设为None，tqdm会自动处理

    # 批量写入函数
    def write_buffer_to_lmdb(buffer, current_count):
        if not buffer:
            return current_count
        with env.begin(write=True) as txn:
            for data_dict in buffer:
                smiles_string = data_dict['smiles']
                # print(smiles_string)

                # --- 关键修改：使用SMILES的哈希值作为key ---
                # 这可以保证key的长度是固定的，并且很短
                key = hashlib.sha256(smiles_string.encode('utf-8')).hexdigest().encode('utf-8')
                # print(key)
                # ---------------------------------------------

                value = pickle.dumps(data_dict)

                try:
                    txn.put(key, value)
                except lmdb.BadValsizeError as e:
                    print(f"\nERROR: lmdb.BadValsizeError for smiles: '{smiles_string}'")
                    print(f"Hashed key size: {len(key)} bytes (Limit is ~511 bytes)")
                    print(f"Value size: {len(value)} bytes (Limit is ~2GB)")
                    print(f"Original error: {e}")
                    continue

        new_count = current_count + len(buffer)
        buffer.clear()
        return new_count

    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            with open(txt_path, 'r', encoding='utf-8') as f:
                # 使用迭代器逐行读取，避免加载整个文件到内存
                smiles_iterator = (line.strip() for line in itertools.islice(f, num_limit) if line.strip())

                # 提交初始的一批任务
                pending_futures = {
                    executor.submit(process_pretrain_smiles, smiles)
                    for smiles in itertools.islice(smiles_iterator, max_pending_tasks)
                }

                with tqdm(total=total_lines, desc="处理进度") as pbar:
                    while pending_futures:
                        # 使用 as_completed 等待任何一个任务完成
                        for future in as_completed(pending_futures):
                            # 1. 从集合中移除已完成的任务
                            pending_futures.remove(future)

                            # 2. 尝试提交一个新任务来填补空位
                            try:
                                next_smiles = next(smiles_iterator)
                                new_future = executor.submit(process_pretrain_smiles, next_smiles)
                                pending_futures.add(new_future)
                            except StopIteration:
                                # 文件已读完，不再提交新任务
                                pass

                            # 3. 处理已完成任务的结果
                            result = future.result()
                            if result is not None:
                                results_buffer.append(result)

                            # 4. 更新进度条并检查是否需要写入缓冲区
                            pbar.update(1)
                            if len(results_buffer) >= write_buffer_size:
                                successful_count = write_buffer_to_lmdb(results_buffer, successful_count)
                                pbar.set_postfix(successful=f'{successful_count:,}')

                            # as_completed返回一个就break，重新进入循环等待下一个完成的任务
                            # 这确保了队列的持续填充
                            break

        # 处理所有任务完成后，确保缓冲区剩余数据也被写入
        if results_buffer:
            print(f"\n正在写入最后 {len(results_buffer):,} 条缓存数据...")
            successful_count = write_buffer_to_lmdb(results_buffer, successful_count)

        # 写入 __len__ 字段
        print("所有数据处理完毕，正在写入 '__len__' 字段...")
        with env.begin(write=True) as txn:
            len_key = b'__len__'
            len_value = pickle.dumps(successful_count)
            txn.put(len_key, len_value)

    finally:
        env.close()
        print(f"\n任务完成！共 {successful_count:,} 条有效数据已保存至 {target_lmdb_path}")

