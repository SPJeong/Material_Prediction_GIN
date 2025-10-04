##### graph_feature_extraction.py

import torch
import torch_geometric
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem


# import CONFIG # custom.py

def generate_mol_graph(mol, add_hydrogen=False, use_pos=False):
    """
    generate graph features from mol file.
        -- node attribute: [total  57-dim]
            # atom symbol check (27-dim)
            # check hybridization (7-dim)
            # degree (covalent bond) (6-dim)
            # number of hydrogens (5-dim)
            # chirality (4-dim)
            # formal charge (5-dim)
            # aromaticity (1-dim)
            # ring (1-dim)
            # radical electrons (1-dim)

        -- edge index

        -- edge attribute: [total  12-dim]
            # bond type (4-dim)
            # bondStereo (6-dim)
            # in ring (1-dim)
            # in conjugated bond (1-dim)

        --  edge_weight
            # bond type

        -- 3D position (optional): both arguments (add_hydrogen and use_pos) should be 'True'
    """
    if add_hydrogen:
        mol2 = Chem.AddHs(mol)  # optional for 3D coodination
    else:
        mol2 = mol  # normal case
    all_bonds = mol2.GetBonds()
    all_atoms = mol2.GetAtoms()
    n_bonds = len(all_bonds)
    n_atoms = len(all_atoms)

    ### node attribute
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Atom

    node_attr = []
    for atm_id in range(n_atoms):
        # select an atom
        atm = mol2.GetAtomWithIdx(atm_id)

        # atom symbol check (27-dim)
        valid_atoms = {'*': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 'P': 7, 'Cl': 8, 'Br': 9, 'B': 10,
                       'I': 11, 'H': 12, 'Li': 13, 'Na': 14,
                       'Ca': 15, 'Mg': 16, 'Cd': 17, 'Fe': 18, 'Co': 19, 'Pd': 20, 'Ag': 21, 'Ti': 22, 'Al': 23,
                       'Sn': 24, 'Se': 25, 'Unknown': 26}
        sym = atm.GetSymbol()
        atm_one_hot = [0] * len(valid_atoms)
        if sym not in valid_atoms:
            atm_one_hot[26] = 1
        else:
            idx = valid_atoms[sym]
            atm_one_hot[idx] = 1

        # check hybridization (7-dim)
        hybrid = atm.GetHybridization()
        hybrid_one_hot = [0] * 7
        if hybrid == Chem.HybridizationType.SP3:
            hybrid_one_hot[0] = 1
        elif hybrid == Chem.HybridizationType.SP2:
            hybrid_one_hot[1] = 1
        elif hybrid == Chem.HybridizationType.SP:
            hybrid_one_hot[2] = 1
        elif hybrid == Chem.HybridizationType.S:
            hybrid_one_hot[3] = 1
        elif hybrid == Chem.HybridizationType.SP3D:
            hybrid_one_hot[4] = 1
        elif hybrid == Chem.HybridizationType.SP3D2:
            hybrid_one_hot[5] = 1
        else:  # no hybridization or cannot define by hybrid
            hybrid_one_hot[6] = 1

        # degree (covalent bond) (6-dim, one_hot)
        # 0, 1, 2, 3, 4, >=5
        degree_one_hot = [0] * 6
        degree = atm.GetTotalDegree()
        if degree >= 5:
            degree_one_hot[5] = 1
        else:
            degree_one_hot[degree] = 1

        # number of hydrogens (5-dim, one_hot)
        num_h = atm.GetTotalNumHs()
        hydrogen_one_hot = [0] * 5
        if num_h >= 4:
            hydrogen_one_hot[4] = 1
        else:
            hydrogen_one_hot[num_h] = 1

        # chirality (4-dim, one_hot)
        chiral = atm.GetChiralTag()
        if chiral == Chem.rdchem.ChiralType.CHI_OTHER:
            chiral_one_hot = [1, 0, 0, 0]
        elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            chiral_one_hot = [0, 1, 0, 0]
        elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            chiral_one_hot = [0, 0, 1, 0]
        elif chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            chiral_one_hot = [0, 0, 0, 1]
        else:
            chiral_one_hot = [0, 0, 0, 0]

        # formal charge one-hot encoding (5-dim)
        formal_charge_set = [-2, -1, 0, 1, 2]
        formal_charge = atm.GetFormalCharge()
        charge_one_hot = [0] * len(formal_charge_set)
        if formal_charge in formal_charge_set:
            charge_idx = formal_charge_set.index(formal_charge)
            charge_one_hot[charge_idx] = 1
        elif formal_charge >= 2:
            charge_one_hot[len(formal_charge_set) - 1] = 1
        elif formal_charge <= -2:
            charge_one_hot[0] = 1
        else:
            charge_one_hot = [0, 0, 0, 0, 0]

        # is aromatic? (1-dim)
        if atm.GetIsAromatic():
            arom = 1
        else:
            arom = 0

        # is in ring? (1-dim)
        if atm.IsInRing():
            ring_flag = 1
        else:
            ring_flag = 0

        # radical electrons (1-dim)
        if atm.GetNumRadicalElectrons():
            radical_electron = 1
        else:
            radical_electron = 0

        attr = atm_one_hot + hybrid_one_hot + degree_one_hot + hydrogen_one_hot + chiral_one_hot + charge_one_hot + \
               [arom] + [ring_flag] + [radical_electron]  # [in total  57-dim] 27-dim, 7-dim, 6-dim, 5-dim, 4-dim, 5-dim, 1-dim, 1-dim, 1-dim

        # print(atm_id, attr)
        node_attr.append(attr)

    ### Edge attribute
    edge_index = []
    edge_attr = []
    edge_weight = []

    for idx in range(n_bonds):
        bond = mol2.GetBondWithIdx(idx)
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

        # bond type (4-dimensional one-hot)
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            bond_one_hot = [1, 0, 0, 0]
            edge_weight.extend([1.0, 1.0])
        elif btype == Chem.rdchem.BondType.AROMATIC:
            bond_one_hot = [0, 1, 0, 0]
            edge_weight.extend([1.5, 1.5])
        elif btype == Chem.rdchem.BondType.DOUBLE:
            bond_one_hot = [0, 0, 1, 0]
            edge_weight.extend([2.0, 2.0])
        elif btype == Chem.rdchem.BondType.TRIPLE:
            bond_one_hot = [0, 0, 0, 1]
            edge_weight.extend([3.0, 3.0])

        # bondStereo (6-dimensional one-hot)
        stype = bond.GetStereo()
        if stype == Chem.rdchem.BondStereo.STEREOANY:
            stereo_one_hot = [1, 0, 0, 0, 0, 0]
        elif stype == Chem.rdchem.BondStereo.STEREOCIS:
            stereo_one_hot = [0, 1, 0, 0, 0, 0]
        elif stype == Chem.rdchem.BondStereo.STEREOE:
            stereo_one_hot = [0, 0, 1, 0, 0, 0]
        elif stype == Chem.rdchem.BondStereo.STEREONONE:
            stereo_one_hot = [0, 0, 0, 1, 0, 0]
        elif stype == Chem.rdchem.BondStereo.STEREOTRANS:
            stereo_one_hot = [0, 0, 0, 0, 1, 0]
        elif stype == Chem.rdchem.BondStereo.STEREOZ:
            stereo_one_hot = [0, 0, 0, 0, 0, 1]
        else:
            stereo_one_hot = [0, 0, 0, 0, 0, 0]

        # is this bond included in a ring?
        if bond.IsInRing():
            ring_bond = 1
        else:
            ring_bond = 0

        # is this bond a conjugated bond
        if bond.GetIsConjugated():
            conjugate = 1
        else:
            conjugate = 0

        attr = bond_one_hot + stereo_one_hot + [ring_bond, conjugate]  # in total 12-dimensional edge attribute # bond type (4-dim), bond stereo (6-dim), (ring, conjugate)

        edge_attr.append(attr)
        edge_attr.append(attr)

    ### 3D coordination (optional)
    if use_pos:
        val = AllChem.EmbedMolecule(mol2)
        if val != 0:
            print(f"Error while generating 3D: {Chem.MoleToSmeils(mol)}")
            return None

        pos_list = []
        for atm_id in range(n_atoms):
            # get atomic position
            atm_pos = mol2.GetConformer(0).GetAtomPosition(atm_id)
            crd = [atm_pos.x, atm_pos.y, atm_pos.z]
            pos_list.append(crd)
        pos = torch.tensor(pos_list, dtype=torch.float)
    else:
        pos = None

    # Pytorch tensor conversion
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_attr = torch.tensor(node_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # (number of edges x 2)
    edge_index = edge_index.t().contiguous()  # (2 x number of edges)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # print(node_attr.shape)
    # print(node_attr)
    # print(edge_attr.shape)
    # print(edge_attr)
    # print(edge_index.shape)
    # print(edge_index)
    # print(pos.shape)
    # print(pos)

    return torch_geometric.data.Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
                                     edge_weight=edge_weight)


if __name__ == "__main__":
    # e.g.
    smi = '*CCCCCOCc1ccc(OCCC*)c(COC)c1'
    mol = Chem.MolFromSmiles(smi)
    test = generate_mol_graph(mol, add_hydrogen=False, use_pos=False)
    print(test)
