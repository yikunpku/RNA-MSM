from Bio.PDB.MMCIF2Dict import MMCIF2Dict

residue_lsit = ["A", "U", "C", "G"]


def get_seq_num(pdb_file, chain_id):
    pdb_file = pdb_file
    chian_id = chain_id
    mon_id_seq_nums = []
    records = MMCIF2Dict(pdb_file)
    for pdb_strand_id, mon_id, pdb_seq_num, pdb_ins_code, in zip(
            records["_pdbx_poly_seq_scheme.pdb_strand_id"],
            records["_pdbx_poly_seq_scheme.mon_id"],
            records["_pdbx_poly_seq_scheme.pdb_seq_num"],
            records["_pdbx_poly_seq_scheme.pdb_ins_code"],
    ):
        if pdb_strand_id == chain_id:
            mon_id_seq_nums.append((chain_id, mon_id, pdb_seq_num,pdb_ins_code))
    return mon_id_seq_nums


def make_nt_id_have_XN(mon_id_seq_nums):
    mon_id_seq_nums = mon_id_seq_nums
    nt_id_list = []
    for mon_id_seq_num in mon_id_seq_nums:
        chain_id, mon_id, seq_num,pdb_ins_code = mon_id_seq_num[0], mon_id_seq_num[1], mon_id_seq_num[2],mon_id_seq_num[3]
        if mon_id[0] in residue_lsit:
            nt_id = chain_id + "." + mon_id + str(seq_num)
            nt_id_list.append(nt_id)
    return nt_id_list


def make_nt_id(mon_id_seq_nums):
    mon_id_seq_nums = mon_id_seq_nums
    nt_id_list = []
    for mon_id_seq_num in mon_id_seq_nums:
        chain_id, mon_id, seq_num, pdb_ins_code = mon_id_seq_num[0], mon_id_seq_num[1], mon_id_seq_num[2],mon_id_seq_num[3]
        if pdb_ins_code ==".":
            nt_id = chain_id + "." + mon_id + str(seq_num)
        else:
            nt_id = chain_id + "." + mon_id + str(seq_num) + "^" + pdb_ins_code
        nt_id_list.append(nt_id)
    return nt_id_list


def check_missing_nt(sequence_nt, structure_nt):
    sequence_nt = sequence_nt
    dssr_nt = structure_nt
    missing_nt = []
    real_structure_nt = []
    for nt_id in sequence_nt:
        if nt_id not in dssr_nt:
            missing_nt.append(nt_id)
    for nt in sequence_nt:
        if nt in dssr_nt:
            real_structure_nt.append(nt)
    # print(missing_nt)
    return missing_nt  # , real_structure_nt
