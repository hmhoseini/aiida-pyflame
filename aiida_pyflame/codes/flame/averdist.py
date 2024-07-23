import os
import json
import yaml
from aiida.orm import Group
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.flame.core import get_confs_from_list, write_SE_ann_input
from aiida_pyflame.workflows.settings import output_dir, Flame_dir

def write_averdist_files(folder, nat):
    pymatgen_structures = []
    with open(os.path.join(output_dir,'seeds_bulk.json'), 'r', encoding='utf-8') as fhandle:
        structures = json.loads(fhandle.read())
    for a_structure in structures:
        a_pymatgen_structure = Structure.from_dict(a_structure)
        if len(a_pymatgen_structure) == nat:
            pymatgen_structures.append(a_pymatgen_structure.as_dict())
    todump = get_confs_from_list(pymatgen_structures, len(structures)*['bulk'], len(structures)*[0.01], False)
    with folder.open('position_force_divcheck.yaml', 'w', encoding='utf-8') as fhandle:
        yaml.dump_all(todump, fhandle, default_flow_style=None)
    with folder.open('list_posinp_check.yaml', 'w', encoding='utf-8') as fhandle:
        fhandle.write('files:'+'\n')
        fhandle.write(' - position_force_divcheck.yaml'+'\n')
    write_SE_ann_input(folder, None)
    with folder.open('nat.dat', 'w', encoding='utf-8') as fhandle:
        fhandle.write(str(nat))
    provenance_exclude_list = ['position_force_divcheck.yaml']
    return provenance_exclude_list

def store_averdist_results():
    wf_averdist_group = Group.collection.get(label='wf_averdist') 
    tmp_dict = {}
    for a_wf_node in wf_averdist_group.nodes:
        a_node = a_wf_node.called[-1]
        if not a_node.is_finished_ok:
            continue
        tmp_dict.update(a_node.outputs.output_parameters)
    with open(os.path.join(Flame_dir,'aver_dist.json'), 'w', encoding='utf-8') as fhandle:
        json.dump(tmp_dict, fhandle)
