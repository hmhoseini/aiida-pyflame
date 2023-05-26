import os
import yaml
import json
from collections import defaultdict
from aiida.orm import Group
from aiida_pyflame.codes.utils import is_structure_valid
from aiida_pyflame.codes.flame.core import conf2pymatgenstructure
from aiida_pyflame.workflows.core import log_write
from aiida_pyflame.workflows.settings import inputs, output_dir

def collect_random_structures(outfile):
    random_bulk_structures = []
    min_d_prefactor = inputs['min_distance_prefactor']
    try:
        confs = yaml.load_all(outfile, Loader=yaml.SafeLoader)
    except:
        return None, None
    pymatgen_structures = conf2pymatgenstructure(confs)
    nat = len(pymatgen_structures[0].sites)
    for a_pymatgen_structure in pymatgen_structures:
        if is_structure_valid(a_pymatgen_structure, min_d_prefactor, True, False):
            random_bulk_structures.append(a_pymatgen_structure.as_dict())
    return nat, random_bulk_structures

def store_gensymcrys_structures():
    todump_dict = defaultdict(list)
    wf_gensymcrys_group = Group.get(label='wf_gensymcrys')
    for a_wf_node in wf_gensymcrys_group.nodes:
        a_node = a_wf_node.called[-1]
        if not a_node.is_finished_ok:
            continue
        output_folder = a_node.outputs.retrieved
        with output_folder.open('posout.yaml', 'rb') as fhandle:
            nat, random_bulk_structures = collect_random_structures(fhandle)
        if nat and random_bulk_structures:
            todump_dict[nat].extend(random_bulk_structures)
    with open(os.path.join(output_dir,'random_bulk_structures.json'), 'w', encoding='utf-8') as fhandle:
        json.dump(todump_dict, fhandle)
    for keys in todump_dict.keys():
        log_write('{} random bulk structures with {} atoms are generated'.format(len(todump_dict[keys]), keys)+'\n')
