import os
import json
from itertools import combinations_with_replacement
import yaml
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.utils import get_element_list
import aiida_pyflame.workflows.settings as settings

def conf2pymatgenstructure(confs):
    pymatgen_structures = []
    for a_conf in confs:
        lattice = a_conf['conf']['cell']
        crdnts = []
        spcs = []
        for coord in a_conf['conf']['coord']:
            crdnts.append([coord[0],coord[1],coord[2]])
            spcs.append(coord[3])
        try:
            pymatgen_structures.append(Structure(lattice,spcs,crdnts,coords_are_cartesian=True, to_unit_cell=True))
        except:
            pass
    return pymatgen_structures

def r_cut():
    with open(os.path.join(settings.Flame_dir,'cycle-1','train','training_data.json'), 'r', encoding='utf8') as fhandle:
        data = json.loads(fhandle.read())
    all_rcuts = []
    structures = []
    for a_data in data:
        if a_data['bc'] == 'bulk':
            structures.append(a_data['structure'])
    n_atom_in_rcut = max(settings.inputs['bulk_number_of_atoms']) * 1.3
    for a_struct in structures:
        for d in range(6,30):
            if len(Structure.from_dict(a_struct).get_sites_in_sphere([0,0,0],d)) < n_atom_in_rcut:
                continue
            all_rcuts.append(d)
            break
    return sum(all_rcuts)/len(all_rcuts)

def get_confs_from_list(struct_list, bc_list, energy_list, force_list): # structure as dict, energy in eV
    confs = []
    for s_i in range(len(struct_list)):
        tmp_dict = {}
        tmp_dict = {'conf':{}}
        lattice = Structure.from_dict(struct_list[s_i]).lattice.matrix
        sites   = Structure.from_dict(struct_list[s_i]).sites
        tmp_dict['conf']['bc'] = bc_list[s_i]
        tmp_dict['conf']['cell'] = []
        tmp_dict['conf']['cell'] = lattice.tolist()
        tmp_dict['conf']['coord'] = []
        for i in range(len(sites)):
            element = struct_list[s_i]['sites'][i]['species'][0]['element']
            tmp_dict['conf']['coord'].append(sites[i].coords.tolist() + [element, 'TTT'])
        if energy_list:
            energy  = energy_list[s_i]
            tmp_dict['conf']['epot'] = energy*0.036749309
        if force_list:
            forces = force_list[s_i]
            tmp_dict['conf']['force'] = []
            for i in range(len(forces)):
                tmp_dict['conf']['force'].append([round(forces[i][0]*0.01944689673,14), round(forces[i][1]*0.01944689673,14), round(forces[i][2]*0.01944689673,14)])
        tmp_dict['conf']['nat'] = len(sites)
        tmp_dict['conf']['units_length'] = 'angstrom'
        confs.append(tmp_dict)
    return confs

def write_SE_ann_input(folder, cycle_number):
    rcut = r_cut() * 1.88973
    element_list = get_element_list()
    if cycle_number:
        with open(os.path.join(settings.Flame_dir,cycle_number,'train','ann_input.yaml'), 'r', encoding='utf-8') as fhandle:
            ann_input = yaml.safe_load(fhandle)
        c_no = int(cycle_number.split('-')[-1])
        number_of_nodes = settings.inputs['number_of_nodes'][c_no-1]
    else:
        with open(os.path.join(settings.PyFLAME_directory,'codes/flame/flame_files','ann_input.yaml'), 'r', encoding='utf-8') as fhandle:
            ann_input = yaml.safe_load(fhandle)
        number_of_nodes = 10

    g02_list = ann_input['g02']
    g05_list = ann_input['g05']

    ener_ref = 0.0
    method = 'behler'
    combinations_index = list(combinations_with_replacement(range(len(element_list)),2))
    for elmnt in element_list:
        fname = str(elmnt) + '.ann.input.yaml'
        with folder.open(fname, 'w', encoding='utf-8') as fhandle:
            fhandle.write('main:'+'\n')
            fhandle.write(f'    nodes: [{number_of_nodes},{number_of_nodes}]'+'\n')
            fhandle.write(f'    rcut:        {rcut}'+'\n')
            fhandle.write(f'    ener_ref:    {ener_ref}'+'\n')
            fhandle.write(f'    method:      {method}'+'\n'+'\n')
            fhandle.write('symfunc:'+'\n')
            n = 0
            for g in g02_list:
                for i in range(len(element_list)):
                    n = n + 1
                    fhandle.write(f'    g02_{n:03d}: {g}    {element_list[i]}'+'\n')
            n = 0
            for g in g05_list:
                for i in range(len(combinations_index)):
                    n = n + 1
                    fhandle.write(f'    g05_{n:03d}: {g}    {element_list[combinations_index[i][0]]}    {element_list[combinations_index[i][1]]}'+'\n')
