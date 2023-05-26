import os
import yaml
from itertools import combinations_with_replacement
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.utils import get_element_list, r_cut
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
            pymatgen_structures.append(Structure(lattice,spcs,crdnts,coords_are_cartesian=True))
        except:
            pass
    return pymatgen_structures

def get_confs_from_list(struct_list, bc_list, energy_list, force_list): # structure as dict, energy in eV
    confs = []
    for s_i in range(len(struct_list)):
        tmp_dict = {}
        tmp_dict = {'conf':{}}
        lattice = Structure.from_dict(struct_list[s_i]).lattice.matrix
        sites   = Structure.from_dict(struct_list[s_i]).sites
        energy  = energy_list[s_i]
        tmp_dict['conf']['bc'] = bc_list[s_i]
        tmp_dict['conf']['cell'] = []
        tmp_dict['conf']['cell'].append([float(lattice[0][0]),float(lattice[0][1]),float(lattice[0][2])])
        tmp_dict['conf']['cell'].append([float(lattice[1][0]),float(lattice[1][1]),float(lattice[1][2])])
        tmp_dict['conf']['cell'].append([float(lattice[2][0]),float(lattice[2][1]),float(lattice[2][2])])
        tmp_dict['conf']['coord'] = []
        for i in range(len(sites)):
            elements = struct_list[s_i]['sites'][i]['species'][0]['element']
            tmp_dict['conf']['coord'].append([float(sites[i].x), float(sites[i].y), float(sites[i].z), elements, 'TTT'])
        tmp_dict['conf']['epot'] = energy*0.036749309
        if force_list:
            forces = force_list[s_i]
            tmp_dict['conf']['force'] = []
            for i in range(len(forces)):
                tmp_dict['conf']['force'].append([forces[i][0]*0.01944689673, forces[i][1]*0.01944689673, forces[i][2]*0.01944689673])
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
    method = settings.inputs['method']
    combinations_index = list(combinations_with_replacement(range(len(element_list)),2))
    for elmnt in element_list:
        fname = str(elmnt) + '.ann.input.yaml'
        with folder.open(fname, 'w', encoding='utf-8') as fhandle:
            fhandle.write('main:'+'\n')
            fhandle.write('    nodes: [{},{}]'.format(number_of_nodes,number_of_nodes)+'\n')
            fhandle.write('    rcut:        {}'.format(rcut)+'\n')
            fhandle.write('    ener_ref:    {}'.format(ener_ref)+'\n')
            fhandle.write('    method:      {}'.format(method)+'\n'+'\n')
            fhandle.write('symfunc:'+'\n')
            n = 0
            for g in g02_list:
                for i in range(len(element_list)):
                    n = n + 1
                    fhandle.write('    g02_{:03d}: {}    {}'.format(n,g,element_list[i])+'\n')
            n = 0
            for g in g05_list:
                for i in range(len(combinations_index)):
                    n = n + 1
                    fhandle.write('    g05_{:03d}: {}    {}    {}'.format(n,g,element_list[combinations_index[i][0]],\
                                                                   element_list[combinations_index[i][1]])+'\n')
