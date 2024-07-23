import os
import json
import yaml
from random import sample
from pymatgen.core.structure import Molecule
from aiida.orm import Group
from aiida_pyflame.codes.flame.core import get_confs_from_list, write_SE_ann_input, conf2pymatgenstructure
from aiida_pyflame.workflows.core import log_write
from aiida_pyflame.workflows.settings import inputs, Flame_dir

def write_divcheck_files(folder, cycle_number, nat, bc):
    if 'bulk' in bc:
        bulk_structures, energies = get_bulk_structures(cycle_number, nat)
        todump = get_confs_from_list(bulk_structures, len(bulk_structures)*['bulk'], energies, False)
    if 'free' in bc:
        cluster_structures, energies = get_cluster_structures(cycle_number, nat)
        todump = get_confs_from_list(cluster_structures, len(cluster_structures)*['free'], energies, False)
    with folder.open('position_force_divcheck.yaml', 'w', encoding='utf-8') as fhandle:
        yaml.dump_all(todump, fhandle, default_flow_style=None)
    with folder.open('list_posinp_check.yaml', 'w', encoding='utf-8') as fhandle:
        fhandle.write('files:'+'\n')
        fhandle.write(' - position_force_divcheck.yaml'+'\n')
    write_SE_ann_input(folder, None)
    with folder.open('data.dat', 'w', encoding='utf-8') as fhandle:
        fhandle.write(str(nat)+'\n')
        fhandle.write(str(len(todump)))
    provenance_exclude_list = ['position_force_divcheck.yaml']
    return provenance_exclude_list

def divcheck_report():
    wf_divcheck_group = Group.collection.get(label='wf_divcheck')
    for a_wf_node in wf_divcheck_group.nodes:
        a_node = a_wf_node.called[-1]
        if a_node.is_finished_ok:
            nat = a_node.outputs.output_parameters['nat']
            nposin = a_node.outputs.output_parameters['nposin']
            nposout = a_node.outputs.output_parameters['nposout']
            if 'bulk' in a_node.label:
                log_write(f'Diversity check for bulk structures with {nat} atoms: #posin {nposin}, #posout {nposout}'+'\n')
            if 'cluster' in a_node.label:
                log_write(f'Diversity check for clusters with {nat} atoms: #posin {nposin}, #posout {nposout}'+'\n')

def collect_divcheck_results():
    bulk_structures = []
    cluster_structures = []
    wf_divcheck_group = Group.collection.get(label='wf_divcheck')
    for a_wf_node in wf_divcheck_group.nodes:
        a_node = a_wf_node.called[-1]
        if a_node.is_finished_ok:
            pymatgen_structures = conf2pymatgenstructure(a_node.outputs.output_parameters['confs'])
            if 'bulk' in a_node.label:
                bulk_structures.extend(pymatgen_structures)
            if 'cluster' in a_node.label:
                for a_pymatgen_structure in pymatgen_structures:
                    cart_coords = a_pymatgen_structure.cart_coords
                    maxx = max(cart_coords[:,0:1])[0]
                    minx = min(cart_coords[:,0:1])[0]
                    maxy = max(cart_coords[:,1:2])[0]
                    miny = min(cart_coords[:,1:2])[0]
                    maxz = max(cart_coords[:,2:3])[0]
                    minz = min(cart_coords[:,2:3])[0]
                    a_cluster = maxx-minx+inputs['vacuum_length']
                    b_cluster = maxy-miny+inputs['vacuum_length']
                    c_cluster = maxz-minz+inputs['vacuum_length']
                    molecule = Molecule(a_pymatgen_structure.species, a_pymatgen_structure.cart_coords)
                    try:
                        boxed_molecule = molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster)
                    except:
                        continue
                    cluster_structures.append(boxed_molecule)
    return bulk_structures, cluster_structures

def get_bulk_structures(cycle_number, nat):
    c_no = int(cycle_number.split('-')[-1])
    nat = str(nat)
    poslow_structures = []
    trajectory_structures = []
    energies = []
    for c_no_i in range(1, c_no+1):
        if c_no_i == c_no and len(poslow_structures) >= 10000:
            log_write(f'>>> WARNING: number of poslow bulk structures with {nat} atoms from previous cycles is larger than 10000 <<<'+'\n')
            return None, None
        fpath = os.path.join(Flame_dir,'cycle-'+str(c_no_i),'minimahopping','poslows-'+'cycle-'+str(c_no_i)+'.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as fhandle:
                poslows = json.loads(fhandle.read())
            if nat in poslows.keys():
                poslow_structures.extend(poslows[nat])
                if c_no_i == c_no:
                    energies.extend(len(poslows[nat]) * [+1]) # it is a new structure
                else:
                    energies.extend(len(poslows[nat]) * [-1]) # it is NOT a new structure
            else:
                log_write(f'>>> WARNING: no poslow bulk structure with {nat} atoms (generated with minhocao) from cycle-{str(c_no_i)} is available <<<'+'\n')
        else:
            log_write(f'>>> WARNING: no poslow bulk structure (generated with minhocao) from cycle-{str(c_no_i)} is available <<<'+'\n')
        fpath = os.path.join(Flame_dir,'cycle-'+str(c_no_i),'minimahopping','poslows-bulk-'+'cycle-'+str(c_no_i)+'.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as fhandle:
                poslows = json.loads(fhandle.read())
            if nat in poslows.keys():
                poslow_structures.extend(poslows[nat])
                if c_no_i == c_no:
                    energies.extend(len(poslows[nat]) * [+1]) # it is a new structure
                else:
                    energies.extend(len(poslows[nat]) * [-1]) # it is NOT a new structure
            else:
                log_write(f'>>> WARNING: no poslow bulk structure with {nat} atoms (generated with bulk minhopp) from cycle-{str(c_no_i)} is available <<<'+'\n')
        else:
            log_write(f'>>> WARNING: no poslow bulk structure (generated with bulk minhopp) from cycle-{str(c_no_i)} is available <<<'+'\n')

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','posmds-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            posmds = json.loads(fhandle.read())
        if nat in posmds.keys():
            trajectory_structures.extend(posmds[nat])
        else:
            log_write(f'>>> WARNING: no bulk structure with {nat} atoms (generated with minhocao) from {str(cycle_number)} is available <<<'+'\n')
    else:
        log_write(f'>>> WARNING: no trajectory (generated with minhocao) from {str(cycle_number)} is available <<<'+'\n')

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-bulk-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            trajs = json.loads(fhandle.read())
        if nat in trajs.keys():
            trajectory_structures.extend(trajs[nat])
        else:
            log_write(f'>>> WARNING: no bulk structure with {nat} atoms (generated with bulk minhopp) from {str(cycle_number)} is available <<<'+'\n')
    else:
        log_write(f'>>> WARNING: no trajectory (generated with bulk minhopp) from {str(cycle_number)} is available <<<'+'\n')

    if len(poslow_structures) >= 10000:
        selected_poslow_structures = poslow_structures[:10000]
    else:
        selected_poslow_structures = poslow_structures
        if len(trajectory_structures) >= 10000-len(selected_poslow_structures):
            selected_trajectory_structures = sample(trajectory_structures, 10000-len(selected_poslow_structures))
        else:
            selected_trajectory_structures = trajectory_structures

    bulk_structures = selected_poslow_structures + selected_trajectory_structures
    energies = energies + len(selected_trajectory_structures) * [+1]
    return bulk_structures, energies

def get_cluster_structures(cycle_number, nat):
    c_no = int(cycle_number.split('-')[-1])
    nat = str(nat)
    poslow_structures = []
    trajectory_structures = []
    energies = []
    for c_no_i in range(1, c_no+1):
        if c_no_i == c_no and len(poslow_structures) >= 10000:
            log_write(f'>>> WARNING: number of poslow cluster structures with {nat} atoms from previous cycles is larger than 10000 <<<'+'\n')
            return None, None
        fpath = os.path.join(Flame_dir,'cycle-'+str(c_no_i),'minimahopping','poslows-cluster-'+'cycle-'+str(c_no_i)+'.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as fhandle:
                poslows = json.loads(fhandle.read())
            if nat in poslows.keys():
                poslow_structures.extend(poslows[nat])
                if c_no_i == c_no:
                    energies.extend(len(poslows[nat]) * [+1]) # it is a new structure
                else:
                    energies.extend(len(poslows[nat]) * [-1]) # it is NOT a new structure
            else:
                log_write(f'>>> WARNING: no poslow cluster structure with {nat} atoms (generated with cluster minhopp) from cycle-{str(c_no_i)} is available <<<'+'\n')
        else:
            log_write(f'>>> WARNING: no poslow cluster structure (generated with cluster minhopp) from cycle-{str(c_no_i)} is available <<<'+'\n')

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-cluster-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            trajs = json.loads(fhandle.read())
        if nat in trajs.keys():
            trajectory_structures.extend(trajs[nat])
        else:
            log_write(f'>>> WARNING: no cluster structure with {nat} atoms (generated with cluster minhopp) from {str(cycle_number)} is available <<<'+'\n')
    else:
        log_write(f'>>> WARNING: no trajectory from {str(cycle_number)} is available <<<'+'\n')

    if len(poslow_structures) >= 10000:
        selected_poslow_structures = poslow_structures[:10000]
    else:
        selected_poslow_structures = poslow_structures
        if len(trajectory_structures) >= 10000-len(selected_poslow_structures):
            selected_trajectory_structures = sample(trajectory_structures, 10000-len(selected_poslow_structures))
        else:
            selected_trajectory_structures = trajectory_structures

    cluster_structures = selected_poslow_structures + selected_trajectory_structures
    energies = energies + len(selected_trajectory_structures) * [+1]
    return cluster_structures, energies

def get_bulk_nats(cycle_number):
    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','nats_minhocao.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf8') as fhandle:
            minhocao_nats = json.loads(fhandle.read())
    else:
        minhocao_nats = {}
        minhocao_nats['bulk'] = []
    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','nats_minhopp.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf8') as fhandle:
            minhopp_nats = json.loads(fhandle.read())
    else:
        minhopp_nats = {}
        minhopp_nats['bulk'] = []
    bulk_nats = minhocao_nats['bulk']
    for a_nat in minhopp_nats['bulk']:
        if a_nat not in minhocao_nats['bulk']:
            bulk_nats.append(a_nat)
    return bulk_nats

def get_cluster_nats(cycle_number):
    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','nats_minhopp.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf8') as fhandle:
            minhopp_nats = json.loads(fhandle.read())
    else:
        minhopp_nats = {}
        minhopp_nats['cluster'] = []
    cluster_nats = minhopp_nats['cluster']
    return cluster_nats
