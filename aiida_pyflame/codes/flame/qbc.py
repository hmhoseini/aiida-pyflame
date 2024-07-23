import os
import json
from random import sample
import numpy as np
import yaml
from pymatgen.core.structure import Structure
from aiida.orm import Group
from aiida_pyflame.codes.flame.core import get_confs_from_list, conf2pymatgenstructure
from aiida_pyflame.codes.utils import get_element_list
from aiida_pyflame.workflows.core import log_write
from aiida_pyflame.workflows.settings import Flame_dir, output_dir

def write_SP_files(folder, cycle_number, nat, job_type, SP_n):
    todump = []
    if 'ref' in job_type:
        structure, energy, bc = get_ref_data(cycle_number)
        todump = get_confs_from_list(structure, bc, energy, False)
    if 'bulk' in job_type:
        bulk_structures = get_bulk_structures(cycle_number, nat)
        todump = get_confs_from_list(bulk_structures, len(bulk_structures)*['bulk'], False, False)
    if 'free' in job_type:
        cluster_structures = get_cluster_structures(cycle_number, nat)
        todump = get_confs_from_list(cluster_structures, len(cluster_structures)*['free'], False, False)
    with folder.open('posinp.yaml', 'w', encoding='utf-8') as fhandle:
        yaml.dump_all(todump, fhandle, default_flow_style=None)
    element_list = get_element_list()
    for elmnt in element_list:
        with open(os.path.join(output_dir,cycle_number,'train','train_number_'+SP_n+'_'+elmnt+'.ann.param.yaml'), 'rb') as fhandle:
            ann_param = yaml.safe_load(fhandle)
        with folder.open(elmnt+'.ann.param.yaml', 'w', encoding='utf-8') as fhandle:
            yaml.dump(ann_param, fhandle)
    provenance_exclude_list = ['posinp.yaml']
    return provenance_exclude_list

def store_to_be_labeled_structures(cycle_number):
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping', 'aver_sigma.dat'), 'r', encoding='utf-8') as fhandle:
        aver_sigma = float(fhandle.read().strip())
    to_be_labeled_bulks = []
    to_be_labeled_clusters = []
    clusters = []
    bulks = []
    wf_qbc_group = Group.collection.get(label='wf_qbc')
    workflow_node = list(wf_qbc_group.nodes)
    for a_wf_node in workflow_node:
        epa_list = []
        structure_list = []
        for a_node in a_wf_node.called:
            if not a_node.is_finished_ok:
                return None, None
            if not structure_list:
                structure_list = a_node.outputs.output_parameters['confs']
            epas = []
            for a_conf in a_node.outputs.output_parameters['confs']:
                nat = a_conf['conf']['nat']
                epas.append(a_conf['conf']['epot']/nat)
            epa_list.append(epas)
        epa_list = np.array(epa_list)
        average = np.mean(epa_list, axis=0)
        sigmas = ((np.sum((epa_list - average)**2, axis=0))/3)**0.5
        for index, a_sigma in enumerate(sigmas):
            if a_sigma > aver_sigma:
                if 'free' in structure_list[index]['conf']['bc']:
                    to_be_labeled_clusters.append(structure_list[index])
                if 'bulk' in structure_list[index]['conf']['bc']:
                    to_be_labeled_bulks.append(structure_list[index])
    for a_structure in conf2pymatgenstructure(to_be_labeled_bulks):
        bulks.append(a_structure.as_dict())
    for a_structure in conf2pymatgenstructure(to_be_labeled_clusters):
        clusters.append(a_structure.as_dict())
    log_write(f'Number of to_be_labeled bulks: {len(bulks)}'+'\n')
    log_write(f'Number of to_be_labeled clusters: {len(clusters)}'+'\n')
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','to_be_labeled_cluster.json'), 'w', encoding='utf8') as fhandle:
        json.dump(clusters, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','to_be_labeled_bulk.json'), 'w', encoding='utf8') as fhandle:
        json.dump(bulks, fhandle)

def get_to_be_labeled_structures(cycle_number):
    bulk_structures = []
    cluster_structures = []
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','to_be_labeled_cluster.json'), 'r', encoding='utf8') as fhandle:
        clusters = json.loads(fhandle.read())
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','to_be_labeled_bulk.json'), 'r', encoding='utf8') as fhandle:
        bulks = json.loads(fhandle.read())
    for a_structure in bulks:
        bulk_structures.append(Structure.from_dict(a_structure))
    for a_structure in clusters:
        cluster_structures.append(Structure.from_dict(a_structure))
    return bulk_structures, cluster_structures

def store_ref_sigma(cycle_number):
    wf_qbc_group = Group.collection.get(label='wf_qbc')
    workflow_node = list(wf_qbc_group.nodes)
    for a_wf_node in workflow_node:
        epa_list = []
        for a_node in a_wf_node.called:
            if not a_node.is_finished_ok:
                return False
            epas = []
            for a_conf in a_node.outputs.output_parameters['confs']:
                nat = a_conf['conf']['nat']
                epas.append(a_conf['conf']['epot']/nat)
            epa_list.append(epas)
    epa_list = np.array(epa_list)
    average = np.mean(epa_list, axis=0)
    sigmas = ((np.sum((epa_list - average)**2, axis=0))/3)**0.5
    aver_sigma = sum(sigmas)/len(sigmas)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping', 'aver_sigma.dat'), 'w', encoding='utf-8') as fhandle:
        fhandle.writelines('%s\n' % aver_sigma)

def get_qbc_data(cycle_number):
    with open(os.path.join(Flame_dir,cycle_number,'train','training_data.json'), 'r', encoding='utf-8') as fhandle:
        training_data = json.loads(fhandle.read())
    if len(training_data) > 2000:
        selected_data = sample(training_data, 2000)
    else:
        selected_data = training_data
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','qbc_smaple.json'), 'w', encoding='utf-8') as fhandle:
        json.dump(selected_data, fhandle)

def get_ref_data(cycle_number):
    structure = []
    energy = []
    bc = []
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','qbc_smaple.json'), 'r', encoding='utf-8') as fhandle:
        qbc_sample = json.loads(fhandle.read())
    for a_sample in qbc_sample:
        structure.append(a_sample['structure'])
        energy.append(a_sample['energy'])
        bc.append(a_sample['bc'])
    return structure, energy, bc

def get_bulk_structures(cycle_number, nat):
    nat = str(nat)
    poslow_structures = []
    trajectory_structures = []

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            poslows = json.loads(fhandle.read())
        if nat in poslows.keys():
            poslow_structures.extend(poslows[nat])

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-bulk-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            poslows = json.loads(fhandle.read())
        if nat in poslows.keys():
            poslow_structures.extend(poslows[nat])

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','posmds-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            posmds = json.loads(fhandle.read())
        if nat in posmds.keys():
            trajectory_structures.extend(posmds[nat])

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-bulk-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            trajs = json.loads(fhandle.read())
        if nat in trajs.keys():
            trajectory_structures.extend(trajs[nat])

    bulk_structures = poslow_structures + trajectory_structures
    return bulk_structures

def get_cluster_structures(cycle_number, nat):
    nat = str(nat)
    poslow_structures = []
    trajectory_structures = []

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-cluster-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            poslows = json.loads(fhandle.read())
        if nat in poslows.keys():
            poslow_structures.extend(poslows[nat])

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-cluster-'+cycle_number+'.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            trajs = json.loads(fhandle.read())
        if nat in trajs.keys():
            trajectory_structures.extend(trajs[nat])

    cluster_structures = poslow_structures + trajectory_structures
    return cluster_structures

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
