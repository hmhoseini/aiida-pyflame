import os
import json
from random import sample
from random import randint
from collections import defaultdict
import matplotlib.pyplot as plt
import yaml
from aiida.orm import Group, QueryBuilder, WorkChainNode, CalcJobNode
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.utils import is_structure_valid
from aiida_pyflame.codes.flame.core import get_confs_from_list, conf2pymatgenstructure
from aiida_pyflame.codes.flame.flame_functions.latvec2dproj import latvec2dproj
from aiida_pyflame.codes.flame.flame_functions.io_yaml import dict2atoms
from aiida_pyflame.workflows.core import log_write
from aiida_pyflame.workflows.settings import inputs, Flame_dir

def write_minhocao_files(folder, cycle_number, structure_dict):
    provenance_exclude_list = []
    c_no = int(cycle_number.split('-')[-1])
    conf = get_confs_from_list([structure_dict], ['bulk'], [0.1], None)
    atoms = dict2atoms(conf[0])
    atoms.cellvec,atoms.rat=latvec2dproj(atoms.cellvec,atoms.rat,atoms.nat)
    with folder.open('poscur.ascii', 'w', encoding='utf-8') as fhandle:
        fhandle.write("%d\n" % atoms.nat)
        fhandle.write("%24.15E%24.15E%24.15E\n" % (atoms.cellvec[0][0],atoms.cellvec[1][0],atoms.cellvec[1][1]))
        fhandle.write("%24.15E%24.15E%24.15E\n" % (atoms.cellvec[2][0],atoms.cellvec[2][1],atoms.cellvec[2][2]))
        for i in range(atoms.nat):
            x=atoms.rat[i][0]
            y=atoms.rat[i][1]
            z=atoms.rat[i][2]
            fhandle.write("%24.15E%24.15E%24.15E%5s\n" % (x,y,z,atoms.sat[i]))
    minhocao_temp = randint(500*c_no, 1000*2**(c_no-1))
    with folder.open('ioput', 'w', encoding='utf-8') as fhandle:
        fhandle.write(f'  0.01        {minhocao_temp}       {minhocao_temp*2}         ediff, temperature, maximal temperature'+'\n')
    with folder.open('earr.dat', 'w', encoding='utf-8') as fhandle:
        fhandle.write(f'  0         {inputs["minhocao_steps"][c_no-1]}          # No. of minima already found, no. of minima to be found in consecutive run'+'\n')
        fhandle.write('  0.400000E-03  0.150000E+00  # delta_enthalpy, delta_fingerprint')
    return provenance_exclude_list

def write_minhopp_files(folder, structure_dict, bc):
    provenance_exclude_list = []
    conf = get_confs_from_list([structure_dict], [bc], [0.1], None)
    with folder.open('posinp.yaml', 'w', encoding='utf-8') as fhandle:
        yaml.dump_all(conf, fhandle, default_flow_style=None)
    with folder.open('input.minhopp', 'w', encoding='utf-8') as fhandle:
        fhandle.write('             0 number of minima already found'+'\n')
        fhandle.write('   0.01    0.001  0.05      ediff,ekin,dt'+'\n')
    return provenance_exclude_list

def get_seeds(cycle_number):
    minhopp_seeds_bulk = []
    minhopp_seeds_cluster = []
    minhocao_seeds_bulk = []
    seeds_bulk = []
    seeds_cluster = []
    c_no = int(cycle_number.split('-')[-1])

    with open(os.path.join(Flame_dir,cycle_number,'train','training_data.json'), 'r', encoding='utf8') as fhandle:
        data_from_file = json.loads(fhandle.read())

    for a_data in data_from_file:
        if a_data['bc'] == 'free' and not inputs['cluster_calculation']:
            continue
        pymatgen_structure = Structure.from_dict(a_data['structure'])
        nat = len(pymatgen_structure.sites)
        if a_data['bc'] == 'bulk' and nat in inputs['bulk_number_of_atoms']:
            seeds_bulk.append(pymatgen_structure)
        if a_data['bc'] == 'free' and nat in inputs['cluster_number_of_atoms']:
            seeds_cluster.append(pymatgen_structure)
    if seeds_bulk:
        minhopp_seeds_bulk = sample(seeds_bulk, inputs['bulk_minhopp'][c_no-1])\
                            if len(seeds_bulk) > inputs['bulk_minhopp'][c_no-1]\
                            else seeds_bulk
        if len(minhopp_seeds_bulk) < inputs['bulk_minhopp'][c_no-1]:
            q, r = divmod(inputs['bulk_minhopp'][c_no-1], len(minhopp_seeds_bulk))
            minhopp_seeds_bulk = q * minhopp_seeds_bulk + minhopp_seeds_bulk[:r]

        minhocao_seeds_bulk = sample(seeds_bulk, inputs['bulk_minhocao'][c_no-1])\
                         if len(seeds_bulk) > inputs['bulk_minhocao'][c_no-1]\
                         else seeds_bulk
        if len(minhocao_seeds_bulk) < inputs['bulk_minhocao'][c_no-1]:
            q, r = divmod(inputs['bulk_minhocao'][c_no-1], len(minhocao_seeds_bulk))
            minhocao_seeds_bulk = q * minhocao_seeds_bulk + minhocao_seeds_bulk[:r]

    if seeds_cluster:
        minhopp_seeds_cluster = sample(seeds_cluster, inputs['cluster_minhopp'][c_no-1])\
                               if len(seeds_cluster) > inputs['cluster_minhopp'][c_no-1]\
                               else seeds_cluster
        if len(minhopp_seeds_cluster) < inputs['cluster_minhopp'][c_no-1]:
            q, r = divmod(inputs['cluster_minhopp'][c_no-1], len(minhopp_seeds_cluster))
            minhopp_seeds_cluster = q * minhopp_seeds_cluster + minhopp_seeds_cluster[:r]
    return minhopp_seeds_bulk, minhopp_seeds_cluster, minhocao_seeds_bulk

def plot_minhopp(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c):
    min_epa = 0
    vpas = []
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'epas' in a_node.label:
            min_epa = min(a_node.get_list())
        if 'vpas' in a_node.label:
            vpas = a_node.get_list()
    if plot_nat_b and plot_epa_b:
        plt.figure()
        plt.scatter(plot_nat_b,plot_epa_b, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(plot_nat_b), max(plot_nat_b)], [min_epa, min_epa])
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhopp_bulk_epa-vs-nat.png'))
        plt.close()

        plt.figure()
        plt.scatter(plot_vpa_b,plot_epa_b, label='epa-vs-vpa')
        plt.xlabel(r'vpa ${\AA}^3/atom$')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(vpas), min(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='green')
        plt.plot([max(vpas), max(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='orange')
        plt.plot([min(plot_vpa_b), max(plot_vpa_b)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhopp_bulk_epa-vs-vpa.png'))
        plt.close()

    if plot_nat_c and plot_epa_c:
        plt.figure()
        plt.scatter(plot_nat_c,plot_epa_c, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(plot_nat_c), max(plot_nat_c)], [min_epa, min_epa])
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhopp_cluster_epa-vs-nat.png'))
        plt.close()

def get_calculation_nodes(group_label, node_label):
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='results_group')
    builder.append(WorkChainNode, with_group='results_group', tag='wf_nodes')
    builder.append(CalcJobNode, with_incoming='wf_nodes', filters={'label': node_label}, project='*')
    calcjob_nodes = builder.all(flat=True)
    return calcjob_nodes

def collect_minhopp_results(cycle_number):
    min_epa = 0
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'epas' in a_node.label:
            min_epa = min(a_node.get_list())
    poslows_bulk = defaultdict(list)
    trajs_bulk = defaultdict(list)
    failed_bulk = []
    failed_cluster = []
    poslows_cluster = defaultdict(list)
    trajs_cluster = defaultdict(list)
    rejected_bulk = []
    rejected_cluster = []
    plot_nat_b = []
    plot_epa_b = []
    plot_vpa_b = []
    plot_nat_c = []
    plot_epa_c = []

    calculation_nodes_b = get_calculation_nodes('wf_minimahopping', 'minhopp_bulk_'+cycle_number)
    calculation_nodes_c = get_calculation_nodes('wf_minimahopping', 'minhopp_cluster_'+cycle_number)
    for a_node in calculation_nodes_b + calculation_nodes_c:
        is_cluster = False
        if 'cluster' in a_node.label:
            is_cluster = True
        if not a_node.is_finished_ok:
            if is_cluster:
                failed_cluster.append(a_node.inputs.job_type_info.dict.minhopp['structure'])
            else:
                failed_bulk.append(a_node.inputs.job_type_info.dict.minhopp['structure'])
            continue
        ref_structure = Structure.from_dict(a_node.inputs.job_type_info.dict.minhopp['structure'])
        for a_conf in a_node.outputs.output_parameters['poslows']:
            epot = a_conf['conf']['epot']
            nat = a_conf['conf']['nat']
            epa = 27.2114 * epot/nat
            structure = conf2pymatgenstructure([a_conf])[0]
            is_struct_valid = is_structure_valid(structure, ref_structure, 0.80, False, False, is_cluster)
            if is_struct_valid[0]:
                if is_cluster:
                    if epa > 0 or epa < min_epa:
                        rejected_cluster.append(structure.as_dict())
                        continue
                    plot_nat_c.append(nat)
                    plot_epa_c.append(epa)
                    poslows_cluster[nat].append(structure.as_dict())
                else:
                    if epa > 0 or epa < min_epa:
                        rejected_bulk.append(structure.as_dict())
                        continue
                    volume = structure.volume
                    vpa = volume/nat
                    plot_nat_b.append(nat)
                    plot_epa_b.append(epa)
                    plot_vpa_b.append(vpa)
                    poslows_bulk[nat].append(structure.as_dict())
            elif 'close' in is_struct_valid[1]:
                if is_cluster:
                    rejected_cluster.append(structure.as_dict())
                else:
                    rejected_bulk.append(structure.as_dict())
        for a_conf in a_node.outputs.output_parameters['trajs']:
            epot = a_conf['conf']['epot']
            nat = a_conf['conf']['nat']
            epa = 27.2114 * epot/nat
            structure = conf2pymatgenstructure([a_conf])[0]
            is_struct_valid = is_structure_valid(structure, ref_structure, 0.80, False, False, is_cluster)
            if is_struct_valid[0]:
                if is_cluster:
                    if epa > 0 or epa < min_epa:
                        rejected_cluster.append(structure.as_dict())
                        continue
                    trajs_cluster[nat].append(structure.as_dict())
                else:
                    if epa > 0 or epa < min_epa:
                        rejected_bulk.append(structure.as_dict())
                        continue
                    trajs_bulk[nat].append(structure.as_dict())

    plot_minhopp(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c)
    return poslows_bulk, trajs_bulk, rejected_bulk, failed_bulk, poslows_cluster, trajs_cluster, rejected_cluster, failed_cluster

def store_minhopp_results(cycle_number):
    poslows_bulk, trajs_bulk, rejected_bulk, failed_bulk,\
    poslows_cluster, trajs_cluster, rejected_cluster, failed_cluster = collect_minhopp_results(cycle_number)

    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-bulk-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(poslows_bulk, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-cluster-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(poslows_cluster, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-bulk-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(trajs_bulk, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-cluster-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(trajs_cluster, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','failed_bulk.json'), 'w', encoding='utf8') as fhandle:
        json.dump(failed_bulk, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','failed_cluster.json'), 'w', encoding='utf8') as fhandle:
        json.dump(failed_cluster, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','rejected_bulks.json'), 'w', encoding='utf8') as fhandle:
        json.dump(rejected_bulk, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','rejected_clusters.json'), 'w', encoding='utf8') as fhandle:
        json.dump(rejected_cluster, fhandle)

    nats = {}
    nats_bulk = []
    nats_cluster = []
    log_write('minhopp report:'+'\n')
    for keys, values in poslows_bulk.items():
        log_write(f'Number of generated bulk structures with {keys} atoms: {len(values)}'+'\n')
        nats_bulk.append(str(keys))
    log_write(f'Number of bulk structures from trajectories: {len(trajs_bulk)}'+'\n')
    log_write(f'Number of rejected bulk structures: {len(rejected_bulk)}'+'\n')
    log_write(f'Number of failed bulk calculations: {len(failed_bulk)}'+'\n')
    for keys, values in poslows_cluster.items():
        log_write(f'Number of generated cluster structures with {keys} atoms: {len(values)}'+'\n')
        nats_cluster.append(str(keys))
    log_write(f'Number of cluster structures from trajectories: {len(trajs_cluster)}'+'\n')
    log_write(f'Number of rejected cluster structures: {len(rejected_cluster)}'+'\n')
    log_write(f'Number of failed cluster calculations: {len(failed_cluster)}'+'\n')
    nats['bulk'] = nats_bulk
    nats['cluster'] = nats_cluster
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','nats_minhopp.json'), 'w', encoding='utf8') as fhandle:
        json.dump(nats, fhandle)

def plot_minhocao(cycle_number, plot_nat, plot_epa, plot_vpa):
    min_epa = 0
    vpas = []
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'epas' in a_node.label:
            min_epa = min(a_node.get_list())
        if 'vpas' in a_node.label:
            vpas = a_node.get_list()

    if plot_nat and plot_epa:
        plt.figure()
        plt.scatter(plot_nat,plot_epa, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom')
        plt.plot([min(plot_nat), max(plot_nat)], [min_epa, min_epa])
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhocao_epa-vs-nat.png'))
        plt.close()

    if plot_vpa and plot_epa:
        plt.figure()
        plt.scatter(plot_vpa,plot_epa, label='epa-vs-vpa')
        plt.xlabel(r'vpa ${\AA}^3/atom$')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(vpas), min(vpas)], [min(plot_epa), max(plot_epa)], color='green')
        plt.plot([max(vpas), max(vpas)], [min(plot_epa), max(plot_epa)], color='orange')
        plt.plot([min(plot_vpa), max(plot_vpa)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhocao_epa-vs-vpa.png'))
        plt.close()

def collect_minhocao_results(cycle_number):
    min_epa = 0
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'epas' in a_node.label:
            min_epa = min(a_node.get_list())
    failed_structures = []
    rejected_structures = []
    poslows = defaultdict(list)
    posmds = defaultdict(list)
    plot_nat = []
    plot_epa = []
    plot_vpa = []

    calcjob_nodes = get_calculation_nodes('wf_minimahopping', 'minhocao_'+cycle_number)
    for a_node in calcjob_nodes:
        if not a_node.is_finished_ok:
            failed_structures.append(a_node.inputs.job_type_info.dict.minhocao['structure'])
            continue
        for a_conf in a_node.outputs.output_parameters['poslows']:
            epot = a_conf['conf']['epot']
            nat = a_conf['conf']['nat']
            epa = 27.2114 * epot/nat
            structure = conf2pymatgenstructure([a_conf])[0]
            vpa = structure.volume/len(structure.sites)
            is_struct_valid = is_structure_valid(structure, False, 0.80, True, [0.5,2], False)
            if is_struct_valid[0]:
                if epa > 0 or epa < min_epa:
                    rejected_structures.append(structure.as_dict())
                    continue
                plot_nat.append(nat)
                plot_epa.append(epa)
                plot_vpa.append(vpa)
                poslows[nat].append(structure.as_dict())
            elif 'close' in is_struct_valid[1]:
                rejected_structures.append(structure.as_dict())
        for a_conf in a_node.outputs.output_parameters['posmds']:
            epot = a_conf['conf']['epot']
            nat = a_conf['conf']['nat']
            epa = 27.2114 * epot/nat
            structure = conf2pymatgenstructure([a_conf])[0]
            vpa = structure.volume/len(structure.sites)
            if is_structure_valid(structure, False, 0.80, True, [0.5, 2], False)[0]:
                if epa > 0 or epa < min_epa:
                    rejected_structures.append(structure.as_dict())
                    continue
                posmds[nat].append(structure.as_dict())
    plot_minhocao(cycle_number, plot_nat, plot_epa, plot_vpa)
    return poslows, posmds, rejected_structures, failed_structures

def store_minhocao_results(cycle_number):
    poslows, posmds, rejected_structures, failed_structures = collect_minhocao_results(cycle_number)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(poslows, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','posmds-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(posmds, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','failed_structures.json'), 'w', encoding='utf8') as fhandle:
        json.dump(failed_structures, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','rejected_structures.json'), 'w', encoding='utf8') as fhandle:
        json.dump(rejected_structures, fhandle)

    nats = {}
    nats_bulk = []
    log_write('minhocao report:'+'\n')
    for keys, values in poslows.items():
        log_write(f'Number of generated bulk structures with {keys} atoms: {len(values)}'+'\n')
        nats_bulk.append(str(keys))
    log_write(f'Number of bulk structures from trajectories: {len(posmds)}'+'\n')
    log_write(f'Number of rejected structures: {len(rejected_structures)}'+'\n')
    log_write(f'Number of failed calculations: {len(failed_structures)}'+'\n')
    nats['bulk'] = nats_bulk
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','nats_minhocao.json'), 'w', encoding='utf8') as fhandle:
        json.dump(nats, fhandle)
