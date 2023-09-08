import os
import json
import yaml
from random import sample
from random import randint
import matplotlib.pyplot as plt
from collections import defaultdict
from aiida.orm import Group
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.utils import is_structure_valid
from aiida_pyflame.codes.flame.core import get_confs_from_list, conf2pymatgenstructure
from aiida_pyflame.codes.flame.flame_functions.latvec2dproj import latvec2dproj
from aiida_pyflame.codes.flame.flame_functions.io_yaml import dict2atoms
from aiida_pyflame.workflows.settings import inputs, output_dir, Flame_dir

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
        fhandle.write('  0.01        {}       {}         ediff, temperature, maximal temperature'\
              .format(minhocao_temp, minhocao_temp*2)+'\n')
    with folder.open('earr.dat', 'w', encoding='utf-8') as fhandle:
        fhandle.write('  0         {}          # No. of minima already found, no. of minima to be found in consecutive run'\
               .format(inputs['minhocao_steps'][c_no-1])+'\n')
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

def plot_minhopp(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c):
    with open(os.path.join(output_dir,'min_epa.dat'), 'r', encoding='utf-8') as fhandle:
        min_epa = float(fhandle.readline().strip())
    with open(os.path.join(output_dir,'vpa.dat'), 'r', encoding='utf-8') as fhandle:
        vpas = [float(line.strip()) for line in fhandle]

    if len(plot_nat_b) > 0 and len(plot_epa_b) > 0:
        plt.figure(1)
        plt.scatter(plot_nat_b,plot_epa_b, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(plot_nat_b), max(plot_nat_b)], [min_epa, min_epa])
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhopp_bulk_epa-vs-nat.png'))
        plt.close()

        plt.figure(2)
        plt.scatter(plot_vpa_b,plot_epa_b, label='epa-vs-vpa')
        plt.xlabel(r'vpa ${\AA}^3/atom$')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(vpas), min(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='green')
        plt.plot([max(vpas), max(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='orange')
        plt.plot([min(plot_vpa_b), max(plot_vpa_b)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhopp_bulk_epa-vs-vpa.png'))
        plt.close()

    if len(plot_nat_c) > 0 and len(plot_epa_c) > 0:
        plt.figure(3)
        plt.scatter(plot_nat_c,plot_epa_c, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(plot_nat_c), max(plot_nat_c)], [min_epa, min_epa])
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhopp_cluster_epa-vs-nat.png'))
        plt.close()

def collect_minhopp_results(cycle_number):
    poslows_bulk = defaultdict(list)
    trajs_bulk = defaultdict(list)
    failed_cluster = []
    poslows_cluster = defaultdict(list)
    trajs_cluster = defaultdict(list)
    plot_nat_b = []
    plot_epa_b = []
    plot_vpa_b = []
    plot_nat_c = []
    plot_epa_c = []

    c_no = int(cycle_number.split('-')[-1])
    min_d_prefactor = inputs['min_distance_prefactor'] * ((100-float(inputs['descending_prefactor']))/100)**(c_no)\
                      if inputs['descending_prefactor']\
                      else inputs['min_distance_prefactor']

    wf_minhopp_b_group = Group.get(label='wf_minimahopping')
    for a_wf_node in wf_minhopp_b_group.nodes:
        a_node = a_wf_node.called[-1]
        if 'minhopp' in a_node.label:
            if a_node.is_finished_ok:
                for a_conf in a_node.outputs.output_parameters['poslows']:
                    epot = a_conf['conf']['epot']
                    nat = a_conf['conf']['nat']
                    epa = 27.2114 * epot/nat
                    structure = conf2pymatgenstructure([a_conf])[0]
                    if is_structure_valid(structure, min_d_prefactor, False, False):
                        if 'cluster' in a_node.label:
                            plot_nat_c.append(nat)
                            plot_epa_c.append(epa)
                            poslows_cluster[nat].append(structure.as_dict())
                        if 'bulk' in a_node.label:
                            volume = structure.volume
                            vpa = volume/nat
                            plot_nat_b.append(nat)
                            plot_epa_b.append(epa)
                            plot_vpa_b.append(vpa)
                            poslows_bulk[nat].append(structure.as_dict())
                for a_conf in a_node.outputs.output_parameters['trajs']:
                    epot = a_conf['conf']['epot']
                    nat = a_conf['conf']['nat']
                    epa = 27.2114 * epot/nat
                    structure = conf2pymatgenstructure([a_conf])[0]
                    if is_structure_valid(structure, min_d_prefactor, False, False):
                        if 'cluster' in a_node.label:
                            trajs_cluster[nat].append(structure.as_dict())
                        if 'bulk' in a_node.label:
                            trajs_bulk[nat].append(structure.as_dict())
            else:
                if 'cluster' in a_node.label:
                    failed_cluster.append(a_node.inputs.job_type_info.dict.minhopp['structure'])
    plot_minhopp(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c)
    return poslows_bulk, trajs_bulk, poslows_cluster, trajs_cluster, failed_cluster

def store_minhopp_results(cycle_number):
    poslows_bulk, trajs_bulk, poslows_cluster, trajs_cluster, failed_cluster = collect_minhopp_results(cycle_number)

    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-bulk-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(poslows_bulk, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-cluster-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(poslows_cluster, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-bulk-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(trajs_bulk, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','trajectories-cluster-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(trajs_cluster, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','failed_cluster.json'), 'w', encoding='utf8') as fhandle:
        json.dump(failed_cluster, fhandle)
    keys = {}
    keys['bulk'] = [str(a_key) for a_key in poslows_bulk.keys()]
    keys['cluster'] = [str(a_key) for a_key in poslows_cluster.keys()]
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','nats_minhopp.json'), 'w', encoding='utf8') as fhandle:
        json.dump(keys, fhandle)

def plot_minhocao(cycle_number, plot_nat, plot_epa, plot_vpa):
    with open(os.path.join(output_dir,'min_epa.dat'), 'r', encoding='utf-8') as fhandle:
        min_epa = float(fhandle.readline().strip())
    with open(os.path.join(output_dir,'vpa.dat'), 'r', encoding='utf-8') as fhandle:
        vpas = [float(line.strip()) for line in fhandle]
    if len(plot_nat) > 1:

        plt.figure(1)
        plt.scatter(plot_nat,plot_epa, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom')
        plt.plot([min(plot_nat), max(plot_nat)], [min_epa, min_epa])
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhocao_epa-vs-nat.png'))
        plt.close()

        plt.figure(2)
        plt.scatter(plot_vpa,plot_epa, label='epa-vs-vpa')
        plt.xlabel(r'vpa ${\AA}^3/atom$')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(vpas), min(vpas)], [min(plot_epa), max(plot_epa)], color='green')
        plt.plot([max(vpas), max(vpas)], [min(plot_epa), max(plot_epa)], color='orange')
        plt.plot([min(plot_vpa), max(plot_vpa)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(Flame_dir,cycle_number,'minimahopping','minhocao_epa-vs-vpa.png'))
        plt.close()

def collect_minhocao_results(cycle_number):
    failed_bulk = []
    poslows = defaultdict(list)
    posmds = defaultdict(list)
    plot_nat = []
    plot_epa = []
    plot_vpa = []
    c_no = int(cycle_number.split('-')[-1])
    min_d_prefactor = inputs['min_distance_prefactor'] * ((100-float(inputs['descending_prefactor']))/100)**(c_no)\
                      if inputs['descending_prefactor']\
                      else inputs['min_distance_prefactor']
    with open(os.path.join(output_dir,'vpa.dat'), 'r', encoding='utf8') as fhandle:
        vpas = [float(line.strip()) for line in fhandle]
        vpa_limit = vpas[1]*2
    wf_minhocao_group = Group.get(label='wf_minimahopping')
    for a_wf_node in wf_minhocao_group.nodes:
        a_node = a_wf_node.called[-1]
        if 'minhocao' in a_node.label:
            if a_node.is_finished_ok:
                for a_conf in a_node.outputs.output_parameters['poslows']:
                    epot = a_conf['conf']['epot']
                    nat = a_conf['conf']['nat']
                    epa = 27.2114 * epot/nat
                    structure = conf2pymatgenstructure([a_conf])[0]
                    vpa = structure.volume/len(structure.sites)
                    if is_structure_valid(structure, min_d_prefactor, True, False) and vpa <= vpa_limit:
                        plot_nat.append(nat)
                        plot_epa.append(epa)
                        plot_vpa.append(vpa)
                        poslows[nat].append(structure.as_dict())
                for a_conf in a_node.outputs.output_parameters['posmds']:
                    epot = a_conf['conf']['epot']
                    nat = a_conf['conf']['nat']
                    epa = 27.2114 * epot/nat
                    structure = conf2pymatgenstructure([a_conf])[0]
                    vpa = structure.volume/len(structure.sites)
                    if is_structure_valid(structure, min_d_prefactor, True, False) and vpa <= vpa_limit:
                        posmds[nat].append(structure.as_dict())
            else:
                failed_bulk.append(a_node.inputs.job_type_info.dict.minhocao['structure'])
    plot_minhocao(cycle_number, plot_nat, plot_epa, plot_vpa)
    return poslows, posmds, failed_bulk

def store_minhocao_results(cycle_number):
    poslows, posmds, failed_bulk = collect_minhocao_results(cycle_number)

    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','poslows-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(poslows, fhandle)
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','posmds-'+cycle_number+'.json'), 'w', encoding='utf8') as fhandle:
        json.dump(posmds, fhandle)

    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','failed_bulk.json'), 'w', encoding='utf8') as fhandle:
        json.dump(failed_bulk, fhandle)
    keys = {}
    keys['bulk'] = [str(a_key) for a_key in poslows.keys()]
    with open(os.path.join(Flame_dir,cycle_number,'minimahopping','nats_minhocao.json'), 'w', encoding='utf8') as fhandle:
        json.dump(keys, fhandle)

def get_minhocao_seeds(cycle_number):
    valid_structures = []
    c_no = int(cycle_number.split('-')[-1])
    min_d_prefactor = inputs['min_distance_prefactor'] * ((100-float(inputs['descending_prefactor']))/100)**(c_no-1)\
                      if inputs['descending_prefactor']\
                      else inputs['min_distance_prefactor']
    with open(os.path.join(output_dir,'seeds_bulk.json'), 'r', encoding='utf-8') as fhandle:
        structures = json.loads(fhandle.read())
    for c_no_i in range(1, c_no):
        fpath = os.path.join(Flame_dir,'cycle-'+str(c_no_i),'minimahopping','nextstep_seeds_bulk.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as fhandle:
                next_step_struct = json.loads(fhandle.read())
            structures.extend(next_step_struct)
    for a_structure in structures:
        a_pymatgen_structure = Structure.from_dict(a_structure)
        if is_structure_valid(a_pymatgen_structure, min_d_prefactor, True, False):
            valid_structures.append(a_pymatgen_structure)
    selected_structs = sample(valid_structures, inputs['bulk_minhocao'][c_no-1])\
                     if len(valid_structures) > inputs['bulk_minhocao'][c_no-1]\
                     else valid_structures
    if len(selected_structs) < inputs['bulk_minhocao'][c_no-1]:
        q, r = divmod(inputs['bulk_minhocao'][c_no-1], len(selected_structs))
        selected_structs = q * selected_structs + selected_structs[:r]
    return selected_structs

def get_minhopp_seeds(cycle_number):
    valid_seeds_bulk = []
    valid_seeds_cluster = []
    c_no = int(cycle_number.split('-')[-1])
    min_d_prefactor = inputs['min_distance_prefactor'] * ((100-float(inputs['descending_prefactor']))/100)**(c_no-1)\
                    if inputs['descending_prefactor']\
                    else inputs['min_distance_prefactor']
    with open(os.path.join(output_dir,'seeds_bulk.json'), 'r', encoding='utf-8') as fhandle:
        seeds_bulk = json.loads(fhandle.read())
    for c_no_i in range(1, c_no):
        fpath = os.path.join(Flame_dir,'cycle-'+str(c_no_i),'minimahopping','nextstep_seeds_bulk.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as fhandle:
                seeds_bulk.extend(json.loads(fhandle.read()))
    for a_bulk in seeds_bulk:
        a_pymatgen_structure = Structure.from_dict(a_bulk)
        if is_structure_valid(a_pymatgen_structure, min_d_prefactor, True, False):
            valid_seeds_bulk.append(a_pymatgen_structure)
    selected_seeds_bulk = sample(valid_seeds_bulk, inputs['bulk_minhopp'][c_no-1])\
                        if len(valid_seeds_bulk) > inputs['bulk_minhopp'][c_no-1]\
                        else valid_seeds_bulk
    if len(selected_seeds_bulk) < inputs['bulk_minhopp'][c_no-1]:
        q, r = divmod(inputs['bulk_minhopp'][c_no-1], len(selected_seeds_bulk))
        selected_seeds_bulk = q * selected_seeds_bulk + selected_seeds_bulk[:r]
    if inputs['cluster_calculation']:
        with open(os.path.join(output_dir,'seeds_cluster.json'), 'r', encoding='utf-8') as fhandle:
            seeds_cluster = json.loads(fhandle.read())
        for c_no_i in range(1, c_no):
            fpath = os.path.join(Flame_dir,'cycle-'+str(c_no_i),'minimahopping','nextstep_seeds_cluster.json')
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as fhandle:
                    seeds_cluster.extend(json.loads(fhandle.read()))
        for a_cluster in seeds_cluster:
            a_pymatgen_structure = Structure.from_dict(a_cluster)
            if len(a_pymatgen_structure.sites) in inputs['cluster_number_of_atoms'] and\
               is_structure_valid(a_pymatgen_structure, min_d_prefactor, False, False):
               valid_seeds_cluster.append(a_pymatgen_structure)
        selected_seeds_cluster = sample(valid_seeds_cluster, inputs['cluster_minhopp'][c_no-1])\
                               if len(valid_seeds_cluster) > inputs['cluster_minhopp'][c_no-1]\
                               else valid_seeds_cluster
        if len(selected_seeds_cluster) < inputs['cluster_minhopp'][c_no-1]:
            q, r = divmod(inputs['cluster_minhopp'][c_no-1], len(selected_seeds_cluster))
            selected_seeds_cluster = q * selected_seeds_cluster + selected_seeds_cluster[:r]
    else:
        selected_seeds_cluster = []
    return selected_seeds_bulk, selected_seeds_cluster
