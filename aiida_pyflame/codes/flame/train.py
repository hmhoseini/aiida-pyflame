import os
import sys
from random import sample
from collections import defaultdict
import json
import math
import shutil
import yaml
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from aiida.orm import List, Group, load_node, QueryBuilder, WorkChainNode, CalcJobNode
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.utils import get_element_list, is_structure_valid
from aiida_pyflame.codes.flame.core import r_cut, get_confs_from_list, write_SE_ann_input
from aiida_pyflame.workflows.core import log_write
import aiida_pyflame.workflows.settings as settings

def write_train_files(folder, cycle_number):
    with open(os.path.join(settings.Flame_dir,cycle_number,'train','training_data.json'), 'r', encoding='utf8') as fhandle:
        training_data = json.loads(fhandle.read())
    provenance_exclude_list = write_position_force_file(folder, training_data)
    write_SE_ann_input(folder, cycle_number)
    return provenance_exclude_list

def pre_train(cycle_number):
    calculation_nodes_group = Group.collection.get(label='calculation_nodes')
    path = os.path.join(settings.Flame_dir,cycle_number,'train','training_data.json')
    collected_data_from_node = []
    collected_data_from_file = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf8') as fhandle:
            data_from_file  = json.loads(fhandle.read())
        collected_data_from_file = read_training_data_from_file(data_from_file)
    else:
        if 'cycle-1' in cycle_number:
            collected_data_from_node, calculation_nodes = read_training_data_from_node('results_step3')
            # calculation nodes
            for a_node in calculation_nodes_group.nodes:
                if 'step_3' in a_node.label:
                    calculation_nodes_group.remove_nodes([load_node(a_node.pk)])
            a_node = List(calculation_nodes).store()
            a_node.label = 'step_3'
            calculation_nodes_group.add_nodes(a_node)
        else:
            collected_data_from_node, calculation_nodes = read_training_data_from_node('results_singlepoint')
            c_no = int(cycle_number.split('-')[-1])
            # calculation nodes
            for a_node in calculation_nodes_group.nodes:
                if 'singlepoint' in a_node.label and cycle_number in a_node.label:
                    calculation_nodes_group.remove_nodes([load_node(a_node.pk)])
            a_node = List(calculation_nodes).store()
            a_node.label = 'singlepoint_cycle_' + str(c_no-1)
            calculation_nodes_group.add_nodes(a_node)

            path = os.path.join(settings.Flame_dir,'cycle-'+str(c_no-1),'train','training_data.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf8') as fhandle:
                    data_from_file  = json.loads(fhandle.read())
                collected_data_from_file = read_training_data_from_file(data_from_file)
            else:
                log_write(f'>>> WARNING: no training data from cycle ({cycle_number}) <<<'+'\n')

    training_data = []
    plot_nat_b = []
    plot_epa_b = []
    plot_vpa_b = []
    plot_nat_c = []
    plot_epa_c = []
    plot_force = []

    if collected_data_from_node:
        training_data.extend(collected_data_from_node[0])
        min_epa = collected_data_from_node[1]
        this_min_epa =  collected_data_from_node[2]
        plot_nat_b.extend(collected_data_from_node[3])
        plot_epa_b.extend(collected_data_from_node[4])
        plot_vpa_b.extend(collected_data_from_node[5])
        plot_nat_c.extend(collected_data_from_node[6])
        plot_epa_c.extend(collected_data_from_node[7])
        plot_force.extend(collected_data_from_node[8])
        log_write(f'min. epa ({cycle_number}): {round(this_min_epa, 8)}'+'\n')
        log_write(f'min. epa: {round(min_epa, 8)}'+'\n')
    if collected_data_from_file:
        training_data.extend(collected_data_from_file[0])
        min_epa = collected_data_from_file[1]
        plot_nat_b.extend(collected_data_from_file[2])
        plot_epa_b.extend(collected_data_from_file[3])
        plot_vpa_b.extend(collected_data_from_file[4])
        plot_nat_c.extend(collected_data_from_file[5])
        plot_epa_c.extend(collected_data_from_file[6])
        plot_force.extend(collected_data_from_file[7])
    log_write(f'number of training data (bulk and cluster): {len(plot_nat_b)} and {len(plot_nat_c)}'+'\n')
    # store training data
    with open(os.path.join(settings.Flame_dir,cycle_number,'train','training_data.json'), 'w', encoding='utf8') as fhandle:
        json.dump(training_data, fhandle)
    r_c = r_cut()
    # plot
    for filename in os.listdir(os.path.join(settings.Flame_dir,cycle_number,'train')):
        a_file = os.path.join(settings.Flame_dir,cycle_number,'train',filename)
        if a_file.endswith('.png'):
            os.remove(a_file)
    plot_1_train(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c, min_epa)
    plot_2_train(cycle_number, plot_force)
    all_distances = get_all_distances(training_data, r_c)
    plot_5_train(cycle_number, all_distances)
    if settings.inputs['user_specified_FLAME_files'] and\
       os.path.exists(os.path.join(settings.run_dir,'flame_files','ann_input.yaml')):
        shutil.copyfile(os.path.join(settings.run_dir,'flame_files','ann_input.yaml'),\
                        os.path.join(settings.Flame_dir,cycle_number,'train','ann_input.yaml'))
    else:
        g02s, g05s = get_symmetry_function(all_distances, r_c)
        write_symmetry_function(g02s, g05s, cycle_number)

def select_a_train(cycle_number):
    c_no = int(cycle_number.split('-')[-1])
    number_of_epoch = settings.inputs['number_of_epoch'][c_no-1]
    trains = defaultdict(list)
    element_list = get_element_list()
    train_number = 0

    builder = QueryBuilder()
    builder.append(Group, filters={'label': 'wf_train'}, tag='results_group')
    builder.append(WorkChainNode, with_group='results_group', tag='wf_nodes')
    builder.append(CalcJobNode, with_incoming='wf_nodes', project='*')
    calcjob_nodes = builder.all(flat=True)

    for a_node in calcjob_nodes:
        train_number = train_number + 1
        valid_rmse = []
        a_node.outputs.remote_folder.getfile('train_output.yaml', os.path.join(settings.Flame_dir,cycle_number,'train','train_number_'+str(train_number)+'_train_output.yaml'))
        with a_node.outputs.retrieved.open('train_output.yaml', 'r') as fhandle:
            t_o = yaml.load(fhandle, Loader=yaml.FullLoader)
        for i in range(len(t_o['training iterations'])):
            try:
                valid_rmse.append(t_o['training iterations'][i]['valid']['rmse'])
            except:
                pass
        trains[a_node.pk].extend(valid_rmse)
        for elmnt in element_list:
            try:
                fname = str(elmnt)+'.ann.param.yaml.'+str(number_of_epoch).zfill(5)
                a_node.outputs.remote_folder.getfile(fname, os.path.join(settings.Flame_dir,cycle_number,'train','train_number_'+str(train_number)+'_'+str(elmnt)+'.ann.param.yaml'))
            except:
                log_write(f'>>> ERROR: no ann potential for epoch {number_of_epoch} was found <<<'+'\n')
                sys.exit()

    rmse_min = 100
    node_pk = None
    for keys in trains:
        try:
            if trains[keys][number_of_epoch] < rmse_min:
                rmse_min = trains[keys][number_of_epoch]
                node_pk = keys
        except IndexError:
            log_write(f'>>> WARNING: no RMSE for epoch {number_of_epoch} was found <<<'+'\n')
            continue
    if node_pk:
        a_node = load_node(node_pk)
        for elmnt in element_list:
            fname = str(elmnt)+'.ann.param.yaml.'+str(number_of_epoch).zfill(5)
            a_node.outputs.remote_folder.getfile(fname, os.path.join(settings.Flame_dir,cycle_number,'train',str(elmnt)+'.ann.param.yaml'))
            a_node.outputs.remote_folder.getfile('train_output.yaml', os.path.join(settings.Flame_dir,cycle_number,'train','train_output.yaml'))
    plot_6_train(cycle_number)

def write_position_force_file(folder, training_data):
    provenance_exclude_list = []
    indices_list = list(range(len(training_data)))
    valid_indices = sample(indices_list, int(len(indices_list)/10)) if int(len(indices_list)/10) < 3001 else sample(indices_list, 3000)
    for rem in valid_indices:
        indices_list.remove(rem)
    train_indices = indices_list

    valid_structure_list = []
    valid_energy_list = []
    valid_force_list = []
    valid_bc_list = []
    for i in valid_indices:
        valid_structure_list.append(training_data[i]['structure'])
        valid_energy_list.append(training_data[i]['energy'])
        valid_force_list.append(training_data[i]['forces'])
        valid_bc_list.append(training_data[i]['bc'])
    to_dump = get_confs_from_list(valid_structure_list, valid_bc_list, valid_energy_list, valid_force_list)
    with folder.open('position_force_train_valid.yaml', 'w', encoding='utf-8') as fhandle:
        yaml.dump_all(to_dump, fhandle, default_flow_style=None)
    with folder.open('list_posinp_valid.yaml', 'w', encoding='utf-8') as fhandle:
        fhandle.write('files:'+'\n')
        fhandle.write(' - position_force_train_valid.yaml'+'\n')
    provenance_exclude_list.append('position_force_train_valid.yaml')
    with folder.open('list_posinp_train.yaml', 'w', encoding='utf-8') as fhandle:
        fhandle.write('files:'+'\n')
    t = int(len(indices_list)/10000)
    samp = int(len(indices_list)/(t+1))
    for i in range(1, t+2):
        f_name = 'position_force_train_train_'+'t'+str(i).zfill(3)+'.yaml'
        provenance_exclude_list.append(f_name)
        train_indices_t = sample(train_indices,samp)
        train_structure_list = []
        train_energy_list = []
        train_force_list = []
        train_bc_list = []
        for rem in train_indices_t:
            train_structure_list.append(training_data[rem]['structure'])
            train_energy_list.append(training_data[rem]['energy'])
            train_force_list.append(training_data[rem]['forces'])
            train_bc_list.append(training_data[rem]['bc'])
            train_indices.remove(rem)
        to_dump = get_confs_from_list(train_structure_list, train_bc_list, train_energy_list, train_force_list)
        with folder.open(f_name, 'w', encoding='utf-8') as fhandle:
            yaml.dump_all(to_dump, fhandle, default_flow_style=None)
        with folder.open('list_posinp_train.yaml', 'a', encoding='utf8') as fhandle:
            fhandle.write(f' - position_force_train_train_t{str(i).zfill(3)}.yaml'+'\n')
    return provenance_exclude_list

def collect_node_data(group_label):
    labels = []
    cells = []
    positions = []
    species = []
    forces = []
    energies = []
    calculation_nodes = []
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='results_group')
    builder.append(WorkChainNode, with_group='results_group', tag='wf_nodes')
    builder.append(CalcJobNode, with_incoming='wf_nodes', project='*')
    calcjob_nodes = builder.all(flat=True)
    if 'VASP' in settings.inputs['ab_initio_code']:
        for a_node in calcjob_nodes:
            if a_node.exit_status != 0:
                continue
            misc_node = a_node.base.links.get_outgoing(link_label_filter='misc').all_nodes()[0]
            if not misc_node.dict.run_status['electronic_converged']:
                continue
            labels.append(a_node.label)
            trajectory_node = a_node.base.links.get_outgoing(link_label_filter='trajectory').all_nodes()[0]
            cells.append(trajectory_node.get_array('cells'))
            positions.append(trajectory_node.get_array('positions'))
            structure_node = a_node.base.links.get_outgoing(link_label_filter='structure').all_nodes()[0]
            species.append(structure_node.get_site_kindnames())
            energies_node = a_node.base.links.get_outgoing(link_label_filter='energies').all_nodes()[0]
            energies.append(energies_node.get_array('energy_extrapolated_electronic'))
            forces.append(trajectory_node.get_array('forces'))
            calculation_nodes.append(a_node.pk)
    if 'SIRIUS' in settings.inputs['ab_initio_code'] or 'QS' in settings.inputs['ab_initio_code']:
        for a_node in calcjob_nodes:
            if a_node.exit_status != 0:
                continue
            output_parameters = a_node.base.links.get_outgoing(link_label_filter='output_parameters').all_nodes()[0]
            motion_step = output_parameters['motion_step_info']
            if 'False' in motion_step['scf_converged']:
                continue
            labels.append(a_node.label)
            cells.append(motion_step['cells'])
            positions.append(motion_step['positions'])
            species.append(motion_step['symbols'])
            energies.append(motion_step['energy_eV'])
            forces.append(np.array(motion_step['forces']))
            calculation_nodes.append(a_node.pk)
    return labels, cells, positions, species, energies, forces, calculation_nodes

def read_training_data_from_node(group_label):
    known_structures_group = Group.collection.get(label='known_structures')
    this_min_epa = 0
    min_epa = 0
    for a_node in known_structures_group.nodes:
        if 'epas' in a_node.label:
            min_epa = min(a_node.get_list())
    e_window = settings.inputs['energy_window']
    training_data = []
    plot_nat_b = []
    plot_epa_b = []
    plot_vpa_b = []
    plot_nat_c = []
    plot_epa_c = []
    plot_force = []
    labels, cells, positions, species, energies, forces, calculation_nodes = collect_node_data(group_label)
    for index, label in enumerate(labels):
        is_cluster = False
        if 'cluster' in label:
            if settings.inputs['cluster_calculation']:
                is_cluster = True
            else:
                continue
        if 'molecule' in label:
            is_cluster = True
        cartesian = False
        if 'SIRIUS' in settings.inputs['ab_initio_code'] or 'QS' in settings.inputs['ab_initio_code']:
            cartesian = True
        energy_list = energies[index]
        cell_list = cells[index]
        position_list = positions[index]
        species_list = species[index]
        force_list = forces[index]
        pymatgen_structure = Structure(cell_list[-1], species_list, position_list[-1], to_unit_cell=True, coords_are_cartesian=cartesian)
        epot = energy_list[-1]
        force = force_list[-1]
        nat = len(pymatgen_structure.sites)
        epa = epot/nat
        tot_force = np.linalg.norm(force, axis =1)
        max_tot_force = max(tot_force)
        if max_tot_force > settings.inputs['max_force']:
            continue
#        if is_cluster and epa < min_epa:
#            continue
        min_epa = min(min_epa, epa)
        this_min_epa = min(this_min_epa, epa)
        if epa < min_epa + e_window and\
           is_structure_valid(pymatgen_structure, False, False, True, False, is_cluster)[0]:
            tmp_dict = {'structure' : pymatgen_structure.as_dict(),
                        'forces'    : force.tolist(),
                        'energy'    : epot
                       }

            if is_cluster:
                tmp_dict['bc'] = 'free'
                plot_nat_c.append(nat)
                plot_epa_c.append(epa)
            else:
                tmp_dict['bc'] = 'bulk'
                plot_nat_b.append(nat)
                plot_epa_b.append(epa)
                plot_vpa_b.append(pymatgen_structure.volume/nat)
            training_data.append(tmp_dict)
            plot_force.extend(tot_force)
        if len(energy_list) == 1:
            continue

        fmax = settings.inputs['max_force']
        fmin = 0.01
        steps = int((fmax -fmin) * 10)
        maxmin_force = [[fmin + s*(fmax-fmin)/steps , fmin+(s+1)*(fmax-fmin)/steps] for s in range(steps)]
        found = len(maxmin_force) * [False]
        for ionic_step in range(len(energy_list)-1):
            this_epot = energy_list[ionic_step]
            this_structure = Structure(cell_list[ionic_step], species_list, position_list[ionic_step], to_unit_cell=True, coords_are_cartesian=cartesian)
            nat = len(this_structure.sites)
            this_epa = this_epot/nat
            this_force = force_list[ionic_step]
            this_tot_force = np.linalg.norm(this_force, axis =1)
            max_this_tot_force = max(this_tot_force)
            if max_this_tot_force > settings.inputs['max_force']:
                continue
            if this_epa < min_epa + e_window and\
               is_structure_valid(this_structure, False, False, True, False, is_cluster)[0]:
                for m_f in range(len(maxmin_force)):
                    if False not in found:
                        break
                    if max_this_tot_force >= maxmin_force[m_f][0] and\
                       max_this_tot_force < maxmin_force[m_f][1]:
                        tmp_dict = {'structure': this_structure.as_dict(),
                                    'forces'   : this_force.tolist(),
                                    'energy'   : this_epot,
                                   }
                        if is_cluster:
                            tmp_dict['bc'] = 'free'
                            plot_nat_c.append(len(this_structure.sites))
                            plot_epa_c.append(this_epa)
                        else:
                            tmp_dict['bc'] = 'bulk'
                            plot_nat_b.append(len(this_structure.sites))
                            plot_epa_b.append(this_epa)
                            plot_vpa_b.append(this_structure.volume/nat)
                        training_data.append(tmp_dict)
                        plot_force.extend(this_tot_force)
                        found[m_f] = True
                        break
    collected_data = [training_data,
                      min_epa, this_min_epa,
                      plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c,
                      plot_force
                      ]
    return collected_data, calculation_nodes

def read_training_data_from_file(data_from_file):
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'epas' in a_node.label:
            min_epa = min(a_node.get_list())
    e_window = settings.inputs['energy_window']
    training_data = []
    plot_nat_b = []
    plot_epa_b = []
    plot_vpa_b = []
    plot_nat_c = []
    plot_epa_c = []
    plot_force = []

    for a_data in data_from_file:
        is_cluster = False
        if a_data['bc'] == 'free': 
            if not settings.inputs['cluster_calculation']:
                continue
            else:
                is_cluster = True
        force = np.array(a_data['forces'])
        pymatgen_structure = Structure.from_dict(a_data['structure'])
        nat = len(pymatgen_structure.sites)
        epa = float(a_data['energy'])/nat
        min_epa = min(min_epa, epa)
        tot_force = np.linalg.norm(force, axis =1) 
        max_tot_force = max(tot_force)
        if max_tot_force <= settings.inputs['max_force'] and\
           epa < min_epa + e_window and\
           is_structure_valid(pymatgen_structure, False, False, True, False, is_cluster)[0]:
            training_data.append(a_data)
            plot_force.extend(tot_force)
            if is_cluster and nat in settings.inputs['cluster_number_of_atoms']:
                plot_nat_c.append(nat)
                plot_epa_c.append(epa)
            if not is_cluster and nat in settings.inputs['bulk_number_of_atoms']:
                plot_nat_b.append(nat)
                plot_epa_b.append(epa)
                plot_vpa_b.append(pymatgen_structure.volume/nat)
    collected_data = [training_data,
                      min_epa,
                      plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c,
                      plot_force]
    return collected_data

def write_symmetry_function(g02s, g05s, cycle_number):
    with open(os.path.join(settings.Flame_dir,cycle_number,'train','ann_input.yaml'), 'w', encoding='utf-8') as fhandle:
        fhandle.write('g02:'+'\n')
        for g02 in g02s:
            fhandle.write(f'- {g02:.4f} {0:.4f} {0:.4f} {0:.4f}'+'\n')
        fhandle.write('g05:'+'\n')
        for g05 in g05s:
            fhandle.write(f'- {g05[0]:.4f} {g05[1]:.4f} {g05[2]:.4f} {0:.4f} {0:.4f}'+'\n')

def get_symmetry_function(all_distances, r_c):
    min_distance = min(all_distances)
    min_distance_bohr = min_distance * 1.88973
    g02s = get_g02s(min_distance_bohr, r_c, 20)
    g05s = get_g05s(min_distance_bohr, r_c, 5)
    return g02s, g05s

def get_g02s(min_distance_bohr, r_c, n_g02):
    g02s = []

    r_c_bohr = r_c * 1.88973
    fc_max_eta = (1-(min_distance_bohr/r_c_bohr)**2)**3

    max_eta_g02 = -1.0 * math.log(0.1/fc_max_eta)/(min_distance_bohr**2)
    min_eta_g02 = -1.0 * math.log(0.9)/((0.7*r_c_bohr)**2)

    function = lambda x: 0.5 - math.exp(-1*max_eta_g02*(x**2))*(1-(x/r_c_bohr)**2)**3
    x1_g02 = fsolve(function,2)[0]
    if x1_g02 < 0:
        x1_g02 = x1_g02 * -1
    function = lambda x: 0.5 - math.exp(-1*min_eta_g02*(x**2))*(1-(x/r_c_bohr)**2)**3
    x2_g02 = fsolve(function,1)[0]
    if x2_g02 < 0:
        x2_g02 = x2_g02 * -1

    x_steps = (x2_g02-x1_g02)/(n_g02-1)

    for i in range(n_g02):
        x_bohr = x1_g02+i*x_steps
        fc = (1-(x_bohr/r_c_bohr)**2)**3
        eta = round(-1.0 * math.log(0.5/fc)/(x_bohr)**2, 4)
        g02s.append(eta)
    return g02s

def get_g05s(min_distance_bohr, r_c, n_g05):
    g05s = []

    r_c_bohr = r_c * 1.88973
    fc_max_eta = (1-(min_distance_bohr/r_c_bohr)**2)**3

    max_eta_g05 = -1.0 * math.log(0.1/(fc_max_eta**2))/(2*(min_distance_bohr**2))
    min_eta_g05 = -1.0 * math.log(0.9)/((0.7*r_c_bohr)**2)

    function = lambda x: 0.5 - math.exp(-1*max_eta_g05*2*(x**2))*(1-(x/r_c_bohr)**2)**6
    x1_g05 = fsolve(function,2)[0]
    if x1_g05 < 0:
        x1_g05 = x1_g05 * -1
    function = lambda x: 0.5 - math.exp(-1*min_eta_g05*2*(x**2))*(1-(x/r_c_bohr)**2)**6
    x2_g05 = fsolve(function,1)[0]
    if x2_g05 < 0:
        x2_g05 = x2_g05 * -1

    x_steps = (x2_g05-x1_g05)/(n_g05-1)
    for i in range(n_g05):
        x_bohr = x1_g05+i*x_steps
        fc = (1-(x_bohr/r_c_bohr)**2)**3
        eta = round(-1.0 * math.log(0.5/(fc**2))/(2*(x_bohr**2)), 4)
        g05s.extend([[eta,1,1],[eta,1,-1],[eta,2,1],[eta,2,-1],[eta,4,1],[eta,4,-1],[eta,12,1],[eta,12,-1]])
    return g05s

def get_all_distances(training_data, r_c):
    all_distances = []
    for a_training_data in training_data:
        structure = Structure.from_dict(a_training_data['structure'])
        distances = read_distances(r_c, structure)
        all_distances.extend(distances)
    return all_distances

def read_distances(r_c, structure):
    distances = []
    d_matrix = structure.distance_matrix
    upper_indices = np.triu_indices_from(d_matrix, k=1)
    upper_distances = d_matrix[upper_indices]
    for a_distance in upper_distances:
        if a_distance < r_c:
            distances.append(a_distance)
    return distances

def value_to_step(val, intervals):
    for interval in intervals:
        if val > interval[0] and val <= interval[1]:
            return round(sum(interval)/len(interval), 2)
    return 0

def plot_1_train(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c, min_epa):
    vpas = []
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'vpas' in a_node.label:
            vpas = a_node.get_list()
    if plot_nat_b and plot_epa_b:
        plt.figure()
        plt.scatter(plot_nat_b,plot_epa_b, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa ($eV/atom$)')
        plt.plot([min(plot_nat_b), max(plot_nat_b)], [min_epa, min_epa])
        plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','bulk_epa-vs-nat.png'))
        plt.close()

    if plot_epa_b and plot_vpa_b:
        plt.figure()
        plt.scatter(plot_vpa_b,plot_epa_b, label='epa-vs-vpa')
        plt.xlabel(r'vpa (${\AA}^3/atom$)')
        plt.ylabel(r'epa ($eV/atom$)')
        plt.plot([min(vpas), min(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='green')
        plt.plot([max(vpas), max(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='orange')
        plt.plot([min(plot_vpa_b), max(plot_vpa_b)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','bulk_epa-vs-vpa.png'))
        plt.close()

    if plot_nat_c and plot_epa_c:
        plt.figure()
        plt.scatter(plot_nat_c,plot_epa_c, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa ($eV/atom$)')
        plt.plot([min(plot_nat_c), max(plot_nat_c)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','cluster_epa-vs-nat.png'))
        plt.close()

    if plot_nat_b and plot_vpa_b:
        plt.figure()
        plt.scatter(plot_nat_b,plot_vpa_b, label='vpa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'vpa (${\AA}^3/atom$)')
        plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','bulk_vpa-vs-nat.png'))
        plt.close()

def plot_2_train(cycle_number, forces):
#    forces = []
#    for a_force in forces_v:
#        forces.append(np.linalg.norm(a_force))
    fmin = 0 #min(forces)
    fmax = settings.inputs['max_force']# max(forces)
    steps = int((fmax - fmin) * 10)
    fstep = (fmax - fmin)/steps
    f_intervals = []
    to_plot = defaultdict(int)
    plot_i = []
    for s in range(steps):
        f_intervals.append([fmin+s*fstep, fmin+(s+1)*fstep])
        to_plot[round((fmin+s*fstep + fmin+(s+1)*fstep)/2, 2)] = 0
        plot_i.append(round((fmin+s*fstep + fmin+(s+1)*fstep)/2, 2))
    for a_force in forces:
        val = value_to_step(a_force, f_intervals)
        if val != 0:
            to_plot[val] = to_plot[val] + 1
    plot_d = []
    plot_b = []
    max_value = max(to_plot.values())
    for keys, values in sorted(to_plot.items()):
        plot_d.append(str(keys))
        plot_b.append(values/max_value)
    fig, ax = plt.subplots()
    ax.bar(plot_i, plot_b, width = 0.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel(r'force ($eV/\AA$)')
    plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','forces.png'))
    plt.close()

def plot_5_train(cycle_number, all_distances):
    dmin = 0 #min(all_distances)
    dmax = 5
    steps = int(10 * (dmax - dmin))
    dstep = (dmax - dmin)/steps
    d_intervals = []
    to_plot = defaultdict(int)
    plot_i = []
    for i in range(steps):
        d_intervals.append([dmin+i*dstep,dmin+(i+1)*dstep])
        to_plot[round((dmin+i*dstep + dmin+(i+1)*dstep)/2, 2)] = 0
        plot_i.append(round((dmin+i*dstep + dmin+(i+1)*dstep)/2, 2))
    for a_d in all_distances:
        val = value_to_step(a_d, d_intervals)
        if val != 0:
            to_plot[val] = to_plot[val] + 1
    plot_d = []
    plot_b = []
    max_value = max(to_plot.values())
    for keys, values in sorted(to_plot.items()):
        plot_d.append(str(keys))
        plot_b.append(values/max_value)
    plt.figure()
    plt.bar(plot_i, plot_b, width = 0.05)
    plt.xticks(plot_i, plot_d)
    plt.xticks(rotation='vertical')
    plt.xlabel(r'($\AA$)')
    plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','bonds.png'))
    plt.close()

def plot_6_train(cycle_number):
    plot_epoch = []
    plot_train_rmse = []
    plot_train_frmse = []
    plot_valid_rmse = []
    plot_valid_frmse = []

    with open(os.path.join(settings.Flame_dir,cycle_number,'train','train_output.yaml'), 'r', encoding='utf-8') as fhandle:
        t_o = yaml.load(fhandle, Loader=yaml.FullLoader)
    for i in range(len(t_o['training iterations'])):
        plot_epoch.append(t_o['training iterations'][i]['train']['iter'])
        plot_train_rmse.append(t_o['training iterations'][i]['train']['rmse']*27.2114)
        plot_train_frmse.append(t_o['training iterations'][i]['train']['frmse']*51422.1)
        plot_valid_rmse.append(t_o['training iterations'][i]['valid']['rmse']*27.2114)
        plot_valid_frmse.append(t_o['training iterations'][i]['valid']['frmse']*51422.1)

    plt.figure()
    plt.plot(plot_epoch[1:],plot_train_rmse[1:], label='train_rmse')
    plt.plot(plot_epoch[1:],plot_valid_rmse[1:], label='valid_rmse')
    plt.xlabel('epoch')
    plt.ylabel(r'RMSE ($meV/atom$)')
    plt.legend()
    plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','rmse.png'))
    plt.close()

    plt.figure()
    plt.plot(plot_epoch[1:],plot_train_frmse[1:], label='train_frmse')
    plt.plot(plot_epoch[1:],plot_valid_frmse[1:], label='valid_frmse')
    plt.xlabel('epoch')
    plt.ylabel(r'RMSE ($meV/\AA$)')
    plt.legend()
    plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','frmse.png'))
    plt.close()
