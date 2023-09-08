import os
import sys
import yaml
import json
import math
import shutil
import numpy as np
from scipy.optimize import fsolve
from collections import defaultdict
import matplotlib.pyplot as plt
from random import sample
from aiida.orm import Group, load_node
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.utils import r_cut, get_element_list, is_structure_valid
from aiida_pyflame.codes.flame.core import get_confs_from_list, write_SE_ann_input
from aiida_pyflame.workflows.core import log_write
import aiida_pyflame.workflows.settings as settings

def write_train_files(folder, cycle_number):
    with open(os.path.join(settings.Flame_dir,cycle_number,'train','position_force_train_all.json'), 'r', encoding='utf8') as fhandle:
        training_data = json.loads(fhandle.read())
    provenance_exclude_list = write_position_force_file(folder, training_data)
    write_SE_ann_input(folder, cycle_number)
    return provenance_exclude_list

def pre_train(cycle_number):
    training_data = collect_training_data(cycle_number)
    with open(os.path.join(settings.Flame_dir,cycle_number,'train','position_force_train_all.json'), 'w', encoding='utf8') as fhandle:
        json.dump(training_data, fhandle)
    all_distances = get_all_distances(training_data)
    if settings.inputs['user_specified_FLAME_files'] and\
       os.path.exists(os.path.join(settings.run_dir,'flame_files','ann_input.yaml')):
        shutil.copyfile(os.path.join(settings.run_dir,'flame_files','ann_input.yaml'),\
                        os.path.join(settings.Flame_dir,cycle_number,'train','ann_input.yaml'))
    else:
        g02s, g05s = get_symmetry_function(all_distances)
        write_symmetry_function(g02s, g05s, cycle_number)
    plot_2_train(cycle_number, all_distances)

def select_a_train(cycle_number):
    c_no = int(cycle_number.split('-')[-1])
    number_of_epoch = settings.inputs['number_of_epoch'][c_no-1]
    nodes = []
    trains = defaultdict(list)
    element_list = get_element_list()
    train_number = 0

    wf_train_group = Group.get(label='wf_train')
    workflow_node = list(wf_train_group.nodes)[-1]
    for a_node in workflow_node.called:
        nodes.append(a_node.pk)
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
                a_node.outputs.remote_folder.getfile(fname,\
                os.path.join(settings.Flame_dir,cycle_number,'train','train_number_'+str(train_number)+'_'+str(elmnt)+'.ann.param.yaml.'+str(number_of_epoch).zfill(5)))
            except:
                log_write('>>> ERROR: no ann potential for epoch {} was found <<<'.format(number_of_epoch)+'\n')
                sys.exit()
    with open(os.path.join(settings.Flame_dir,cycle_number,'train', 'nodes.dat'), 'w', encoding='utf-8') as fhandle:
        fhandle.writelines(['%s\n' % node  for node in nodes])
    rmse_min = 100
    node_pk = None
    for keys in trains:
        try:
            if trains[keys][number_of_epoch] < rmse_min:
                rmse_min = trains[keys][number_of_epoch]
                node_pk = keys
        except IndexError:
            log_write('>>> WARNING: no RMSE for epoch {} was found <<<'.format(number_of_epoch)+'\n')
            continue
    if node_pk:
        a_node = load_node(node_pk)
        for elmnt in element_list:
            fname = str(elmnt)+'.ann.param.yaml.'+str(number_of_epoch).zfill(5)
            a_node.outputs.remote_folder.getfile(fname, os.path.join(settings.Flame_dir,cycle_number,'train',str(elmnt)+'.ann.param.yaml'))
            a_node.outputs.remote_folder.getfile('train_output.yaml', os.path.join(settings.Flame_dir,cycle_number,'train','train_output.yaml'))
    plot_3_train(cycle_number)

def write_position_force_file(folder, training_data):
    provenance_exclude_list = []
    indices_list = list(range(len(training_data)))
    valid_indices = sample(indices_list, int(len(indices_list)/5)) if int(len(indices_list)/5) < 10001 else sample(indices_list, 10000)
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
            fhandle.write(' - {}'.format('position_force_train_train_'+'t'+str(i).zfill(3)+'.yaml'+'\n'))
    return provenance_exclude_list

def collect_training_data(cycle_number):
    c_no = int(cycle_number.split('-')[-1])
    with open(os.path.join(settings.output_dir,'min_epa.dat'), 'r', encoding='utf8') as fhandle:
        min_epa = float(fhandle.readline().strip())
    e_window = settings.inputs['energy_window']
    training_data = []
    plot_nat_b = []
    plot_epa_b = []
    plot_vpa_b = []
    plot_nat_c = []
    plot_epa_c = []

    results_group = Group.get(label='results_step3') if c_no == 1 else Group.get(label='results_singlepoint')

    for a_wch_node in results_group.nodes:
        for a_node in a_wch_node.called:
            if not a_node.is_finished_ok:
                continue
            single_ionic_step = False

            if 'VASP' in settings.inputs['ab_initio_code']:
                if not a_node.outputs.misc.dict.run_status['electronic_converged']:
                    continue
                total_energy = float(a_node.outputs.energies.get_array('energy_extrapolated_electronic')[-1])
                pymatgen_structure = a_node.outputs.structure.get_pymatgen()
                forces = a_node.outputs.trajectory.get_array('forces')[-1].tolist()
                if len(a_node.outputs.energies.get_array('electronic_steps')) == 1:
                    single_ionic_step = True
            if 'SIRIUS' in settings.inputs['ab_initio_code'] or 'GTH' in settings.inputs['ab_initio_code']:
                if not a_node.outputs.output_parameters.dict['motion_step_info']['scf_converged'][-1]:
                    continue
                total_energy = float(a_node.outputs.output_parameters.dict['motion_step_info']['energy_eV'][-1])
                pymatgen_structure = a_node.outputs.output_structure.get_pymatgen()
                forces = a_node.outputs.output_parameters.dict['motion_step_info']['forces'][-1]
                if len(a_node.outputs.output_parameters.dict['motion_step_info']['step']) == 1:
                    single_ionic_step = True
            nat = len(pymatgen_structure.sites)
            epa = total_energy/nat
            if epa < min_epa + e_window and is_structure_valid(pymatgen_structure, False, True, False):
                tmp_dict = {'structure' : pymatgen_structure.as_dict(),
                            'forces'    : forces,
                            'energy'    : total_energy
                           }
                if 'bulk' in a_node.label:
                    tmp_dict['bc'] = 'bulk'
                    plot_nat_b.append(nat)
                    plot_epa_b.append(epa)
                    plot_vpa_b.append(pymatgen_structure.volume/nat)
                elif 'cluster' in a_node.label:
                    if not settings.inputs['cluster_calculation']:
                        continue
                    tmp_dict['bc'] = 'free'
                    plot_nat_c.append(nat)
                    plot_epa_c.append(epa)
                training_data.append(tmp_dict)

            if single_ionic_step:
                continue

            maxmin_force = [[5.00,4.50], [4.50,4.00], [4.00,3.50], [3.50,3.00],\
                            [3.00,2.80], [2.80,2.60], [2.60,2.40], [2.40,2.20], [2.20,2.00],\
                            [2.00,1.80], [1.80,1.60], [1.60,1.40], [1.40,1.20], [1.20,1.00],\
                            [1.00,0.80], [0.80,0.60],\
                            [0.60,0.50], [0.50,0.40], [0.40,0.30], [0.30,0.20], [0.20,0.10],\
                            [0.10,0.09], [0.09,0.08], [0.08,0.07], [0.07,0.06], [0.06,0.05]]
            found = len(maxmin_force) * [False]

            if 'VASP' in settings.inputs['ab_initio_code']:
                trajectory = a_node.outputs.trajectory
                for ionic_step in range(len(a_node.outputs.energies.get_array('electronic_steps'))-1, 0, -1):
                    this_epot = float(a_node.outputs.energies.get_array('energy_extrapolated_electronic')[ionic_step])
                    this_structure = Structure(trajectory.get_array('cells')[ionic_step],\
                                               a_node.outputs.structure.get_pymatgen().species,\
                                               trajectory.get_array('positions')[ionic_step])
                    nat = len(this_structure.sites)
                    this_epa = this_epot/nat
                    if this_epa < min_epa + e_window and is_structure_valid(this_structure, False, True, False):
                        this_forces = trajectory.get_array('forces')[ionic_step].tolist()
                        this_tot_forces = []
                        for a_f in range(len(this_forces)):
                            this_tot_forces.append(math.sqrt(this_forces[a_f][0]**2 + this_forces[a_f][1]**2 + this_forces[a_f][2]**2))
                        max_this_tot_force = max(this_tot_forces)
            if 'SIRIUS' in settings.inputs['ab_initio_code'] or 'GTH' in settings.inputs['ab_initio_code']:
                motion_step = a_node.outputs.output_parameters.dict['motion_step_info']
                for ionic_step in range(len(motion_step['step'])-1, 0, -1):
                    if not motion_step['scf_converged'][ionic_step]:
                        continue
                    this_epot = motion_step['energy_eV'][ionic_step]
                    this_structure = Structure(motion_step['cells'][ionic_step],\
                                               motion_step['symbols'],\
                                               motion_step['positions'][ionic_step-1],\
                                               coords_are_cartesian=True)
                    nat = len(this_structure.sites)
                    this_epa = this_epot/nat
                    if this_epa < min_epa + e_window and is_structure_valid(this_structure, False, True, False):
                        this_forces = motion_step['forces'][ionic_step-1]
                        this_tot_forces = []
                        for a_f in range(len(this_forces)):
                            this_tot_forces.append(math.sqrt(this_forces[a_f][0]**2 + this_forces[a_f][1]**2 + this_forces[a_f][2]**2))
                        max_this_tot_force = max(this_tot_forces)

            for m_f in range(len(maxmin_force)):
                if not False in found:
                    break
                if not found[m_f] and max_this_tot_force < maxmin_force[m_f][0] and max_this_tot_force >= maxmin_force[m_f][1]:
                    tmp_dict = {'structure': this_structure.as_dict(),
                                'forces'   : this_forces,
                                'energy'   : this_epot,
                               }
                    if 'bulk' in a_node.label:
                        tmp_dict['bc'] = 'bulk'
                        plot_nat_b.append(len(this_structure.sites))
                        plot_epa_b.append(this_epa)
                        plot_vpa_b.append(this_structure.volume/nat)
                    elif 'cluster' in a_node.label:
                        if not settings.inputs['cluster_calculation']:
                            continue
                        tmp_dict['bc'] = 'free'
                        plot_nat_c.append(len(this_structure.sites))
                        plot_epa_c.append(this_epa)

                    training_data.append(tmp_dict)
                    found[m_f] = True
                    break

    plot_1_train(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c)

    if c_no > 1:
        path = os.path.join(settings.Flame_dir,'cycle-'+str(c_no-1),'train','position_force_train_all.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf8') as fhandle:
                prev_training_data  = json.loads(fhandle.read())
            for a_p_t_d in prev_training_data:
                if float(a_p_t_d["energy"])/len(a_p_t_d["structure"]["sites"]) < min_epa + e_window:
                    training_data.append(a_p_t_d)
        else:
            log_write('>>> WARNING: no training data from cycle ({}) <<<'.format(cycle_number)+'\n')
    return training_data

def write_symmetry_function(g02s, g05s, cycle_number):
    with open(os.path.join(settings.Flame_dir,cycle_number,'train','ann_input.yaml'), 'w', encoding='utf-8') as fhandle:
        fhandle.write('{}'.format('g02:')+'\n')
        for g02 in g02s:
            fhandle.write('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format('-', g02, 0, 0, 0)+'\n')
        fhandle.write('{}'.format('g05:')+'\n')
        for g05 in g05s:
            fhandle.write('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format('-', g05[0], g05[1], g05[2], 0, 0)+'\n')

def get_symmetry_function(all_distances):
    min_distance = min(all_distances)
    min_distance_bohr = min_distance * 1.88973
    g02s = get_g02s(min_distance_bohr, 10)
    g05s = get_g05s(min_distance_bohr, 4)
    return g02s, g05s

def get_g02s(min_distance_bohr, n_g02):
    g02s = []

    r_c_bohr = r_cut() * 1.88973
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

def get_g05s(min_distance_bohr, n_g05):
    g05s = []

    r_c_bohr = r_cut() * 1.88973
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

def get_all_distances(training_data):
    r_c = r_cut()
    all_distances = []
    for a_training_data in training_data:
        all_distances.extend(read_distances(r_c, a_training_data))
    return all_distances

def read_distances(r_c, training_data):
    distances = []
    structure = Structure.from_dict(training_data['structure'])
    d_matrix = structure.distance_matrix
    upper_indices = np.triu_indices_from(d_matrix, k=1)
    upper_distances = d_matrix[upper_indices]
    for a_distance in upper_distances:
        if a_distance < r_c:
            distances.append(a_distance)
    return distances

def plot_1_train(cycle_number, plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c):
    with open(os.path.join(settings.output_dir,'min_epa.dat'), 'r', encoding='utf-8') as fhandle:
        min_epa = float(fhandle.readline().strip())

    with open(os.path.join(settings.output_dir,'vpa.dat'), 'r', encoding='utf-8') as fhandle:
        vpas = [float(line.strip()) for line in fhandle]

    if len(plot_nat_b) > 0 and len(plot_epa_b) > 0:
        plt.figure(1)
        plt.scatter(plot_nat_b,plot_epa_b, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(plot_nat_b), max(plot_nat_b)], [min_epa, min_epa])
        plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','bulk_epa-vs-nat.png'))
        plt.close()

    if len(plot_epa_b) > 0 and len(plot_vpa_b) > 0:
        plt.figure(2)
        plt.scatter(plot_vpa_b,plot_epa_b, label='epa-vs-vpa')
        plt.xlabel(r'vpa ${\AA}^3/atom$')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(vpas), min(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='green')
        plt.plot([max(vpas), max(vpas)], [min(plot_epa_b), max(plot_epa_b)], color='orange')
        plt.plot([min(plot_vpa_b), max(plot_vpa_b)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','bulk_epa-vs-vpa.png'))
        plt.close()

    if len(plot_nat_c) > 0 and len(plot_epa_c) > 0:
        plt.figure(3)
        plt.scatter(plot_nat_c,plot_epa_c, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa $eV/atom$')
        plt.plot([min(plot_nat_b), max(plot_nat_b)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','cluster_epa-vs-nat.png'))
        plt.close()

def plot_2_train(cycle_number, all_distances):
    min_d = min(all_distances)
    max_d = r_cut()
    d_step = (max_d - min_d)/20
    d_intervals = []
    for i in range(20):
        d_intervals.append([min_d+i*d_step,min_d+(i+1)*d_step])
    to_plot = defaultdict(int)
    all_d = 0
    for a_d in all_distances:
        if a_d < max_d:
            val = plot_2_read(a_d, d_intervals)
            if val != 0:
                to_plot[val] = to_plot[val] + 1
                all_d = all_d + 1
    plot_d = []
    plot_b = []
    plot_i = [i*4 for i in range(1,21)]
    for keys, values in sorted(to_plot.items()):
        plot_d.append(str(keys))
        plot_b.append(values/all_d)
    plt.figure(6)
    plt.bar(plot_i, plot_b, width = 4 * d_step)
    plt.xticks(plot_i, plot_d)
    plt.xticks(rotation='vertical')
    plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','bonds.png'))
    plt.close()

def plot_2_read(a_d, d_intervals):
    for d_i in d_intervals:
        if a_d > d_i[0] and a_d <= d_i[1]:
            return round((d_i[0]+d_i[1])/2, 2)
    return 0

def plot_3_train(cycle_number):
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

    plt.figure(1)
    plt.plot(plot_epoch[1:],plot_train_rmse[1:], label='train_rmse')
    plt.plot(plot_epoch[1:],plot_valid_rmse[1:], label='valid_rmse')
    plt.xlabel('epoch')
    plt.ylabel(r'RMSE $meV/atom$')
    plt.legend()
    plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','rmse.png'))
    plt.close()

    plt.figure(2)
    plt.plot(plot_epoch[1:],plot_train_frmse[1:], label='train_frmse')
    plt.plot(plot_epoch[1:],plot_valid_frmse[1:], label='valid_frmse')
    plt.xlabel('epoch')
    plt.ylabel(r'RMSE $meV/\AA$')
    plt.legend()
    plt.savefig(os.path.join(settings.Flame_dir,cycle_number,'train','frmse.png'))
    plt.close()
