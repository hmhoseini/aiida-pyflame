import os
import sys
from collections import defaultdict
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymatgen.core.structure import Structure
from aiida.orm import Group, QueryBuilder, CalcJobNode, Node, load_node
from aiida.manage.configuration import load_profile

load_profile()

def collect_author_data(group_label='imported_calculation_nodes'):
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='group')
    builder.append(Node, with_group='group', filters={'label': 'author_data'}, project='id')
    pk = builder.all(flat=True)[0]
    author_data = load_node(pk).get_dict()
    return author_data

def collect_input_data(group_label='imported_calculation_nodes'):
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='group')
    builder.append(Node, with_group='group', filters={'label': 'inputs'}, project='id')
    pk = builder.all(flat=True)[0]
    inputs = load_node(pk).get_dict()
    return inputs

def collect_protocol(group_label='imported_calculation_nodes'):
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='group')
    builder.append(Node, with_group='group', filters={'label': 'protocol'}, project='id')
    pk = builder.all(flat=True)[0]
    protocol = load_node(pk).get_dict()
    return protocol

def collect_node_data(group_label, code):
    labels = []
    cells = []
    positions = []
    species = []
    forces = []
    energies = []
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='group')
    builder.append(CalcJobNode, with_group='group', project='*')
    calcjob_nodes = builder.all(flat=True)

    code_version_1 = None
    code_version_2 = None
    pps = {}
    if 'VASP' in code:
        for a_node in calcjob_nodes:
            if a_node.exit_status != 0:
                continue
            misc_node = a_node.base.links.get_outgoing(link_label_filter='misc').all_nodes()[0]
            if not misc_node.dict.run_status['electronic_converged']:
                continue
            if not 'VaspCalculation' in a_node.process_label:
                print('wrong code label')
                sys.exit()
            if not code_version_1:
                code_version_1 = a_node.outputs.misc.get('version')
            else:
                if a_node.outputs.misc.get('version') != code_version_1:
                    print('wrong code version')
                    sys.exit()
            if not pps:
                for a_key in a_node.inputs.potential.keys():
                    pps.update({a_key: f"{a_node.inputs.potential[a_key].base.attributes.get('title')} {a_node.inputs.potential[a_key].base.attributes.get('sha512')}"})
            else:
                for a_key in a_node.inputs.potential.keys():
                    if not f"{a_node.inputs.potential[a_key].base.attributes.get('title')} {a_node.inputs.potential[a_key].base.attributes.get('sha512')}" in pps.values():
                        print('wrong potential')
                        sys.exit()
            labels.append(a_node.label)
            trajectory_node = a_node.base.links.get_outgoing(link_label_filter='trajectory').all_nodes()[0]
            cells.append(trajectory_node.get_array('cells'))
            positions.append(trajectory_node.get_array('positions'))
            structure_node = a_node.base.links.get_outgoing(link_label_filter='structure').all_nodes()[0]
            species.append(structure_node.get_site_kindnames())
            energies_node = a_node.base.links.get_outgoing(link_label_filter='energies').all_nodes()[0]
            energies.append(energies_node.get_array('energy_extrapolated_electronic'))
            forces.append(trajectory_node.get_array('forces'))
    if 'SIRIUS' in code or 'QS' in code:
        for a_node in calcjob_nodes:
            if a_node.exit_status != 0:
                continue
            output_parameters = a_node.base.links.get_outgoing(link_label_filter='output_parameters').all_nodes()[0]
            motion_step = output_parameters['motion_step_info']
            if 'False' in motion_step['scf_converged']:
                continue
            if 'Cp2kCalculation' not in a_node.process_label:
                print('wrong code label')
                sys.exit()
            if output_parameters['SIRIUS']:
                if not code_version_1:
                    code_version_1 = output_parameters['cp2k_version']
                else:
                    if output_parameters['cp2k_version'] != code_version_1:
                        print('wrong code version 1')
                        sys.exit()
                if not code_version_2:
                    code_version_2 = output_parameters['sirius_version']
                else:
                    if output_parameters['sirius_version'] != code_version_2:
                        print('wrong code version 2')
                        sys.exit()
            else:
                if not code_version_1:
                    code_version_1 = output_parameters['cp2k_version']
                else:
                    if output_parameters['cp2k_version'] != code_version_1:
                        print('wrong code version')
                        sys.exit()
            if not pps:
                for a_pp in a_node.inputs.parameters.dict.FORCE_EVAL['SUBSYS']['KIND']:
                    pps.update({a_pp['_']: a_pp['POTENTIAL']})
            else:
                for a_pp in a_node.inputs.parameters.dict.FORCE_EVAL['SUBSYS']['KIND']:
                    if not a_pp['POTENTIAL'] in pps.values():
                        print('wrong potential')
                        sys.exit()
            labels.append(a_node.label)
            cells.append(motion_step['cells'])
            positions.append(motion_step['positions'])
            species.append(motion_step['symbols'])
            energies.append(motion_step['energy_eV'])
            forces.append(np.array(motion_step['forces']))
    return labels, cells, positions, species, energies, forces, pps, [code_version_1, code_version_2]

def collect_data(code):
    labels, cells, positions, species, energies, forces, pps, code_version = collect_node_data('imported_calculation_nodes', code)
    min_epa = 0
    training_data = []
    plot_nat_b = []
    plot_epa_b = []
    plot_vpa_b = []
    plot_nat_c = []
    plot_epa_c = []
    plot_force = []
    for index, label in enumerate(labels):
        is_cluster = False
        if 'cluster' in label or 'molecule' in label:
            is_cluster = True
        cartesian = False
        if 'SIRIUS' in code or 'QS' in code:
            cartesian = True
        energy_list = energies[index]
        cell_list = cells[index]
        position_list = positions[index]
        species_list = species[index]
        force_list = forces[index]
        for ionic_step in range(len(energy_list)-0):
            epot = energy_list[ionic_step]
            structure = Structure(cell_list[ionic_step], species_list, position_list[ionic_step], to_unit_cell=True, coords_are_cartesian=cartesian)
            nat = len(structure.sites)
            epa = epot/nat
            min_epa = min(epa, min_epa)
            force = force_list[ionic_step]
            tot_force = np.linalg.norm(force, axis =1)
            tmp_dict = {'structure': structure.as_dict(),
                        'forces'   : force.tolist(),
                        'energy'   : epot,
                       }
            if is_cluster:
                tmp_dict['bc'] = 'free'
                plot_nat_c.append(len(structure.sites))
                plot_epa_c.append(epa)
            else:
                tmp_dict['bc'] = 'bulk'
                plot_nat_b.append(len(structure.sites))
                plot_epa_b.append(epa)
                plot_vpa_b.append(structure.volume/nat)
            training_data.append(tmp_dict)
            plot_force.extend(tot_force)
    collected_data = [training_data,
                      min_epa,
                      plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c,
                      plot_force
                      ]
    return collected_data, pps, code_version

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

def plot_1(plot_nat_b, plot_epa_b, plot_vpa_b, plot_nat_c, plot_epa_c, min_epa):
    if plot_nat_b and plot_epa_b:
        plt.figure()
        plt.scatter(plot_nat_b,plot_epa_b, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa ($eV/atom$)')
        plt.plot([min(plot_nat_b), max(plot_nat_b)], [min_epa, min_epa])
        plt.savefig(os.path.join('.','exported_data','bulk_epa-vs-nat.png'))
        plt.close()

    if plot_epa_b and plot_vpa_b:
        plt.figure()
        plt.scatter(plot_vpa_b,plot_epa_b, label='epa-vs-vpa')
        plt.xlabel(r'vpa (${\AA}^3/atom$)')
        plt.ylabel(r'epa ($eV/atom$)')
        plt.plot([min(plot_vpa_b), max(plot_vpa_b)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join('.','exported_data','bulk_epa-vs-vpa.png'))
        plt.close()

    if plot_nat_c and plot_epa_c:
        plt.figure()
        plt.scatter(plot_nat_c,plot_epa_c, label='epa-vs-nat')
        plt.xlabel('nat')
        plt.ylabel(r'epa ($eV/atom$)')
        plt.plot([min(plot_nat_c), max(plot_nat_c)], [min_epa, min_epa], color='navy')
        plt.savefig(os.path.join('.','exported_data','cluster_epa-vs-nat.png'))
        plt.close()

def plot_2(forces):
    fmin = 0 #min(forces)
    fmax = max(forces)
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
    plt.xlabel(r'force ($meV/\AA$)')
    plt.savefig(os.path.join('.','exported_data','forces.png'))
    plt.close()

def plot_3(training_data):
    all_distances = get_all_distances(training_data, 5)
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
    plt.savefig(os.path.join('.','exported_data','bonds.png'))
    plt.close()

if __name__ == "__main__":
    try:
       os.mkdir(os.path.join('.','exported_data'))
    except FileExistsError:
       print(f'>>> Cannot proceed: {os.path.join(".","exported_data")} exists <<<'+'\n')
       sys.exit()

    author_data = collect_author_data()
    inputs = collect_input_data()
    protocol = collect_protocol()
    collected_data, pps, code_version = collect_data(inputs['ab_initio_code'])
    if not code_version[-1]:
        code_version.pop(-1)
    todump = {'Author data': author_data}
    todump.update(
            {'Chemical formula': inputs['Chemical_formula'],
             'number of data': len(collected_data[0]),
             'number of bulks': len(collected_data[2]),
             'number of clusters': len(collected_data[5]),
             'code': inputs['ab_initio_code'],
             'code_version': code_version,
             'pps': pps,
             'protocol': protocol
            }
    )
    # store
    with open(os.path.join('.','exported_data','metadata.yaml'), 'w', encoding='utf-8') as fhandle:
        yaml.dump(todump, fhandle, default_flow_style=False, sort_keys=False, allow_unicode=True)
    with open(os.path.join('.','exported_data','training_data.json'), 'w', encoding='utf8') as fhandle:
        json.dump(collected_data[0], fhandle)
    # plots
    plot_1(collected_data[2], collected_data[3], collected_data[4], collected_data[5], collected_data[6], collected_data[1])
    plot_2(collected_data[7])
    plot_3(collected_data[0])
