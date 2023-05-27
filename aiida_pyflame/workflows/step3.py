import sys
import os
import json
from time import sleep
from datetime import datetime
from random import sample
from collections import defaultdict
from pymatgen.core.structure import Structure, Molecule
from aiida.orm import Group
from aiida.plugins import DataFactory
from aiida_pyflame.codes.utils import get_allowed_n_atom_for_compositions, is_structure_valid
from aiida_pyflame.workflows.core import log_write, previous_run_exist_check, group_is_empty_check
from aiida_pyflame.workflows.settings import inputs, output_dir, steps_status, job_script

def read_structures():
    """ read known and random bulk structures
    """
    # get composition list
    composition_list = inputs['Chemical_formula']
    if len(composition_list) ==  0:
        log_write('>>> ERROR: no composition is provided <<<'+'\n')
        sys.exit()

    allowed_n_atom_bulk, allowed_n_atom_reference = get_allowed_n_atom_for_compositions(composition_list)

    if len(inputs['bulk_number_of_atoms']) > 0 and inputs['max_number_of_bulk_structures'] != 0 and len(allowed_n_atom_bulk) > 0:
        n_struct_geopt = int(inputs['max_number_of_bulk_structures']/len(allowed_n_atom_bulk))
    else:
        log_write('>>> ERROR: data for step 2 is not complete. Check input.yaml <<<'+'\n')
        sys.exit()

    if inputs['max_number_of_reference_structures'] != 0 and len(allowed_n_atom_reference) > 0 :
        n_struct_geopt_reference = int(inputs['max_number_of_reference_structures']/len(allowed_n_atom_reference))
    else:
        n_struct_geopt_reference = 0
        log_write('>>> WARNING: no reference structure for optimization <<<'+'\n')
    # read known structures, if any
    known_bulk_structures = defaultdict(list)
    if os.path.exists(os.path.join(output_dir,'known_bulk_structures.json')):
        log_write('Reading known bulk structures'+'\n')
        with open(os.path.join(output_dir,'known_bulk_structures.json'), 'r', encoding='utf-8') as fhandle:
            tmp_dict = json.loads(fhandle.read())
        for a_known_structure in tmp_dict:
            pymatgen_structure = Structure.from_dict(a_known_structure)
            if len(pymatgen_structure.sites) in allowed_n_atom_reference:
                known_bulk_structures[len(pymatgen_structure.sites)].append(pymatgen_structure)
    # read generated random structures
    random_bulk_structures_dict = defaultdict(list)
    log_write('Reading random bulk structures'+'\n')
    if os.path.exists(os.path.join(output_dir,'random_bulk_structures.json')):
        with open(os.path.join(output_dir,'random_bulk_structures.json'), 'r', encoding='utf-8') as fhandle:
            tmp_dict = json.loads(fhandle.read())
        for keys in tmp_dict.keys():
            if int(keys) in allowed_n_atom_bulk+allowed_n_atom_reference:
                random_bulk_structures_dict[int(keys)].extend(tmp_dict[keys])
    else:
        log_write('>>> ERROR: no random bulk structure was found <<<'+'\n')
        sys.exit()
    return n_struct_geopt_reference, n_struct_geopt, known_bulk_structures, random_bulk_structures_dict

def add_structures_to_parent_group():
    """ add structures to parent groups
    """
    structures_step3_group = Group.get(label='structures_step3')
    StructureData = DataFactory('structure')
    n_struct_geopt_reference, n_struct_geopt, known_bulk_structures, random_bulk_structures_dict = read_structures()
    cluster_list = []

    for a_n_a in random_bulk_structures_dict.keys():
        indices_rfnc_geopt = []
        indices_schm1_geopt = []
        indices_schm2_geopt = []

        indices = list(range(len(random_bulk_structures_dict[a_n_a])))
        if a_n_a in inputs['reference_number_of_atoms'] and n_struct_geopt_reference > 0:
            if len(indices) < n_struct_geopt_reference:
                log_write('>>> WARNING: not enough structures with {} atoms as references <<<'.format(str(a_n_a))+'\n')
            if len(known_bulk_structures[a_n_a]) < n_struct_geopt_reference:
                indices_rfnc_geopt = sample(indices, n_struct_geopt_reference-len(known_bulk_structures[a_n_a]))
                for i, indx in enumerate(indices_rfnc_geopt):
                    pymatgen_structure = Structure.from_dict(random_bulk_structures_dict[a_n_a][indx])
                    nat = len(pymatgen_structure.sites)
                    rfstrct_node = StructureData(pymatgen=pymatgen_structure).store()
                    rfstrct_node.label = 'reference'
                    rfstrct_node.base.extras.set('job', 'reference-'+str(i+1)+'_'+str(nat)+'-atoms')
                    structures_step3_group.add_nodes(rfstrct_node)
                for i, a_known_struct in enumerate(known_bulk_structures[a_n_a]):
                    nat = len(a_known_struct.sites)
                    knstrct_node = StructureData(pymatgen=a_known_struct).store()
                    knstrct_node.label = 'reference'
                    knstrct_node.base.extras.set('job', 'known-structure-'+str(i+1)+'_'+str(nat)+'-atoms')
                    structures_step3_group.add_nodes(knstrct_node)
            else:
                for i, a_known_struct in enumerate(sample(known_bulk_structures[a_n_a], n_struct_geopt_reference)):
                    nat = len(a_known_struct.sites)
                    knstrct_node = StructureData(pymatgen=a_known_struct).store()
                    knstrct_node.label = 'reference'
                    knstrct_node.base.extras.set('job', 'known-structure-'+str(i+1)+'_'+str(nat)+'-atoms')
                    structures_step3_group.add_nodes(knstrct_node)
            for rem in indices_rfnc_geopt:
                indices.remove(rem)
        if a_n_a in inputs['bulk_number_of_atoms']:
            n_struct_schm1_geopt = int(0.5 * n_struct_geopt)
            n_struct_schm2_geopt = n_struct_geopt - n_struct_schm1_geopt
            if len(indices) >= n_struct_schm1_geopt:
                indices_schm1_geopt = sample(indices, n_struct_schm1_geopt)
            else:
                log_write('>>> WARNING: not enough structures with {} atoms for optimization with scheme 1 <<<'.format(str(a_n_a))+'\n')
                indices_schm1_geopt = indices

            for rem in indices_schm1_geopt:
                indices.remove(rem)

            if len(indices) >= n_struct_schm2_geopt:
                indices_schm2_geopt = sample(indices, n_struct_schm2_geopt)
            else:
                log_write(' >>> WARNING: not enough structures with {} atoms for optimization with scheme 2 <<<'.format(str(a_n_a))+'\n')
                indices_schm2_geopt = indices

            for rem in indices_schm2_geopt:
                indices.remove(rem)

            for i, indx in enumerate(indices_schm1_geopt):
                pymatgen_structure = Structure.from_dict(random_bulk_structures_dict[a_n_a][indx])
                nat = len(pymatgen_structure.sites)
                s1strct_node = StructureData(pymatgen=pymatgen_structure).store()
                s1strct_node.label = 'scheme1'
                s1strct_node.base.extras.set('job', 'scheme1-'+str(i+1)+'_'+str(nat)+'-atoms')
                structures_step3_group.add_nodes(s1strct_node)
                if len(pymatgen_structure.sites) in inputs['cluster_number_of_atoms']:
                    cluster_list.append(pymatgen_structure)
            for i, indx in enumerate(indices_schm2_geopt):
                pymatgen_structure = Structure.from_dict(random_bulk_structures_dict[a_n_a][indx])
                nat = len(pymatgen_structure.sites)
                s2strct_node = StructureData(pymatgen=pymatgen_structure).store()
                s2strct_node.label = 'scheme2'
                s2strct_node.base.extras.set('job', 'scheme2-'+str(i+1)+'_'+str(nat)+'-atoms')
                structures_step3_group.add_nodes(s2strct_node)
                if len(pymatgen_structure.sites) in inputs['cluster_number_of_atoms']:
                    cluster_list.append(pymatgen_structure)

    if inputs['cluster_calculation'] and len(inputs['cluster_number_of_atoms']) > 0:
        boxed_molecule = []
        for a_struct in cluster_list:
            nat = len(a_struct.sites)
            cart_coords = a_struct.cart_coords
            maxx = max(cart_coords[:,0:1])[0]
            minx = min(cart_coords[:,0:1])[0]
            maxy = max(cart_coords[:,1:2])[0]
            miny = min(cart_coords[:,1:2])[0]
            maxz = max(cart_coords[:,2:3])[0]
            minz = min(cart_coords[:,2:3])[0]
            a_cluster = maxx-minx+inputs['vacuum_length']
            b_cluster = maxy-miny+inputs['vacuum_length']
            c_cluster = maxz-minz+inputs['vacuum_length']
            if max(a_cluster, b_cluster, c_cluster) > inputs['box_size']:
                continue
            molecule = Molecule(a_struct.species, cart_coords)
            boxed_molecule.append(molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster))
        if len(boxed_molecule) > len(indices_schm1_geopt):
            selected_boxed_molecule = sample(boxed_molecule, len(indices_schm1_geopt))
        else:
            selected_boxed_molecule = boxed_molecule
        for i, a_boxed_molecule in enumerate(selected_boxed_molecule):
            bmstrct_node = StructureData(pymatgen=a_boxed_molecule).store()
            bmstrct_node.label = 'cluster'
            bmstrct_node.base.extras.set('job', 'cluster-'+str(i)+'_'+str(nat)+'-atoms')
            structures_step3_group.add_nodes(bmstrct_node)

def store_step3_results():
    """ store results
    """
    results_step3_group = Group.get(label='results_step3')
    epas = []
    seeds_bulk = []
    seeds_cluster = []
    for a_node in results_step3_group.nodes:
        if not a_node.is_finished_ok:
            continue
        if 'VASP' in inputs['ab_initio_code']:
            if not a_node.outputs.misc.dict.run_status['electronic_converged']:
                continue
            total_energy = float(a_node.outputs.energies.get_array('energy_extrapolated_electronic')[-1])
            if total_energy > 0:
                continue
            pymatgen_structure = a_node.outputs.structure.get_pymatgen()

        if 'SIRIUS' in inputs['ab_initio_code'] or 'GTH' in inputs['ab_initio_code']:
            total_energy = float(a_node.outputs.output_parameters.dict.energy)
            if total_energy > 0:
                continue
            pymatgen_structure = a_node.outputs.output_structure.get_pymatgen()

        nat = len(pymatgen_structure.sites)
        epa = total_energy/nat
        epas.append(epa)
        if is_structure_valid(pymatgen_structure, False, True, False):
            if nat in inputs['bulk_number_of_atoms'] and 'bulk' in a_node.label:
                seeds_bulk.append(pymatgen_structure.as_dict())
            if nat in inputs['cluster_number_of_atoms'] and 'cluster' in a_node.label:
                cart_coords = pymatgen_structure.cart_coords
                maxx = max(cart_coords[:,0:1])[0]
                minx = min(cart_coords[:,0:1])[0]
                maxy = max(cart_coords[:,1:2])[0]
                miny = min(cart_coords[:,1:2])[0]
                maxz = max(cart_coords[:,2:3])[0]
                minz = min(cart_coords[:,2:3])[0]
                a_cluster = maxx-minx+inputs['vacuum_length']
                b_cluster = maxy-miny+inputs['vacuum_length']
                c_cluster = maxz-minz+inputs['vacuum_length']
                if max(a_cluster, b_cluster, c_cluster) > inputs['box_size']:
                    continue
                molecule = Molecule(pymatgen_structure.species, cart_coords)
                boxed_molecule = molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster)
                seeds_cluster.append(boxed_molecule.as_dict())

    with open(os.path.join(output_dir,'epa.dat'), 'w', encoding='utf8') as fhandle:
        fhandle.write('min_epa: {}'.format(min(epas))+'\n')
        fhandle.write('max_epa: {}'.format(max(epas))+'\n')
        fhandle.write('ave_epa: {}'.format(sum(epas)/len(epas))+'\n')
    with open(os.path.join(output_dir,'min_epa.dat'), 'w', encoding='utf8') as fhandle:
        fhandle.write('{}'.format(min(epas))+'\n')
    with open(os.path.join(output_dir,'seeds_bulk.json'), 'w', encoding='utf-8') as fhandle:
        json.dump(seeds_bulk, fhandle)
    with open(os.path.join(output_dir,'seeds_cluster.json'), 'w', encoding='utf-8') as fhandle:
        json.dump(seeds_cluster, fhandle)
    log_write('Number of bulk seeds: {}'.format(len(seeds_bulk))+'\n')
    log_write('Number of cluster seeds: {}'.format(len(seeds_cluster))+'\n')

def step_3():
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 3'+'\n')
    log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    # check
    previous_run_exist_check()
    group_is_empty_check('wf_step3')
    # clear groups
    for a_group_label in ['structures_step3', 'results_step3']:
        a_group, _ = Group.objects.get_or_create(a_group_label)
        a_group.clear()
    # add structures
    add_structures_to_parent_group()
    # submit jobs
    if 'SIRIUS' in inputs['ab_initio_code'] or 'GTH' in inputs['ab_initio_code']:
        from aiida_pyflame.codes.cp2k.cp2k_launch_calculations import CP2KSubmissionController
        log_write('Ab-initio calculations with {}'.format(inputs['ab_initio_code'])+'\n')
        controller = CP2KSubmissionController(
            parent_group_label='structures_step3',
            group_label='wf_step3',
            max_concurrent=job_script['geopt']['number_of_jobs'],
            GTHorSIRIUS=inputs['ab_initio_code'])
    elif inputs['ab_initio_code']=='VASP':
        from aiida_pyflame.codes.vasp.vasp_launch_calculations import VASPSubmissionController
        log_write('Ab-initio calculations with VASP'+'\n')
        controller = VASPSubmissionController(
            parent_group_label='structures_step3',
            group_label='wf_step3',
            max_concurrent=job_script['geopt']['number_of_jobs'])
    else:
        log_write('>>> ERROR: no ab_initio code is provided <<<'+'\n')
        sys.exit()
    # wait until all jobs are done
    while controller.num_to_run > 0 or controller.num_active_slots > 0:
        if controller.num_to_run > 0:
            controller.submit_new_batch(dry_run=False)
        sleep(60)
    # store
    store_step3_results()
    log_write('STEP 3 ended'+'\n')
    log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    return steps_status[3]
