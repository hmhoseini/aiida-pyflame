import sys
from time import sleep
from random import sample
from pymatgen.core.structure import Structure, Molecule
from aiida.orm import Group
from aiida.plugins import DataFactory
from aiida_pyflame.codes.utils import get_time, get_reference_structures, get_structures_from_local_db, get_allowed_n_atom_for_compositions, is_structure_valid, store_calculation_nodes
from aiida_pyflame.workflows.core import log_write, previous_run_exist_check, group_is_empty_check, report
from aiida_pyflame.workflows.settings import inputs, steps_status, job_script

def read_structures():
    """ Read known and random bulk structures
    """
    random_structures_group = Group.collection.get(label='random_structures')
    random_bulk_structures_dict = {}
    composition_list = inputs['Chemical_formula']
    if not composition_list:
        log_write('>>> ERROR: no composition is provided <<<'+'\n')
        sys.exit()

    allowed_n_atom_bulk = get_allowed_n_atom_for_compositions(composition_list)

    if inputs['bulk_number_of_atoms'] and inputs['number_of_bulk_structures'] != 0 and allowed_n_atom_bulk:
        n_struct_geopt = int(inputs['number_of_bulk_structures']/len(allowed_n_atom_bulk))
    else:
        log_write('>>> ERROR: data for step 3 is not complete. Check input.yaml <<<'+'\n')
        sys.exit()
    # read random bulk structures
    for a_node in random_structures_group.nodes:
        if int(a_node.label) in allowed_n_atom_bulk:
            random_bulk_structures_dict[int(a_node.label)] = a_node.get_dict()[a_node.label]
    # get reference structures, if any
    reference_structures = get_reference_structures(EAH=False)
    # from local db
    bulk_structures, molecule_structures = get_structures_from_local_db()
    if bulk_structures:
        log_write(f'Number of bulk structures from the local database: {len(bulk_structures)}'+'\n')
    elif inputs['from_local_db']:
        log_write(' >>> WARNING: no bulk strucutre is availalbe in the local database <<<'+'\n')
    if molecule_structures:
        log_write(f'Number of molecule structures from the local database: {len(molecule_structures)}'+'\n')
    elif inputs['from_local_db']:
        log_write(' >>> WARNING: no molecule strucutre is availalbe in the local database <<<'+'\n')

    return n_struct_geopt, reference_structures, random_bulk_structures_dict, bulk_structures, molecule_structures

def add_structures_to_parent_group():
    """ add structures to parent groups
    """
    pg_step3_group = Group.collection.get(label='pg_step3')
    StructureData = DataFactory('structure')
    n_struct_geopt, reference_structures, random_bulk_structures_dict, bulk_structures, molecule_structures = read_structures()
    cluster_list = []
    for a_key in random_bulk_structures_dict.keys():
        indices_schm1_geopt = []
        indices_schm2_geopt = []

        indices = list(range(len(random_bulk_structures_dict[a_key])))
        n_struct_schm1_geopt = int(0.5 * n_struct_geopt)
        n_struct_schm2_geopt = n_struct_geopt - n_struct_schm1_geopt
        if len(indices) >= n_struct_schm1_geopt:
            indices_schm1_geopt = sample(indices, n_struct_schm1_geopt)
        else:
            log_write(f'>>> WARNING: not enough structures with {str(a_key)} atoms for optimization with scheme 1 <<<'+'\n')
            indices_schm1_geopt = indices

        for rem in indices_schm1_geopt:
            indices.remove(rem)

        if len(indices) >= n_struct_schm2_geopt:
            indices_schm2_geopt = sample(indices, n_struct_schm2_geopt)
        else:
            log_write(f' >>> WARNING: not enough structures with {str(a_key)} atoms for optimization with scheme 2 <<<'+'\n')
            indices_schm2_geopt = indices

        for rem in indices_schm2_geopt:
            indices.remove(rem)

        for i, indx in enumerate(indices_schm1_geopt):
            a_structure = Structure.from_dict(random_bulk_structures_dict[a_key][indx])
            s1strct_node = StructureData(pymatgen=a_structure).store()
            s1strct_node.label = 'scheme1'
            s1strct_node.base.extras.set('job', 'scheme1-'+str(i+1)+'_'+str(a_key)+'-atoms')
            pg_step3_group.add_nodes(s1strct_node)
            if a_key in inputs['cluster_number_of_atoms']:
                cluster_list.append(a_structure)
        for i, indx in enumerate(indices_schm2_geopt):
            a_structure = Structure.from_dict(random_bulk_structures_dict[a_key][indx])
            s2strct_node = StructureData(pymatgen=a_structure).store()
            s2strct_node.label = 'scheme2'
            s2strct_node.base.extras.set('job', 'scheme2-'+str(i+1)+'_'+str(a_key)+'-atoms')
            pg_step3_group.add_nodes(s2strct_node)
            if a_key in inputs['cluster_number_of_atoms']:
                cluster_list.append(a_structure)
        if inputs['cluster_calculation'] and cluster_list:
            boxed_molecule = []
            for a_struct in cluster_list:
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
                molecule = Molecule(a_struct.species, cart_coords)
                b_m = molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster)
                if not is_structure_valid(b_m, False, False, False, False, True)[0]:
                    continue
                boxed_molecule.append(b_m)
            if len(boxed_molecule) > len(indices_schm1_geopt) :
                selected_boxed_molecule = sample(boxed_molecule, len(indices_schm1_geopt))
            else:
                selected_boxed_molecule = boxed_molecule
            for i, a_boxed_molecule in enumerate(selected_boxed_molecule):
                bmstrct_node = StructureData(pymatgen=a_boxed_molecule).store()
                bmstrct_node.label = 'cluster'
                bmstrct_node.base.extras.set('job', 'cluster-'+str(i)+'_'+str(a_key)+'-atoms')
                pg_step3_group.add_nodes(bmstrct_node)

    for i, ref_strct in enumerate(reference_structures+bulk_structures):
        a_structure = Structure.from_dict(ref_strct)
        nat = len(a_structure.sites)
        rfstrct_node = StructureData(pymatgen=a_structure).store()
        rfstrct_node.label = 'scheme3'
        rfstrct_node.base.extras.set('job', 'scheme3-'+str(i+1)+'_'+str(nat)+'-atoms')
        pg_step3_group.add_nodes(rfstrct_node)

    if molecule_structures:
        for i, molecule in enumerate(molecule_structures):
            boxed_molecule = []
            a_struct = Structure.from_dict(molecule)
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
            if max(a_cluster, b_cluster, c_cluster) > 50: #max. box size 50 A
                continue
            molecule = Molecule(a_struct.species, cart_coords)
            boxed_molecule.append(molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster))
        for i, a_boxed_molecule in enumerate(selected_boxed_molecule):
            nat = len(a_boxed_molecule.sites)
            bmstrct_node = StructureData(pymatgen=a_boxed_molecule).store()
            bmstrct_node.label = 'molecule'
            bmstrct_node.base.extras.set('job', 'molecule-'+str(i)+'_'+str(nat)+'-atoms')
            pg_step3_group.add_nodes(bmstrct_node)

def step_3():
    """ Step 3
    """
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 3'+'\n')
    log_write(f'start time: {get_time()}'+'\n')
    # check
    previous_run_exist_check()
    group_is_empty_check('wf_step3')
    # clear groups
    for a_group_label in ['pg_step3', 'results_step3']:
        a_group, _ = Group.collection.get_or_create(a_group_label)
        a_group.clear()
    # add structures
    add_structures_to_parent_group()
    # submit jobs
    if 'SIRIUS' in inputs['ab_initio_code'] or 'QS' in inputs['ab_initio_code']:
        from aiida_pyflame.codes.cp2k.cp2k_launch_calculations import CP2KSubmissionController
        log_write(f'Ab-initio calculations with {inputs["ab_initio_code"]}'+'\n')
        controller = CP2KSubmissionController(
            parent_group_label='pg_step3',
            group_label='wf_step3',
            max_concurrent=job_script['geopt']['number_of_jobs'],
            QSorSIRIUS=inputs['ab_initio_code'])
    elif inputs['ab_initio_code']=='VASP':
        from aiida_pyflame.codes.vasp.vasp_launch_calculations import VASPSubmissionController
        log_write('Ab-initio calculations with VASP'+'\n')
        controller = VASPSubmissionController(
            parent_group_label='pg_step3',
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
    total_computing_time, submitted_jobs, finished_job = report('wf_step3')
    log_write(f'submitted jobs: {submitted_jobs}, succesful jobs: {finished_job}'+'\n')
    log_write(f'total computing time: {round(total_computing_time, 2)} core-hours'+'\n')
    log_write('STEP 3 ended'+'\n')
    log_write(f'end time: {get_time()}'+'\n')
    if not steps_status[3]:
        store_calculation_nodes()
        log_write('End of the step 3. Bye!'+'\n')
    return steps_status[3]
