import sys
import yaml
from time import sleep
from collections import defaultdict
from aiida.orm import Group, Dict
from aiida_pyflame.codes.utils import get_allowed_n_atom_for_compositions, get_time, store_calculation_nodes
from aiida_pyflame.codes.flame.flame_launch_calculations import GenSymCrysSubmissionController
from aiida_pyflame.workflows.core import log_write, previous_run_exist_check, group_is_empty_check
from aiida_pyflame.workflows.settings import inputs, job_script, steps_status
from aiida_pyflame.codes.utils import is_structure_valid
from aiida_pyflame.codes.flame.core import conf2pymatgenstructure

def collect_random_structures(outfile):
    random_bulk_structures = []
    try:
        confs = yaml.load_all(outfile, Loader=yaml.SafeLoader)
    except:
        return None, None
    pymatgen_structures = conf2pymatgenstructure(confs)
    nat = len(pymatgen_structures[0].sites)
    for a_pymatgen_structure in pymatgen_structures:
        if is_structure_valid(a_pymatgen_structure, False, False, True, False, False)[0]:
            random_bulk_structures.append(a_pymatgen_structure.as_dict())
    return nat, random_bulk_structures

def store_step2_results():
    todump_dict = defaultdict(list)
    wf_step2_group = Group.collection.get(label='wf_step2')
    random_structures_group = Group.collection.get(label='random_structures')
    random_structures_group.clear()
    for a_wf_node in wf_step2_group.nodes:
        a_node = a_wf_node.called[-1]
        if not a_node.is_finished_ok:
            continue
        output_folder = a_node.outputs.retrieved
        with output_folder.open('posout.yaml', 'rb') as fhandle:
            nat, random_bulk_structures = collect_random_structures(fhandle)
        if nat and random_bulk_structures:
            todump_dict[str(nat)].extend(random_bulk_structures)
    for a_key in todump_dict.keys():
        a_node = Dict({a_key: todump_dict[a_key]}).store()
        a_node.label = a_key
        random_structures_group.add_nodes(a_node)
        log_write(f'{len(todump_dict[a_key])} random bulk structures with {a_key} atoms are generated'+'\n')

def step_2():
    """ Step 2
    """    
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 2'+'\n')
    log_write(f'start time: {get_time()}'+'\n')
    log_write('random structure generation with gensymcrys'+'\n')
    # check
    previous_run_exist_check()
    group_is_empty_check('wf_step2')

    composition_list = inputs['Chemical_formula']
    if len(composition_list) ==  0:
        log_write('>>> ERROR: no composition is provided <<<'+'\n')
        sys.exit()
    data_dict = {}
    for a_comp in composition_list:
        allowed_n_atom = get_allowed_n_atom_for_compositions([a_comp])
        attempts = max(round(5*(inputs['number_of_bulk_structures']/len(inputs['bulk_number_of_atoms']))/230), 1)
        data_dict[a_comp] = [allowed_n_atom, attempts]
    # submit jobs
    controller = GenSymCrysSubmissionController(
        group_label='wf_step2',
        max_concurrent=job_script['gensymcrys']['number_of_jobs'],
        data_dict=data_dict)
    # wait until all jobs are done
    while controller.num_to_run > 0 or controller.num_active_slots > 0:
        if controller.num_to_run > 0:
            controller.submit_new_batch(dry_run=False)
        sleep(60)
    # store resutls
    store_step2_results()
    log_write('STEP 2 ended'+'\n')
    log_write(f'end time: {get_time()}'+'\n')
    if not steps_status[2]:
        store_calculation_nodes()
        log_write('End of the step 2. Bye!'+'\n')
    return steps_status[2]
