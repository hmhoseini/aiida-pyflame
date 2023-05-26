import sys
from time import sleep
from datetime import datetime
from aiida_pyflame.codes.utils import get_allowed_n_atom_for_compositions
from aiida_pyflame.codes.flame.flame_launch_calculations import GenSymCrysSubmissionController
from aiida_pyflame.codes.flame.gensymcrys import store_gensymcrys_structures
from aiida_pyflame.workflows.core import log_write, previous_run_exist_check, group_is_empty_check
from aiida_pyflame.workflows.settings import inputs, job_script, steps_status

def step_2():
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 2'+'\n')
    log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    log_write('random structure generation with gensymcrys'+'\n')
    # check
    previous_run_exist_check()
    group_is_empty_check('wf_gensymcrys')

    composition_list = inputs['Chemical_formula']
    if len(composition_list) ==  0:
        log_write('>>> ERROR: no composition is provided <<<'+'\n')
        sys.exit()
    data_dict = {}
    for a_comp in composition_list:
        allowed_n_atom_bulk, allowed_n_atom_reference = get_allowed_n_atom_for_compositions([a_comp])
        allowed_n_atom = allowed_n_atom_bulk + list(set(allowed_n_atom_reference) - set(allowed_n_atom_bulk))
        attempts = max(round(5*(inputs['max_number_of_bulk_structures']/len(inputs['bulk_number_of_atoms']) +\
                              inputs['max_number_of_reference_structures']/len(inputs['reference_number_of_atoms']))/230), 1)
        data_dict[a_comp] = [allowed_n_atom, attempts]
    # submit jobs
    controller = GenSymCrysSubmissionController(
        group_label='wf_gensymcrys',
        max_concurrent=job_script['gensymcrys']['number_of_jobs'],
        data_dict=data_dict)
    # wait until all jobs are done
    while controller.num_to_run > 0 or controller.num_active_slots > 0:
        if controller.num_to_run > 0:
            controller.submit_new_batch(dry_run=False)
        sleep(60)
    # store resutls
    store_gensymcrys_structures()
    log_write('STEP 2 ended'+'\n')
    log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    return steps_status[2]
