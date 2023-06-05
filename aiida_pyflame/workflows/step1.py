import os
import sys
import json
from datetime import datetime
from aiida.orm import Group
from aiida_pyflame.codes.utils import get_known_structures
from aiida_pyflame.workflows.core import log_write, previous_run_exist_check
from aiida_pyflame.workflows.settings import inputs, output_dir, groups, steps_status

def step_1():
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    log_write('Starting PyFLAME'+'\n')
    log_write('STEP 1'+'\n')
    log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    # check
    previous_run_exist_check()
    # create/clear groups
    for group_list in groups.values():
        for a_group_label in group_list:
            group, _ = Group.objects.get_or_create(a_group_label)
            group.clear()
    # composition list
    composition_list = inputs['Chemical_formula']
    if len(composition_list) == 0:
        log_write('>>> ERROR: no composition is provided <<<'+'\n')
        sys.exit()
    # get_known_structures
    known_structures, vpas = get_known_structures(composition_list)
    if not inputs['from_db'] and not inputs['from_local_db']:
        log_write('>>> WARNING: No database is specified <<<'+'\n')
    elif len(known_structures) > 0:
        log_write('Number of atomic structures with the given number of atoms in given databases:{}'.format(len(known_structures))+'\n')
    else:
        log_write('>>> WARNING: No bulk structure with given number of atoms was found in databases <<<'+'\n')
    # store
    with open(os.path.join(output_dir, 'known_bulk_structures.json'),'w', encoding='utf-8') as fhandle:
        json.dump(known_structures, fhandle)
    with open(os.path.join(output_dir, 'vpa.dat'), 'w', encoding='utf-8') as fhandle:
        fhandle.writelines(['%s\n' % vpa  for vpa in vpas])
    log_write('STEP 1 ended'+'\n')
    log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    return steps_status[1]
