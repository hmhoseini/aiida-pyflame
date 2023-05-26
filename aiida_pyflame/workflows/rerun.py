import sys
from aiida.orm import Group
from aiida_pyflame.workflows.core import log_write
from aiida_pyflame.workflows.settings import groups

def find_active_group():
    active_groups = []
    for a_group_label in groups['workflows_group_list']:
        try:
            a_group = Group.get(label=a_group_label)
        except:
            continue
        for a_node in a_group.nodes:
            if not a_node.is_terminated:
                active_groups.append(a_group)
                break
    if len(active_groups) == 0:
        log_write('>>> nothing to do <<<'+'\n')
        sys.exit()
    if len(active_groups) > 1:
        log_write('>>> ERROR: more than one group is active (pk: {}) <<<'.format(active_groups)+'\n')
        sys.exit()
    return active_groups[0]

def rerun():
    active_group = find_active_group()
    if 'gensymcrys' in active_group.label:
        pass
    elif 'step3' in active_group.label:
        pass
    elif 'minimahopping' in active_group.label:
        pass
    elif 'singlepoint' in active_group.label:
        pass
    else:
        log_write('>>> cannot rerun {} <<<'.format(active_group.label)+'\n')
