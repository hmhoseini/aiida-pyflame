import sys
from aiida.orm import Group
from aiida_pyflame.workflows.settings import groups, log_file

def log_write(txt):
    """ Write to log file
    """
    with open(log_file, 'a', encoding='utf8') as fhandle:
        fhandle.write(txt)

def previous_run_exist_check():
    """ Check if unfinished job exist
    """
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
    if len(active_groups) != 0:
        for a_a_g in active_groups:
            log_write('>>> ERROR: unfinished workflow(s) in group {} (pk: {}) <<<'.format(a_a_g.label, a_a_g.pk)+'\n')
        sys.exit()

def group_is_empty_check(group_label):
    """ Check if a group is empty
    """
    try:
        group = Group.get(label=group_label)
    except:
        return True
    if not group.is_empty:
        log_write('>>> ERROR: group {} (pk: {}) is not empty <<<'.format(group.label, group.pk)+'\n')
        sys.exit()
