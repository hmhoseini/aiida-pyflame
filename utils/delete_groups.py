import os
from aiida.orm import Group
import aiida_pyflame.workflows.settings as settings

for group_list in settings.groups.values():
    for a_group_label in group_list:
        try:
            group = Group.get(label=a_group_label)
        except:
            continue
        os.system("verdi group delete -f "+ str(group.pk))
