import os
from aiida.orm import Group
from aiida.manage.configuration import load_profile

load_profile()
group, _ = Group.collection.get_or_create('imported_calculation_nodes')
group.clear()

os.system('verdi archive import -G imported_calculation_nodes exported_calculation_nodes.aiida')
