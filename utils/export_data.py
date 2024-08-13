import os
import yaml
from aiida.orm import Dict, Group, load_node
from aiida_pyflame.workflows.settings import run_dir, PyFLAME_directory
from aiida.manage.configuration import load_profile

load_profile()

tmp_group, _ = Group.collection.get_or_create('tmp_group')

pks = []
with open(os.path.join(run_dir,'input.yaml'), 'r', encoding='utf8') as fhandle:
    inputs = yaml.safe_load(fhandle)
a_node = Dict(inputs)
a_node.label = 'inputs'
store = a_node.store()
pks.append(store.pk)

with open(os.path.join('.','author_data.yaml'), 'r', encoding='utf8') as fhandle:
    author_data = yaml.safe_load(fhandle)
a_node = Dict(author_data)
a_node.label = 'author_data'
store = a_node.store()
pks.append(store.pk)

CP2K_input_files_path = os.path.join(run_dir,'cp2k_files')\
                        if inputs['user_specified_CP2K_files']\
                        else os.path.join(PyFLAME_directory,'codes/cp2k','cp2k_files')
VASP_input_files_path = os.path.join(run_dir,'vasp_files')\
                        if inputs['user_specified_VASP_files']\
                        else os.path.join(PyFLAME_directory,'codes/vasp','vasp_files')
if 'VASP' in inputs['ab_initio_code']:
    with open(os.path.join(VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
        vasp_protocol = yaml.safe_load(fhandle)
    single_point_protocol = vasp_protocol['single_point']    
    a_node = Dict(single_point_protocol)
    a_node.label = 'protocol'
    store = a_node.store()
    pks.append(store.pk)
if 'SIRIUS' in inputs['ab_initio_code']:
    with open(os.path.join(CP2K_input_files_path,'protocol_SIRIUS.yml'), 'r', encoding='utf8') as fhandle:
        cp2k_protocol = yaml.safe_load(fhandle)
    single_point_protocol = cp2k_protocol['single_point']
    single_point_protocol['FORCE_EVAL'].pop('SUBSYS')
    a_node = Dict(single_point_protocol)
    a_node.label = 'protocol'
    store = a_node.store()
    pks.append(store.pk)
if 'QS' in inputs['ab_initio_code']:
    with open(os.path.join(CP2K_input_files_path,'protocol_QS.yml'), 'r', encoding='utf8') as fhandle:
        cp2k_protocol = yaml.safe_load(fhandle)
    single_point_protocol = cp2k_protocol['single_point']    
    a_node = Dict(single_point_protocol)
    a_node.label = 'protocol'
    store = a_node.store()
    pks.append(store.pk)

calculation_nodes_group = Group.collection.get(label='calculation_nodes')
calculation_nodes =[]

for a_node in calculation_nodes_group.nodes:
    calculation_nodes.extend(a_node.get_list())

known_structures_group = Group.collection.get(label='known_structures')
known_structures_nodes = []
for a_node in known_structures_group.nodes:
    known_structures_nodes.append(a_node.pk)

all_nodes_pk = known_structures_nodes+calculation_nodes[1:1000]+pks
for a_pk in all_nodes_pk:
    a_node = load_node(a_pk)
    tmp_group.add_nodes(a_node)

os.system('verdi archive create --no-call-calc-backward --no-call-work-backward --no-create-backward exported_calculation_nodes.aiida --groups tmp_group')
Group.collection.delete(tmp_group.pk)
