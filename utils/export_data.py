import os
import yaml
import argparse
from aiida.orm import Dict, Group, load_node
from aiida_pyflame.workflows.settings import run_dir, PyFLAME_directory
from aiida.manage.configuration import load_profile

load_profile()

def add_input_node(pks):
    with open(os.path.join(run_dir,'input.yaml'), 'r', encoding='utf8') as fhandle:
        inputs = yaml.safe_load(fhandle)
    a_node = Dict(inputs)
    a_node.label = 'inputs'
    store = a_node.store()
    pks.append(store.pk)
    return pks, inputs

def add_author_data(pks):
    with open(os.path.join(run_dir,'author_data.yaml'), 'r', encoding='utf8') as fhandle:
        author_data = yaml.safe_load(fhandle)
    a_node = Dict(author_data)
    a_node.label = 'author_data'
    store = a_node.store()
    pks.append(store.pk)
    return pks

def add_protocol(pks, inputs):
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
    return pks

def add_known_structures(pks):
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        pks.append(a_node.pk)
    return pks

def add_calculation_nodes(pks):
    calculation_nodes_group = Group.collection.get(label='calculation_nodes')
    for a_node in calculation_nodes_group.nodes:
        pks.extend(a_node.get_list())
    return pks

def add_nodes(pks):
    tmp_group, _ = Group.collection.get_or_create('tmp_group')
    tmp_group.clear()
    for a_pk in pks:
        try:
            a_node = load_node(a_pk)
        except:
            continue
        tmp_group.add_nodes(a_node)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='export_data.py',
                    description='Exprot calculation node to an AiiDA archive')

    parser.add_argument('filename', help='Output file name *.aiida')
    args = parser.parse_args()


    pks = []
    pks, inputs = add_input_node(pks)
    pks = add_author_data(pks)
    pks = add_protocol(pks, inputs)
    pks = add_known_structures(pks)
    pks = add_calculation_nodes(pks)
    add_nodes(pks)
    os.system(f"verdi archive create --no-call-calc-backward --no-call-work-backward --no-create-backward {args.filename} --groups tmp_group")
    with open('node_pks.dat', 'w', encoding='utf-8') as fhandle:
        for a_pk in pks:
            fhandle.write(f'{a_pk}'+'\n')

    tmp_group, _ = Group.collection.get_or_create('tmp_group')
    Group.collection.delete(tmp_group.pk)


