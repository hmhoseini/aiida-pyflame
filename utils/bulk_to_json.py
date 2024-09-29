import os
import json
from ase.io import read
from aiida.plugins import DataFactory
from aiida.manage.configuration import load_profile

load_profile()
StructureData = DataFactory("core.structure")

def read_files(path):
    bulks = []
    for a_file in os.listdir(path):
        if not os.path.isfile(os.path.join(path, a_file)):
            continue
        if a_file.endswith('.cif'):
            ase_structure = read(a_file, 0, format='cif')
            bulks.append(StructureData(ase=ase_structure))
        if a_file.endswith('.vasp') or 'POSCAR' in a_file:
            ase_structure = read(a_file, 0, format='vasp')
            bulks.append(StructureData(ase=ase_structure))
    return bulks

def write_json(bulks, path):
    to_dump = []
    for a_bulk in bulks:
        to_dump.append(a_bulk.get_pymatgen_structure().as_dict())
    with open(path, 'w', encoding='utf-8') as fhandle:
        json.dump(to_dump, fhandle)

if __name__ == "__main__":
    path_in = os.path.join('.')
    bulks = read_files(path_in)
    path_out = os.path.join('.', 'bulk_structures.json')
    write_json(bulks, path_out)
