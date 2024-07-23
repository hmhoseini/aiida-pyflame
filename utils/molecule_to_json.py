umport os
import json
from ase.io import read
from aiida.plugins import DataFactory
from aiida.manage.configuration import load_profile

load_profile()
StructureData = DataFactory("structure")

def read_files(path):
    molecules = []
    for a_file in os.listdir(path):
        if not os.path.isfile(os.path.join(path, a_file)):
            continue
        if a_file.endswith('.xyz'):
            ase_structure = read(a_file, 0, format='xyz')
            molecules.append(StructureData(ase=ase_structure))
    return molecules

def write_json(molecules, path):
    to_dump = []
    for a_molecule in molecules:
        to_dump.append(a_molecul.get_pymatgen_molecule().as_dict())
    with open(path, 'w', encoding='utf-8') as fhandle:
        json.dump(to_dump, fhandle)

if __name__ == "__main__":
    path_in = os.path.join('.', 'local_db')
    molecules = read_files(path_in)
    path_out = os.path.join('.', 'molecule_structures.json')
    write_json(molecules, path_out)
