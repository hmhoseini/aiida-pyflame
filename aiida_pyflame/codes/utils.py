import os
import json
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.ext.optimade import OptimadeRester
from aiida_pyflame.workflows.settings import inputs, output_dir, run_dir, Flame_dir

def get_pertured_failed_structures(cycle_number):
    """ Perturb fialed crystal structures.
    """
    c_no = int(cycle_number.split('-')[-1])
    min_d_prefactor = inputs['min_distance_prefactor'] * ((100-float(inputs['descending_prefactor']))/100)**(c_no-1)\
                      if inputs['descending_prefactor'] else inputs['min_distance_prefactor']

    failed_b_structures = []
    failed_c_structures = []

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','failed_bulk.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            failed_b_structures.extend(json.loads(fhandle.read()))

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','failed_cluster.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            failed_c_structures.extend(json.loads(fhandle.read()))

    perturbed_b_structures = []
    perturbed_c_structures = []

    for a_b_struct in failed_b_structures:
        a_b_structure = Structure.from_dict(a_b_struct)
        for p in (0.03, 0.05):
            a_b_s = a_b_structure.copy()
            a_b_s.perturb(p)
            if is_structure_valid(a_b_s, min_d_prefactor, False, False):
                perturbed_b_structures.append(a_b_s)
    for a_c_struct in failed_c_structures:
        a_c_structure = Structure.from_dict(a_c_struct)
        for p in (0.03, 0.05):
            a_c_s = a_c_structure.copy()
            a_c_s.perturb(p)
            if is_structure_valid(a_c_s, min_d_prefactor, False, False):
                perturbed_c_structures.append(a_c_s)
    return perturbed_b_structures, perturbed_c_structures

def get_element_list():
    """ Retruns the list of elements.
    """
    element_list = []
    composition_list = inputs['Chemical_formula']
    if len(composition_list) == 0:
        return None
    for a_composition in composition_list:
        for elmnt in Composition(a_composition).elements:
            if str(elmnt) not in element_list:
                element_list.append(str(elmnt))
    return element_list

def get_known_structures(composition_list):
    """ Retruns retrieved structures (and max/min of their volume/atom)
        available in given databases
    """
    structures_provides = {}
    known_structures = []
    vpas = []
    if inputs['from_db']:
        opt = OptimadeRester(inputs['from_db'], timeout=30)
        for a_composition in composition_list:
            optimade_filter = '(chemical_formula_reduced="{}")'.format(a_composition)
            results = opt.get_structures_with_filter(optimade_filter)
            structures_provides.update(results)
    for structures_from_db in structures_provides.values():
        for a_structure_from_db in structures_from_db.values():
            vpas.append(a_structure_from_db.volume/len(a_structure_from_db.sites))
            if len(a_structure_from_db.sites) in inputs['bulk_number_of_atoms']+inputs['reference_number_of_atoms']:
                known_structures.append(a_structure_from_db.as_dict())
            a_structure_from_db_primitive = a_structure_from_db.get_primitive_structure()
            if len(a_structure_from_db_primitive.sites) != len(a_structure_from_db.sites) and\
               len(a_structure_from_db_primitive.sites) in inputs['bulk_number_of_atoms']+inputs['reference_number_of_atoms']:
                known_structures.append(a_structure_from_db_primitive.as_dict())
    if inputs['from_local_db'] and os.path.exists(os.path.join(run_dir,'local_db','known_bulk_structures.json')):
        with open(os.path.join(run_dir,'local_db','known_bulk_structures.json'), 'r', encoding='utf-8') as fhandle:
            structures_from_local_db = json.loads(fhandle.read())
        for a_s_f_l_db in structures_from_local_db:
            a_structure_from_local_db = Structure.from_dict(a_s_f_l_db)
            vpas.append(a_structure_from_local_db.volume/len(a_structure_from_local_db.sites))
            if len(a_structure_from_local_db.sites) in inputs['bulk_number_of_atoms']+inputs['reference_number_of_atoms']:
                known_structures.append(a_structure_from_local_db.as_dict())
            a_structure_from_local_db_primitive = a_structure_from_local_db.get_primitive_structure()
            if len(a_structure_from_local_db_primitive.sites) != len(a_structure_from_local_db.sites) and\
               len(a_structure_from_local_db_primitive.sites) in inputs['bulk_number_of_atoms']+inputs['reference_number_of_atoms']:
                known_structures.append(a_structure_from_local_db_primitive.as_dict())
    if len(vpas) < 3:
        covalent_radius = CovalentRadius.radius
        pre_fact = 1
        for a_composition in composition_list:
            elements = []
            nelement = []
            for elmnt, nelmnt in Composition(a_composition).items():
                elements.append(str(elmnt))
                nelement.append(int(nelmnt))
            vol = 0
            for i in range(len(elements)):
                vol += 8 * covalent_radius[elements[i]]**3 * nelement[i]
                if elements[i] in ['H', 'N', 'O']:
                    pre_fact = pre_fact + nelement[i] * 0.10
            vpas.append(pre_fact * vol/sum(nelement))
        vpas.append(0.8 * min(vpas))
        vpas.append(2.0 * min(vpas))
    minmaxvpa = [min(vpas), max(vpas)]
    return known_structures, minmaxvpa

def get_allowed_n_atom_for_compositions(composition_list):
    """ Returns allowed number of atoms for given crystal structures.
    """
    allowed_n_atom_bulk = []
    allowed_n_atom_reference = []
    for a_n_a in inputs['bulk_number_of_atoms']+inputs['reference_number_of_atoms']:
        for a_comp in composition_list:
            n_elmnt = []
            for n in Composition(a_comp).values():
                n_elmnt.append(int(n))
            n_element = n_elmnt
            if a_n_a % sum(n_element) == 0:
                if a_n_a in inputs['bulk_number_of_atoms'] and a_n_a not in allowed_n_atom_bulk:
                    allowed_n_atom_bulk.append(a_n_a)
                if a_n_a in inputs['reference_number_of_atoms'] and a_n_a not in allowed_n_atom_reference:
                    allowed_n_atom_reference.append(a_n_a)
    return allowed_n_atom_bulk, allowed_n_atom_reference

def is_structure_valid(structure, min_d_prefactor, check_angles, check_vpa):
    """ Check if a crystal structure is a valid one.
        criteria: minimum distance between atoms, angles, and vpa.
    """
    if check_angles:
        angles = structure.lattice.angles
        if angles[0] > 135 or angles[0] < 45 or\
           angles[1] > 135 or angles[1] < 45 or\
           angles[2] > 135 or angles[2] < 45:
            return False
    if check_vpa:
        with open(os.path.join(output_dir,'vpa.dat'), 'r', encoding='utf-8') as fhandle:
            vpas = [float(line.strip()) for line in fhandle]
        vpa = structure.volume/len(structure.sites)
        if vpa < vpas[0] or vpa > vpas[1]:
            return False
    try:
        d_matrix = structure.distance_matrix
    except:
        return False
    for isites in range(len(structure.sites)):
        if abs(max(structure[isites].coords)) > max(structure.lattice.abc):
            return False
        if not structure.get_neighbors(structure[isites], 5):
            return False
        if min_d_prefactor:
            for jsites in range(len(structure.sites)):
                min_d = get_min_d(structure[isites].species_string, structure[jsites].species_string) * min_d_prefactor
                if d_matrix[isites,jsites] > 0 and d_matrix[isites,jsites] < min_d:
                    return False
    return True

def get_min_d(elmnt1, elmnt2, X=False):
    """ Returns allowed minimum distance between two elements.
    """
    covalent_radius = CovalentRadius.radius
    min_d = covalent_radius[elmnt1] + covalent_radius[elmnt2]
    x_prefactor = 1
    if X:
        try:
            x_prefactor = 1 - (abs(Element(elmnt1).X - Element(elmnt2).X)/(3.98 - 0.79)) * 0.2
        except ValueError:
            pass
    return min_d * x_prefactor

def r_cut():
    all_rcuts = []
    structures = []
    n_atom_in_rcut = max(inputs['bulk_number_of_atoms']) * 1.3
    with open(os.path.join(output_dir,'seeds_bulk.json'), 'r', encoding='utf-8') as fhandle:
        structures = json.loads(fhandle.read())
    for a_struct in structures:
        for d in range(6,30):
            if len(Structure.from_dict(a_struct).get_sites_in_sphere([0,0,0],d)) < n_atom_in_rcut:
                continue
            all_rcuts.append(d)
            break
    return sum(all_rcuts)/len(all_rcuts)
