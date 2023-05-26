import os
import json
from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from mp_api.client import MPRester
from aiida_pyflame.workflows.settings import inputs, output_dir, configs, Flame_dir

def get_pertured_failed_structures(cycle_number):
    c_no = int(cycle_number.split('-')[-1])
    min_d_prefactor = inputs['min_distance_prefactor'] * ((100-float(inputs['descending_prefactor']))/100)**(c_no-1)\
                      if inputs['descending_prefactor'] else inputs['min_distance_prefactor']

    failed_b_structures = []
    failed_c_structures = []

    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','failed_structures.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            f_s = json.loads(fhandle.read())
        failed_b_structures.extend(f_s)
    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','failed_bulk.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            f_s = json.loads(fhandle.read())
        failed_b_structures.extend(f_s)
    fpath = os.path.join(Flame_dir,cycle_number,'minimahopping','failed_cluster.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            f_s = json.loads(fhandle.read())
        failed_c_structures.extend(f_s)

    perturbed_b_structures = []
    perturbed_c_structures = []

    for a_b_struct in failed_b_structures:
        a_b_structure = Structure.from_dict(a_b_struct)
        perturbed_b_structures.append(a_b_structure)
        for p in (0.02, 0.05):
            a_b_s = a_b_structure.copy()
            a_b_s.perturb(p)
            if is_structure_valid(a_b_s, min_d_prefactor, False, False):
                perturbed_b_structures.append(a_b_s)
        for s in (0.95, 1.05):
            a_b_s = a_b_structure.copy()
            a_b_s.scale_lattice(a_b_s.volume*s)
            if is_structure_valid(a_b_s, min_d_prefactor, False, False):
                perturbed_b_structures.append(a_b_s)

    for a_c_struct in failed_c_structures:
        spcs = Structure.from_dict(a_c_struct).species
        cart_coords = Structure.from_dict(a_c_struct).cart_coords
        maxx = max(cart_coords[:,0:1])[0]
        minx = min(cart_coords[:,0:1])[0]
        maxy = max(cart_coords[:,1:2])[0]
        miny = min(cart_coords[:,1:2])[0]
        maxz = max(cart_coords[:,2:3])[0]
        minz = min(cart_coords[:,2:3])[0]
        a_cluster = maxx-minx+inputs['vacuum_length']
        b_cluster = maxy-miny+inputs['vacuum_length']
        c_cluster = maxz-minz+inputs['vacuum_length']
        if max(a_cluster, b_cluster, c_cluster) > inputs['box_size']:
            continue
        molecule = Molecule(spcs, cart_coords)
        try:
            boxed_molecule = molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster)
        except:
            continue
        perturbed_c_structures.append(boxed_molecule)
        for p in (0.02, 0.05):
            a_c_structure = boxed_molecule.copy()
            a_c_structure.perturb(p)
            if is_structure_valid(a_c_structure, min_d_prefactor, False, False):
                perturbed_c_structures.append(a_c_structure)
    return perturbed_b_structures, perturbed_c_structures

def get_element_list():
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
    mpr= MPRester(configs['api_key'])
    known_structures = []
    primitive_known_structures = []
    vpas = []
    for a_composition in composition_list:
        docs = mpr.materials.search(formula=a_composition, fields=["structure"])
        known_structures.extend(docs)
    for a_k_s in known_structures:
        p_a_k_s = a_k_s.structure.get_primitive_structure()
        vpas.append(p_a_k_s.volume/len(p_a_k_s.sites))
        if len(p_a_k_s.sites) in inputs['bulk_number_of_atoms']+inputs['reference_number_of_atoms']:
            primitive_known_structures.append(p_a_k_s.as_dict())
    if len(vpas) < 2:
        covalent_radius = CovalentRadius.radius
        minmaxvpa = []
        elements = []
        nelement = []
        for elmnt, nelmnt in Composition(composition_list[0]).items():
            elements.append(str(elmnt))
            nelement.append(int(nelmnt))
        if len(vpas) == 1:
            minmaxvpa.append(vpas[0]*0.8)
        else:
            vol = 0
            for i in range(len(elements)):
                vol += 8 * covalent_radius[elements[i]]**3 * nelement[i]
            minmaxvpa.append(vol/sum(nelement))
        pre_fac = 2 if vpas[0] < 10 else 1.5
        minmaxvpa.append((minmaxvpa[0]/0.8) * pre_fac)
    else:
        minmaxvpa = [min(vpas), max(vpas)]
    return primitive_known_structures, minmaxvpa

def get_allowed_n_atom_for_compositions(composition_list):
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

    d_matrix = structure.distance_matrix
    for isites in range(len(structure.sites)):
        if not structure.get_neighbors(structure[isites], 5):
            return False
        if min_d_prefactor:
            for jsites in range(len(structure.sites)):
                min_d = get_min_d(structure[isites].species_string, structure[jsites].species_string) * min_d_prefactor
                if d_matrix[isites,jsites] > 0 and d_matrix[isites,jsites] < min_d:
                    return False
    return True

def get_min_d(elmnt1, elmnt2, X=False):
    covalent_radius = CovalentRadius.radius
    min_d = (covalent_radius[elmnt1] + covalent_radius[elmnt2])
    x_prefactor = 1
    if X:
        try:
            x_prefactor = 1 - (abs(Element(elmnt1).X - Element(elmnt2).X)/(3.98 - 0.79)) * 0.2
        except:
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
