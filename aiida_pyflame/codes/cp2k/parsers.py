import re
import numpy as np

def parse_cp2k_output_simple(fstring):
    """Parse CP2K output into a dictionary
    """
    lines = fstring.splitlines()
    SIRIUS = False
    if 'SIRIUS' in fstring:
        SIRIUS = True
    result_dict = {"exceeded_walltime": False}
    result_dict = {"SIRIUS": SIRIUS}
    energy = None
    bohr2ang = 0.529177208590000
    Eh2eV = 27.211324570273

    for i_line, line in enumerate(lines):
        if line.startswith(" GLOBAL| Run type"):
            result_dict["run_type"] = line.split()[-1]
        if line.startswith("lattice vectors"):
            lattice_vertor_A = [bohr2ang*float(lines[i_line+1].split()[2]),
                                bohr2ang*float(lines[i_line+1].split()[3]),
                                bohr2ang*float(lines[i_line+1].split()[4])]
            lattice_vertor_B = [bohr2ang*float(lines[i_line+2].split()[2]),
                                bohr2ang*float(lines[i_line+2].split()[3]),
                                bohr2ang*float(lines[i_line+2].split()[4])]
            lattice_vertor_C = [bohr2ang*float(lines[i_line+3].split()[2]),
                                bohr2ang*float(lines[i_line+3].split()[3]),
                                bohr2ang*float(lines[i_line+3].split()[4])]
            result_dict["lattice_vectors"] = np.array([lattice_vertor_A,
                                                       lattice_vertor_B,
                                                       lattice_vertor_C], np.float64)
        if "The number of warnings for this run is" in line:
            result_dict["nwarnings"] = int(line.split()[-1])

        if line.startswith(" ENERGY| "):
            energy = float(line.split()[8])
            result_dict["energy"] = energy*Eh2eV
            result_dict["energy_units"] = "eV"

        if "run_type" in result_dict.keys():
            # Initialization
            if "motion_step_info" not in result_dict:
                result_dict["motion_opt_converged"] = False
                result_dict["motion_step_info"] = {
                    "step": [],  # MOTION step
                    "energy_eV": [],  # total energy
                    "scf_converged": [],  # SCF converged in this motions step (bool)
                }
                step = 0
                energy = None
                if SIRIUS:
                    scf_converged = False
                else:
                    scf_converged = True
            print_now = False
            data = line.split()
            if re.search(r"SCF run NOT converged", line):
                scf_converged = False
            if SIRIUS and re.search(r"converged after", line):
                scf_converged = True
            if result_dict["run_type"] in ["ENERGY_FORCE"]:
                if energy is not None and not result_dict["motion_step_info"]["step"]:
                    print_now = True
                    if SIRIUS and "converged after" in fstring:
                        scf_converged = True
            if result_dict["run_type"] in ["GEO_OPT", "CELL_OPT"]:
                # Note: with CELL_OPT/LBFGS there is no "STEP 0", while there is with CELL_OPT/BFGS
                if re.search(r"Informations at step", line):
                    step = int(data[5])
                if (len(data) == 1 and data[0] == "---------------------------------------------------"):
                    print_now = True
                if re.search(
                    r"Reevaluating energy at the minimum", line):
                    result_dict["motion_opt_converged"] = True

            if print_now and energy is not None:
                if step == 0 and result_dict["run_type"] in ["GEO_OPT", "CELL_OPT"]: #BFGS or CS
                    continue
                result_dict["motion_step_info"]["step"].append(step)
                result_dict["motion_step_info"]["energy_eV"].append(energy*Eh2eV)
                result_dict["motion_step_info"]["scf_converged"].append(scf_converged)
                if SIRIUS:
                    scf_converged = False
                else:
                    scf_converged = True
    return result_dict

def parse_lines(lines, start, end):
    parsed_lines = []
    for line in lines[start:end]:
        parsed_lines.append(line.split()[-3:])
    return parsed_lines

def read_positions(content):
    start_line = []
    positions = []
    pattern = re.compile("^\s[i]", re.MULTILINE)
    lines = content.splitlines()
    for i_line, line in enumerate(lines):
        for match in re.finditer(pattern, line):
            start_line.append(i_line+1)
    nlines = start_line[1]-start_line[0]-2
    for a_s_l in start_line:
        parsed_lines = parse_lines(lines, a_s_l, a_s_l+nlines)
        positions.append(np.array(parsed_lines, np.float64))
    return positions

def read_coordinates(content):
    coordinates_str = []
    symbols = []
    lines = content.splitlines()[2:]
    for line in lines:
        coordinates_str.append(line.split()[-3:])
        symbols.append(line.split()[0])
    coordinates = [np.array(coordinates_str, np.float64)]
    return symbols, coordinates

def read_s_p_forces(content):
    lines = content.splitlines()
    for i_line, line in enumerate(lines):
        if line.startswith(" # Atom"):
            start_line = i_line + 1
        if line.startswith(" SUM"):
            end_line = i_line
    parsed_lines = parse_lines(lines, start_line, end_line)
    s_p_forces = [(np.array(parsed_lines, np.float64))]
    return s_p_forces

def read_forces(content):
    HaB2eVA = 51.42208619083232
    start_lines = []
    forces = []
    pattern = re.compile("^\s[i]", re.MULTILINE)
    lines = content.splitlines()
    for i_line, line in enumerate(lines):
        for match in re.finditer(pattern, line):
            start_lines.append(i_line+1)
    nlines = start_lines[1]-start_lines[0]-2
    for a_s_l in start_lines:
        parsed_lines = parse_lines(lines, a_s_l, a_s_l+nlines)
        forces.append(np.array(parsed_lines, np.float64) * HaB2eVA)
    return forces

def read_cell_parameters(content):
    cell_parameters = []
    lines = content.splitlines()[1:]
    for line in lines:
        cell_a = line.split()[2:5]
        cell_b = line.split()[5:8]
        cell_c = line.split()[8:11]
        cell_parameters.append(np.array([cell_a, cell_b, cell_c], np.float64))
    return cell_parameters
    
def read_lattice_parameters(content):
    bohr2ang = 0.529177208590000
    match = re.search(r"\n\s*&CELL\n(.*?)\n\s*&END CELL\n", content, re.DOTALL)
    cell_lines = [line.strip().split() for line in match.group(1).splitlines()]
    cell_str = [line[1:] for line in cell_lines if line[0] in "ABC"]
    cell = np.array(cell_str, np.float64) * bohr2ang
    return cell
