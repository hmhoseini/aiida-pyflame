import os
import sys
import json
import math
from time import sleep
from datetime import datetime
from aiida.orm import Group
from aiida.plugins import DataFactory
from aiida_pyflame.codes.flame.flame_launch_calculations import (
    AverDistSubmissionController,
    TrainSubmissionController,
    MinimaHoppingSubmissionController,
    DivCheckSubmissionController)
from aiida_pyflame.codes.utils import get_pertured_failed_structures
from aiida_pyflame.codes.flame.averdist import store_averdist_results
from aiida_pyflame.codes.flame.train import pre_train, select_a_train
from aiida_pyflame.codes.flame.minimahopping import (
    get_minhocao_seeds,
    store_minhocao_results,
    get_minhopp_seeds,
    store_minhopp_results)
from aiida_pyflame.codes.flame.divcheck import (
    get_bulk_nats,
    get_cluster_nats,
    collect_divcheck_results,
    divcheck_report)
from aiida_pyflame.workflows.core import log_write, previous_run_exist_check, group_is_empty_check
import aiida_pyflame.workflows.settings as settings

def collect_singlepoint_results():
    nextstep_seeds_bulk = []
    nextstep_seeds_cluster = []
    with open(os.path.join(settings.output_dir,'min_epa.dat'), 'r', encoding='utf8') as fhandle:
        min_epa = float(fhandle.readline().strip())
    e_window = settings.inputs['energy_window']
    results_singlepoint_group = Group.get(label='results_singlepoint')
    for a_node in results_singlepoint_group.nodes:
        if 'VASP' in settings.inputs['ab_initio_code']:
            if not a_node.outputs.misc.dict.run_status['electronic_converged']:
                continue
            total_energy = float(a_node.outputs.energies.get_array('energy_extrapolated_electronic')[-1])
            pymatgen_structure = a_node.outputs.structure.get_pymatgen()
            nat = len(pymatgen_structure.sites)
            epa = total_energy/nat
            if epa < min_epa + e_window:
                forces = a_node.outputs.trajectory.get_array('forces')[-1].tolist()
                tot_forces = []
                for a_f in range(len(forces)):
                    tot_forces.append(math.sqrt(forces[a_f][0]**2 + forces[a_f][1]**2 + forces[a_f][2]**2))
                max_tot_foce = max(tot_forces)
                if 'bulk' in a_node.label:
                    if epa < min_epa:
                        min_epa = epa
                        with open(os.path.join(settings.output_dir,'min_epa.dat'), 'w', encoding='utf8') as fhandle:
                            fhandle.write(str(min_epa))
                    if max_tot_foce < 0.71:
                        nextstep_seeds_bulk.append(pymatgen_structure.as_dict())
                if 'cluster' in a_node.label:
                    if max_tot_foce < 1.01:
                        nextstep_seeds_cluster.append(pymatgen_structure.as_dict())

        if 'SIRIUS' in settings.inputs['ab_initio_code'] or 'GTH' in settings.inputs['ab_initio_code']:
            if not a_node.outputs.output_parameters.dict['motion_step_info']['scf_converged'][-1]:
                continue
            total_energy = float(a_node.outputs.output_parameters.dict['motion_step_info']['energy_eV'][-1])
            pymatgen_structure = a_node.outputs.output_structure.get_pymatgen()
            nat = len(pymatgen_structure.sites)
            epa = total_energy/nat
            if epa < min_epa + e_window:
                forces = a_node.outputs.output_parameters.dict['motion_step_info']['forces'][-1]
                tot_forces = []
                for a_f in range(len(forces)):
                    tot_forces.append(math.sqrt(forces[a_f][0]**2 + forces[a_f][1]**2 + forces[a_f][2]**2))
                max_tot_foce = max(tot_forces)
                if 'bulk' in a_node.label:
                    if epa < min_epa:
                        min_epa = epa
                        with open(os.path.join(settings.output_dir,'min_epa.dat'), 'w', encoding='utf8') as fhandle:
                            fhandle.write(str(min_epa))
                    if max_tot_foce < 0.71:
                        nextstep_seeds_bulk.append(pymatgen_structure.as_dict())
                if 'cluster' in a_node.label:
                    if max_tot_foce < 1.01:
                        nextstep_seeds_cluster.append(pymatgen_structure.as_dict())
    return nextstep_seeds_bulk, nextstep_seeds_cluster

def store_seeds(cycle_number):
    nextstep_seeds_bulk, nextstep_seeds_cluster = collect_singlepoint_results()
    if len(nextstep_seeds_bulk) > 0:
        with open(os.path.join(settings.Flame_dir,cycle_number,'minimahopping','nextstep_seeds_bulk.json'), 'w', encoding='utf8') as fhandle:
            json.dump(nextstep_seeds_bulk, fhandle)
        log_write('Number of bulk seeds for the next step: {}'.format(len(nextstep_seeds_bulk))+'\n')
    if len(nextstep_seeds_cluster) > 0:
        with open(os.path.join(settings.Flame_dir,cycle_number,'minimahopping','nextstep_seeds_cluster.json'), 'w', encoding='utf8') as fhandle:
            json.dump(nextstep_seeds_cluster, fhandle)
        log_write('Number of cluster seeds for the next step: {}'.format(len(nextstep_seeds_cluster))+'\n')

def step_4():
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 4'+'\n')
    log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    # check
    previous_run_exist_check()
    # average distance calculations
    if not os.path.exists(os.path.join(settings.Flame_dir,'aver_dist.json')):
        log_write("-----------------------------------------------"+'\n')
        log_write('Aaverage distance calculation'+'\n')
        log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
        # check
        group_is_empty_check('wf_averdist')
        # submit jobs
        controller = AverDistSubmissionController(
            group_label='wf_averdist',
            max_concurrent=len(settings.inputs['bulk_number_of_atoms']),
            nats=settings.inputs['bulk_number_of_atoms'])
        # wait until all jobs are done
        while controller.num_to_run > 0 or controller.num_active_slots > 0:
            if controller.num_to_run > 0:
                controller.submit_new_batch(dry_run=False)
            sleep(60)
        # store results
        store_averdist_results()
        # clear group
#        wf_averdist_group = Group.get(label='wf_averdist')
#        wf_averdist_group.clear()
        log_write('Aaverage distance calculation ended'+'\n')
        log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
####################TRAINING LOOP####################
    cycle_name = settings.restart['training_loop_start'][1]
    for c_no in range(settings.restart['training_loop_start'][0], settings.restart['training_loop_stop'][0]+1):
        cycle_number = 'cycle-'+str(c_no)
        # mkdir
        try:
            os.mkdir(os.path.join(settings.Flame_dir,cycle_number))
        except FileExistsError:
            pass
######FLAME train######
        if cycle_name == 'train':
            log_write("-----------------------------------------------"+'\n')
            log_write('cycle-{}: train calculations'.format(c_no)+'\n')
            log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # check
            group_is_empty_check('wf_train')
            if len(settings.inputs['number_of_nodes']) < c_no:
                log_write('>>> Cannot proceed: parameters for cycle-{} {} are not provided <<<'.format(c_no, cycle_name)+'\n')
                sys.exit()
            # mkdir
            try:
                os.mkdir(os.path.join(settings.Flame_dir,cycle_number,'train'))
            except FileExistsError:
                log_write('>>> Cannot proceed: {} exists <<<'.format(os.path.join(settings.Flame_dir,cycle_number,'train'))+'\n')
                sys.exit()
            # pre train
            pre_train(cycle_number)
            # submit jobs
            controller = TrainSubmissionController(
                group_label='wf_train',
                max_concurrent=settings.job_script['train']['number_of_jobs'],
                cycle_number=cycle_number)
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            select_a_train(cycle_number)
            log_write('cycle-{}: train calculations ended'.format(c_no)+'\n')
            log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'train':
                log_write('End of the training loop after {} cycles. Bye!'.format(c_no)+'\n')
                sys.exit()
            else:
                cycle_name = 'minimahopping'
######FLAME minimahopping######
        if cycle_name == 'minimahopping':
            log_write("-----------------------------------------------"+'\n')
            log_write('cycle-{}: minima hopping calculations'.format(c_no)+'\n')
            log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # check
            group_is_empty_check('wf_minimahopping')
            if len(settings.inputs['bulk_minhocao']) < c_no or\
                len(settings.inputs['minhocao_steps']) < c_no:
                log_write('>>> Cannot proceed: parameters for cycle-{} {} are not provided <<<'.format(c_no, cycle_name)+'\n')
                sys.exit()
            if len(settings.inputs['bulk_minhopp']) < c_no or\
               len(settings.inputs['minhopp_steps']) < c_no:
                log_write('>>> Cannot proceed: parameters for cycle-{} {} are not provided <<<'.format(c_no, cycle_name)+'\n')
                sys.exit()
            if settings.inputs['cluster_calculation'] and\
               len(settings.inputs['cluster_minhopp']) < c_no:
                log_write('>>> Cannot proceed: parameters for cycle-{} {} are not provided <<<'.format(c_no, cycle_name)+'\n')
                sys.exit()
            # mkdir
            try:
                os.mkdir(os.path.join(settings.Flame_dir,cycle_number,'minimahopping'))
            except FileExistsError:
                log_write('>>> Cannot proceed: {} exists <<<'.format(os.path.join(settings.Flame_dir,cycle_number,'minimahopping'))+'\n')
                sys.exit()
            # add seeds to the parent group
            StructureData = DataFactory('structure')
            structures_minimahopping_group, _ = Group.objects.get_or_create('structures_minimahopping')
            structures_minimahopping_group.clear()
            minhopp_seeds_bulk, minhopp_seeds_cluster = get_minhopp_seeds(cycle_number)
            if settings.inputs['cluster_calculation']:
                for i, a_seed in enumerate(minhopp_seeds_cluster):
                    seed_node = StructureData(pymatgen=a_seed).store()
                    nat = len(a_seed.sites)
                    seed_node.label = 'minhopp_cluster'
                    seed_node.base.extras.set('job', 'minhopp_cluster-'+str(i)+'_'+str(nat)+'-atoms')
                    structures_minimahopping_group.add_nodes(seed_node)
            for i, a_seed in enumerate(minhopp_seeds_bulk):
                seed_node = StructureData(pymatgen=a_seed).store()
                nat = len(a_seed.sites)
                seed_node.label = 'minhopp_bulk'
                seed_node.base.extras.set('job', 'minhopp_bulk-'+str(i)+'_'+str(nat)+'-atoms')
                structures_minimahopping_group.add_nodes(seed_node)
            minhocao_seeds = get_minhocao_seeds(cycle_number)
            for i, a_seed in enumerate(minhocao_seeds):
                seed_node = StructureData(pymatgen=a_seed).store()
                seed_node.label = 'minhocao'
                seed_node.base.extras.set('job', 'minhocao-'+str(i)+'_'+str(nat)+'-atoms')
                structures_minimahopping_group.add_nodes(seed_node)
            # submit jobs
            controller = MinimaHoppingSubmissionController(
                parent_group_label='structures_minimahopping',
                group_label='wf_minimahopping',
                max_concurrent=settings.job_script['minimahopping']['number_of_jobs'],
                cycle_number=cycle_number)
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            store_minhocao_results(cycle_number)
            store_minhopp_results(cycle_number)
            log_write('cycle-{}: minima hopping calculations ended'.format(c_no)+'\n')
            log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'minimahopping':
                log_write('End of the training loop after {} cycles. Bye!'.format(c_no)+'\n')
                sys.exit()
            else:
                cycle_name = 'divcheck'
######FLAME divcheck######
        if cycle_name == 'divcheck':
            log_write("-----------------------------------------------"+'\n')
            log_write('cycle-{}: divcheck calculations'.format(c_no)+'\n')
            log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # check
            group_is_empty_check('wf_divcheck')
            if not settings.inputs['dtol_prefactor']:
                log_write('>>> Cannot proceed: parameters for cycle-{} {} are not provided <<<'.format(c_no, cycle_name)+'\n')
                sys.exit()
            #
            nats = {}
            if settings.inputs['cluster_calculation']:
                nats['cluster'] = get_cluster_nats(cycle_number)
            else:
                nats['cluster'] = []
            nats['bulk'] = get_bulk_nats(cycle_number)
            # submit jobs
            controller = DivCheckSubmissionController(
                group_label='wf_divcheck',
                max_concurrent=settings.job_script['divcheck']['number_of_jobs'],
                cycle_number=cycle_number,
                nats=nats)
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            divcheck_report()
            log_write('cycle-{}: divcheck calculations ended'.format(c_no)+'\n')
            log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'divcheck':
                log_write('End of the training loop after {} cycles. Bye!'.format(c_no)+'\n')
                sys.exit()
            else:
                cycle_name = 'SP_calculation'
######Single point calculations######
        if cycle_name == 'SP_calculation':
            log_write("-----------------------------------------------"+'\n')
            log_write("cycle-{}: ab initio single point calculations".format(c_no)+'\n')
            log_write('start time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # check
            group_is_empty_check('wf_singlepoint')
            # clear groups
            structures_singlepoint_group, _ = Group.objects.get_or_create('structures_singlepoint')
            structures_singlepoint_group.clear()
            results_singlepoint_group, _ = Group.objects.get_or_create('results_singlepoint')
            results_singlepoint_group.clear()
            # add structurres to groups
            StructureData = DataFactory('structure')
            bulk_structures, cluster_structures = collect_divcheck_results()
            failed_bulk_structures, failed_cluster_structures = get_pertured_failed_structures(cycle_number)
            for i, a_pymatgen_structure in enumerate(bulk_structures+failed_bulk_structures):
                nat = len(a_pymatgen_structure.sites)
                bulk_structure_node = StructureData(pymatgen=a_pymatgen_structure).store()
                bulk_structure_node.label = 'bulk'
                bulk_structure_node.base.extras.set('job', 'bulk-'+str(i)+'_'+str(nat)+'-atoms')
                structures_singlepoint_group.add_nodes(bulk_structure_node)
            if settings.inputs['cluster_calculation']:
                for i, a_pymatgen_structure in enumerate(cluster_structures+failed_cluster_structures):
                    nat = len(a_pymatgen_structure.sites)
                    cluster_structure_node = StructureData(pymatgen=a_pymatgen_structure).store()
                    cluster_structure_node.label = 'cluster'
                    cluster_structure_node.base.extras.set('job', 'cluster-'+str(i)+'_'+str(nat)+'-atoms')
                    structures_singlepoint_group.add_nodes(cluster_structure_node)
            # run jobs
            if 'VASP' in settings.inputs['ab_initio_code']:
                from aiida_pyflame.codes.vasp.vasp_launch_calculations import VASPSPSubmissionController
                log_write('Ab-initio calculations with {}'.format(settings.inputs['ab_initio_code'])+'\n')
                controller = VASPSPSubmissionController(
                    parent_group_label='structures_singlepoint',
                    group_label='wf_singlepoint',
                    max_concurrent=settings.job_script['geopt']['number_of_jobs'])
            elif 'SIRIUS' in settings.inputs['ab_initio_code'] or 'GTH' in settings.inputs['ab_initio_code']:
                from aiida_pyflame.codes.cp2k.cp2k_launch_calculations import CP2KSPSubmissionController
                log_write('Ab-initio calculations with {}'.format(settings.inputs['ab_initio_code'])+'\n')
                controller = CP2KSPSubmissionController(
                    parent_group_label='structures_singlepoint',
                    group_label='wf_singlepoint',
                    max_concurrent=settings.job_script['geopt']['number_of_jobs'],
                    GTHorSIRIUS=settings.inputs['ab_initio_code'])
            else:
                log_write('>>> ERROR: no ab_initio code is provided <<<'+'\n')
                sys.exit()
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            # store new seeds
            store_seeds(cycle_number)
            log_write('cycle-{}: ab initio single point calculations ended'.format(c_no)+'\n')
            log_write('end time: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'SP_calculation':
                log_write('End of the training loop after {} cycles. Bye!'.format(c_no)+'\n')
                sys.exit()
            else:
                # clear groups
                for a_group_label in ['wf_train', 'wf_minimahopping', 'wf_divcheck', 'wf_singlepoint']:
                    a_group = Group.get(label=a_group_label)
                    a_group.clear()
                cycle_name = 'train'
