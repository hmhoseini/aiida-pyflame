import sys
from datetime import datetime
from time import sleep
from aiida.orm import Group
from aiida_pyflame.workflows.core import log_write
from aiida_pyflame.workflows.settings import groups
from aiida_pyflame.workflows.step3 import store_step3_results
from aiida_pyflame.workflows.step4 import store_seeds
from aiida_pyflame.codes.flame.minimahopping import store_minhocao_results, store_minhopp_results
import aiida_pyflame.workflows.settings as settings

def find_active_group():
    active_groups = []
    for a_group_label in groups['workflows_group_list']:
        try:
            a_group = Group.collection.get(label=a_group_label)
        except:
            continue
        for a_node in a_group.nodes:
            if not a_node.is_terminated:
                active_groups.append(a_group)
                break
    return active_groups

def rerun():
    active_groups = find_active_group()
    if len(active_groups) == 0:
        log_write('>>> nothing to do <<<'+'\n')
        sys.exit()
    if len(active_groups) > 1:
        log_write('>>> ERROR: more than one group is active (pk: {}) <<<'.format(active_groups)+'\n')
        sys.exit()
    active_group = active_groups[0]

    if 'step3' in active_group.label:
        log_write('resuming step-3 calculations')
        if 'SIRIUS' in settings.inputs['ab_initio_code'] or 'GTH' in settings.inputs['ab_initio_code']:
            from aiida_pyflame.codes.cp2k.cp2k_launch_calculations import CP2KSubmissionController
            log_write('Ab-initio calculations with {}'.format(settings.inputs['ab_initio_code'])+'\n')
            controller = CP2KSubmissionController(
                parent_group_label='structures_step3',
                group_label='wf_step3',
                max_concurrent=settings.job_script['geopt']['number_of_jobs'],
                GTHorSIRIUS=settings.inputs['ab_initio_code'])
        elif settings.inputs['ab_initio_code']=='VASP':
            from aiida_pyflame.codes.vasp.vasp_launch_calculations import VASPSubmissionController
            log_write('Ab-initio calculations with VASP'+'\n')
            controller = VASPSubmissionController(
                parent_group_label='structures_step3',
                group_label='wf_step3',
                max_concurrent=settings.job_script['geopt']['number_of_jobs'])
        else:
            log_write('>>> ERROR: no ab_initio code is provided <<<'+'\n')
            sys.exit()
        # wait until all jobs are done
        while controller.num_to_run > 0 or controller.num_active_slots > 0:
            if controller.num_to_run > 0:
                controller.submit_new_batch(dry_run=False)
            sleep(60)
        # store
        store_step3_results()
    elif 'minimahopping' in active_group.label:
        from aiida_pyflame.codes.flame.flame_launch_calculations import MinimaHoppingSubmissionController
        c_no = settings.restart['training_loop_start'][0]
        cycle_number = 'cycle-'+str(c_no)
        log_write('resuming minima hopping calculations for cycle {}'.format(c_no)+'\n')
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
    elif 'singlepoint' in active_group.label:
        c_no = settings.restart['training_loop_start'][0]
        cycle_number = 'cycle-'+str(c_no)
        log_write('resuming single point calculations for cycle {}'.format(c_no)+'\n')
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
    else:
        log_write('>>> cannot rerun {} <<<'.format(active_group.label)+'\n')
