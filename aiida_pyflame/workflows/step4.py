import os
import sys
from time import sleep
from aiida.orm import Group
from aiida.plugins import DataFactory
from aiida_pyflame.codes.flame.flame_launch_calculations import (
    AverDistSubmissionController,
    TrainSubmissionController,
    MinimaHoppingSubmissionController,
    DivCheckSubmissionController,
    QBCSubmissionController)
from aiida_pyflame.codes.utils import get_time, get_pertured_failed_structures, get_rejected_structures, store_calculation_nodes 
from aiida_pyflame.codes.flame.averdist import store_averdist_results
from aiida_pyflame.codes.flame.train import pre_train, select_a_train
from aiida_pyflame.codes.flame.minimahopping import (
    get_seeds,
    store_minhocao_results,
    store_minhopp_results)
from aiida_pyflame.codes.flame.qbc import store_ref_sigma, store_to_be_labeled_structures, get_to_be_labeled_structures, get_qbc_data
from aiida_pyflame.codes.flame.divcheck import (
    get_bulk_nats,
    get_cluster_nats,
    collect_divcheck_results,
    divcheck_report)
from aiida_pyflame.workflows.core import log_write, previous_run_exist_check, group_is_empty_check, report
import aiida_pyflame.workflows.settings as settings

StructureData = DataFactory('structure')
def step_4():
    """ Step 4
    """
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 4'+'\n')
    # check
    previous_run_exist_check()
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
            log_write('-----------------------------------------------'+'\n')
            log_write(f'cycle-{c_no}: train calculations'+'\n')
            log_write(f'start time: {get_time()}'+'\n')
            # check
            if len(settings.inputs['number_of_nodes']) < c_no:
                log_write('>>> Cannot proceed: parameters for cycle-{c_no} {cycle_name} are not provided <<<'+'\n')
                sys.exit()
            if os.path.exists(os.path.join(settings.Flame_dir,cycle_number,'train','training_data.json')):
                # and\
#               os.path.exists(os.path.join(settings.Flame_dir,cycle_number,'train','ann_input.yaml')):
                log_write(f'Found training data in {os.path.join(settings.Flame_dir,cycle_number,"train")}'+'\n')
            else:
                try:
                    os.mkdir(os.path.join(settings.Flame_dir,cycle_number,'train'))
                except FileExistsError:
                    log_write(f'>>> Cannot proceed: {os.path.join(settings.Flame_dir,cycle_number,"train")} exists <<<'+'\n')
                    sys.exit()
            pre_train(cycle_number)
            # submit jobs
            Group.collection.get(label='wf_train').clear()
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
            # report
            total_computing_time, submitted_jobs, finished_job = report('wf_train')
            log_write(f'submitted jobs: {submitted_jobs}, succesful jobs: {finished_job}'+'\n')
            log_write(f'total computing time: {round(total_computing_time, 2)} core-hours'+'\n')
            log_write(f'cycle-{c_no}: train calculations ended'+'\n')
            log_write(f'end time: {get_time()}'+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'train':
                store_calculation_nodes()
                log_write(f'End of the training loop after {c_no} cycles. Bye!'+'\n')
                sys.exit()
            else:
                cycle_name = 'MH'
######FLAME minimahopping######
        if cycle_name == 'MH':
            log_write('-----------------------------------------------'+'\n')
            log_write(f'cycle-{c_no}: minima hopping calculations'+'\n')
            log_write(f'start time: {get_time()}'+'\n')
            # check
            if len(settings.inputs['bulk_minhocao']) < c_no or\
                len(settings.inputs['minhocao_steps']) < c_no:
                log_write(f'>>> Cannot proceed: parameters for cycle-{c_no} {cycle_name} are not provided <<<'+'\n')
                sys.exit()
            if len(settings.inputs['bulk_minhopp']) < c_no or\
               len(settings.inputs['minhopp_steps']) < c_no:
                log_write(f'>>> Cannot proceed: parameters for cycle-{c_no} {cycle_name} are not provided <<<''\n')
                sys.exit()
            if settings.inputs['cluster_calculation'] and\
               len(settings.inputs['cluster_minhopp']) < c_no:
                log_write(f'>>> Cannot proceed: parameters for cycle-{c_no} {cycle_name} are not provided <<<'+'\n')
                sys.exit()
            # mkdir
            try:
                os.mkdir(os.path.join(settings.Flame_dir,cycle_number,'minimahopping'))
            except FileExistsError:
                log_write(f'>>> Cannot proceed: {os.path.join(settings.Flame_dir,cycle_number,"minimahopping")} exists <<<'+'\n')
                sys.exit()
            # add seeds to the parent group
            pg_minimahopping_group, _ = Group.collection.get_or_create('pg_minimahopping')
            pg_minimahopping_group.clear()
            minhopp_seeds_bulk, minhopp_seeds_cluster, minhocao_seeds = get_seeds(cycle_number)
            if minhopp_seeds_bulk:
                log_write(f'{len(minhopp_seeds_bulk)} bulk seeds for minima hopping'+'\n')
            else:
                log_write('>>> Warning: no nulk seeds for minima hopping <<<')
            for i, a_seed in enumerate(minhopp_seeds_bulk):
                seed_node = StructureData(pymatgen=a_seed).store()
                nat = len(a_seed.sites)
                seed_node.label = 'minhopp_bulk'
                seed_node.base.extras.set('job', 'minhopp_bulk-'+str(i)+'_'+str(nat)+'-atoms')
                pg_minimahopping_group.add_nodes(seed_node)
            if settings.inputs['cluster_calculation']:
                if minhopp_seeds_cluster:
                    log_write(f'{len(minhopp_seeds_cluster)} cluster seeds for minima hopping'+'\n')
                else:
                    log_write('>>> Warning: no cluster seeds for minima hopping <<<'+'\n')
                for i, a_seed in enumerate(minhopp_seeds_cluster):
                    seed_node = StructureData(pymatgen=a_seed).store()
                    nat = len(a_seed.sites)
                    seed_node.label = 'minhopp_cluster'
                    seed_node.base.extras.set('job', 'minhopp_cluster-'+str(i)+'_'+str(nat)+'-atoms')
                    pg_minimahopping_group.add_nodes(seed_node)
            if minhocao_seeds:
                log_write(f'{len(minhocao_seeds)} bulk seeds for minima hopping (variable cell)'+'\n')
            else:
                log_write('>>> Warning: no nulk seeds for minima hopping (variable cell) <<<'+'\n')
            for i, a_seed in enumerate(minhocao_seeds):
                nat = len(a_seed.sites)
                seed_node = StructureData(pymatgen=a_seed).store()
                seed_node.label = 'minhocao'
                seed_node.base.extras.set('job', 'minhocao-'+str(i)+'_'+str(nat)+'-atoms')
                pg_minimahopping_group.add_nodes(seed_node)
            # submit jobs
            Group.collection.get(label='wf_minimahopping').clear()
            controller = MinimaHoppingSubmissionController(
                parent_group_label='pg_minimahopping',
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
            total_computing_time, submitted_jobs, finished_job = report('wf_minimahopping')
            log_write(f'submitted jobs: {submitted_jobs}, succesful jobs: {finished_job}'+'\n')
            log_write(f'total computing time: {round(total_computing_time, 2)} core-hours'+'\n')
            log_write(f'cycle-{c_no}: minima hopping calculations ended'+'\n')
            log_write(f'end time: {get_time()}'+'\n')
            # clear groups
            for a_group_label in ['wf_divcheck', 'wf_qbc', 'wf_singlepoint']:
                a_group = Group.collection.get(label=a_group_label)
                a_group.clear()
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'MH':
                store_calculation_nodes()
                log_write(f'End of the training loop after {c_no} cycles. Bye!'+'\n')
                sys.exit()
            else:
                if 'DVC' in settings.inputs['selecting_method']:
                    cycle_name = 'FDC'
                else:
                    cycle_name = 'QBC'
######QBC######
        if cycle_name == 'QBC':
            log_write('-----------------------------------------------'+'\n')
            log_write(f'cycle-{c_no}: QBC'+'\n')
            log_write(f'start time: {get_time()}'+'\n')
            # check
            group_is_empty_check('wf_qbc')
            nats = {}
            if settings.inputs['cluster_calculation']:
                nats['cluster'] = get_cluster_nats(cycle_number)
            else:
                nats['cluster'] = []
            nats['bulk'] = get_bulk_nats(cycle_number)
            if len(nats['cluster']) == 0 and len(nats['bulk']) == 0:
                log_write('nothing to do'+'\n')
            get_qbc_data(cycle_number)
            # submit ref jobs
            controller = QBCSubmissionController(
                group_label='wf_qbc',
                max_concurrent=1,
                cycle_number=cycle_number,
                nats={},
                ref=True)
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            # store reference sigme
            store_ref_sigma(cycle_number)
            # clear group
            Group.collection.get(label='wf_qbc').clear()
            # submit jobs
            n_job = int(settings.job_script['QBC']['number_of_jobs']/3)
            if n_job == 0:
                n_job = 1
            controller = QBCSubmissionController(
                group_label='wf_qbc',
                max_concurrent=n_job,
                cycle_number=cycle_number,
                nats=nats,
                ref=False)
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            store_to_be_labeled_structures(cycle_number)
            total_computing_time, submitted_jobs, finished_job = report('wf_qbc')
            log_write(f'submitted jobs: {submitted_jobs}, succesful jobs: {finished_job}'+'\n')
            log_write(f'total computing time: {round(total_computing_time, 2)} core-hours'+'\n')
            log_write(f'cycle-{c_no}: QBC calculations ended'+'\n')
            log_write(f'end time: {get_time()}'+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'QBC':
                store_calculation_nodes()
                log_write(f'End of the training loop after {c_no} cycles. Bye!'+'\n')
                sys.exit()
            else:
                cycle_name = 'SP'
######FLAME divcheck######
        if cycle_name == 'FDC':
            log_write('-----------------------------------------------'+'\n')
            log_write(f'cycle-{c_no}: divcheck calculations'+'\n')
            log_write(f'start time: {get_time()}'+'\n')
            # check
            group_is_empty_check('wf_divcheck')
            if not settings.inputs['dtol_prefactor']:
                log_write(f'>>> Cannot proceed: parameters for cycle-{c_no} {cycle_name} are not provided <<<'+'\n')
                sys.exit()
            # average distance calculations
            if c_no == 1 and not os.path.exists(os.path.join(settings.Flame_dir,'aver_dist.json')):
                log_write('-----------------------------------------------'+'\n')
                log_write('Aaverage distance calculation'+'\n')
                log_write(f'start time: {get_time()}'+'\n')
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
                log_write('Aaverage distance calculation ended'+'\n')
                log_write(f'end time: {get_time()}'+'\n')

            nats = {}
            if settings.inputs['cluster_calculation']:
                nats['cluster'] = get_cluster_nats(cycle_number)
            else:
                nats['cluster'] = []
            nats['bulk'] = get_bulk_nats(cycle_number)
            if len(nats['cluster']) == 0 and len(nats['bulk']) == 0:
                log_write('nothing to do'+'\n')
            # submit jobs
            controller = DivCheckSubmissionController(
                group_label='wf_divcheck',
                max_concurrent=settings.job_script['QBC']['number_of_jobs'],
                cycle_number=cycle_number,
                nats=nats)
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            divcheck_report()
            log_write(f'cycle-{c_no}: divcheck calculations ended'+'\n')
            log_write(f'end time: {get_time()}'+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'FDC':
                store_calculation_nodes()
                log_write(f'End of the training loop after {c_no} cycles. Bye!'+'\n')
                sys.exit()
            else:
                cycle_name = 'SP'
######Single point calculations######
        if cycle_name == 'SP':
            log_write('-----------------------------------------------'+'\n')
            log_write(f'cycle-{c_no}: ab initio single point calculations'+'\n')
            log_write(f'start time: {get_time()}'+'\n')
            # check
            group_is_empty_check('wf_singlepoint')
            # clear groups
            pg_singlepoint_group, _ = Group.collection.get_or_create('pg_singlepoint')
            pg_singlepoint_group.clear()
            results_singlepoint_group, _ = Group.collection.get_or_create('results_singlepoint')
            results_singlepoint_group.clear()
            # add structurres to groups
            if 'FDV' in settings.inputs['selecting_method']:
                bulk_structures, cluster_structures = collect_divcheck_results()
            else:
                bulk_structures, cluster_structures = get_to_be_labeled_structures(cycle_number)
            failed_bulk_structures, failed_cluster_structures = get_pertured_failed_structures(cycle_number)
            rejected_bulk_structures, rejected_cluster_structures = get_rejected_structures(cycle_number)
            for i, a_pymatgen_structure in enumerate(bulk_structures+failed_bulk_structures+rejected_bulk_structures):
                nat = len(a_pymatgen_structure.sites)
                bulk_structure_node = StructureData(pymatgen=a_pymatgen_structure).store()
                bulk_structure_node.label = 'bulk'
                bulk_structure_node.base.extras.set('job', 'bulk-'+str(i)+'_'+str(nat)+'-atoms')
                pg_singlepoint_group.add_nodes(bulk_structure_node)
            if settings.inputs['cluster_calculation']:
                for i, a_pymatgen_structure in enumerate(cluster_structures+failed_cluster_structures+rejected_cluster_structures):
                    nat = len(a_pymatgen_structure.sites)
                    cluster_structure_node = StructureData(pymatgen=a_pymatgen_structure).store()
                    cluster_structure_node.label = 'cluster'
                    cluster_structure_node.base.extras.set('job', 'cluster-'+str(i)+'_'+str(nat)+'-atoms')
                    pg_singlepoint_group.add_nodes(cluster_structure_node)
            # run jobs
            if 'VASP' in settings.inputs['ab_initio_code']:
                from aiida_pyflame.codes.vasp.vasp_launch_calculations import VASPSPSubmissionController
                log_write(f'Ab-initio calculations with {settings.inputs["ab_initio_code"]}'+'\n')
                controller = VASPSPSubmissionController(
                    parent_group_label='pg_singlepoint',
                    group_label='wf_singlepoint',
                    max_concurrent=settings.job_script['geopt']['number_of_jobs'])
            elif 'SIRIUS' in settings.inputs['ab_initio_code'] or 'GTH' in settings.inputs['ab_initio_code']:
                from aiida_pyflame.codes.cp2k.cp2k_launch_calculations import CP2KSPSubmissionController
                log_write(f'Ab-initio calculations with {settings.inputs["ab_initio_code"]}'+'\n')
                controller = CP2KSPSubmissionController(
                    parent_group_label='pg_singlepoint',
                    group_label='wf_singlepoint',
                    max_concurrent=settings.job_script['geopt']['number_of_jobs'],
                    QSorSIRIUS=settings.inputs['ab_initio_code'])
            else:
                log_write('>>> ERROR: no ab_initio code is provided <<<'+'\n')
                sys.exit()
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            # report
            total_computing_time, submitted_jobs, finished_job = report('wf_singlepoint')
            log_write(f'submitted jobs: {submitted_jobs}, succesful jobs: {finished_job}'+'\n')
            log_write(f'total computing time: {round(total_computing_time, 2)} core-hours'+'\n')
            # end of FLAME loop?
            if settings.restart['training_loop_stop'][0] == c_no and settings.restart['training_loop_stop'][1] == 'SP':
                store_calculation_nodes()
                log_write(f'End of the training loop after {c_no} cycles. Bye!'+'\n')
                sys.exit()
            else:
                cycle_name = 'train'
