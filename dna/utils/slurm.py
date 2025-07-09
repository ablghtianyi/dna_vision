import os
import time

def get_slurm_id():
    # List of possible Slurm job ID environment variables
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')    
    id = None
    
    # Check each variable and return the first one that is set
    if slurm_job_id:
        id = f'{slurm_job_id}'
    
    if slurm_array_task_id:
        id = f'{id}_{slurm_array_task_id}'

    if id is None:
        id = time.strftime('%Y-%m-%d-%H-%M-%S')

    return id