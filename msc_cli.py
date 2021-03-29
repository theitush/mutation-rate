import argparse
import os
import pandas as pd
from subprocess import call
from utils import create_pbs_cmd_file, submit_cmdfile_to_pbs

MODELING_REPO = "/sternadi/home/volume2/ita/wf_modeling/"
PBS_CMD_PATH = "/opt/pbs/bin/qsub"
QUEUE = 'adistzachi'
PYTHON_PATH = "/a/home/cc/lifesci/ita/.conda/envs/envita/bin/python3"


def msc_cli(input_path, output_dir):
    df = pd.read_table(input_path)
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    cmd_path = os.path.join(logs_dir, 'msc_cli.cmd')
    process.run()
    cmd = "PYTHONPATH={PYTHON_PATH}\n"
    cmd += f"{PYTHON_PATH} {MODELING_REPO}/modeling.py -i {input_path} -o {output_dir} " + "-l ${PBS_ARRAY_INDEX}"
    jnums = (0, len(df)-1)
    create_pbs_cmd_file(path=cmd_path, alias='msc_cli', output_logs_dir=logs_dir, jnums=jnums, cmd=cmd, queue=QUEUE)
    job_id = submit_cmdfile_to_pbs(cmd_path, pbs_cmd_path=PBS_CMD_PATH)
    if job_id:
        print(f"Submitted jod: {job_id}")
        print(f"Output files will be in {output_dir}")
    else:
        print(f"Could not submit job to queue!")
    print(f"cmd file and logs are in {logs_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="path to data file", required=True)
    parser.add_argument("-o", "--output_dir", type=str, help="directory for output files", required=True)
    args = parser.parse_args()
    msc_cli(input_path=args.input_path, output_dir=args.output_dir)
