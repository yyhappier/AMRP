"""Shared launcher for the checked-in, trusted experiment configurations.

Configuration values retain Bash parameter expansions so existing environment
overrides have the same meaning. Bash resolves values and argument arrays only;
the training process is launched by Python without shell=True.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def resolve(task, preset, overrides=None):
    document = json.loads((ROOT / 'examples' / task / 'configs.json').read_text())
    if preset not in document['presets']:
        raise ValueError(f'Unknown preset: {preset}. Use --list to see available experiments.')
    defaults = document['defaults']
    selected = document['presets'][preset]
    config = {**defaults, **selected}
    config['variables'] = {**defaults['variables'], **selected['variables']}
    env = dict(os.environ)
    env.update(overrides or {})
    env.update(AMRP_ROOT=str(ROOT), SCRIPT_DIR=str(ROOT / 'examples' / task),
               WORK_DIR=env.get('PROJECT_ROOT_DIR') or str(ROOT),
               TIMESTAMP=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    # Only checked-in configuration expressions are evaluated, never CLI values.
    lines = ['set -eu', 'REPO_ROOT="${REPO_ROOT:-$AMRP_ROOT}"']
    for name, value in config['variables'].items():
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', name):
            raise ValueError(f'Invalid configuration variable: {name}')
        lines.append(f'{name}={value}')
    lines += [f'OUTPUT_DIR={config["output"]}',
              f'AMRP_DEFAULT_LOG_DIR={config["log"]}',
              'LOG_DIR="${LOG_DIR:-$AMRP_DEFAULT_LOG_DIR}"',
              'LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.txt}"']
    conditions = {'MODEL_TYPE_ARGS': '[ -n "${MODEL_TYPE:-}" ]',
                  'VALUE_MODEL_TYPE_ARGS': '[ -n "${VALUE_MODEL_TYPE:-}" ]',
                  'RESUME_ARGS': '[ -n "${RESUME_CHECKPOINT_PATH:-}" ]',
                  'VLLM_ARGS': '[ "${USE_VLLM:-0}" = 1 ]'}
    for name, expression in config['optional_arguments'].items():
        lines += [f'{name}=()', f'if {conditions[name]}; then {name}=({expression}); fi']
    lines += ['export ' + ' '.join(config['exports'] + ['OUTPUT_DIR', 'LOG_DIR', 'LOG_FILE',
              'RUN_NAME', 'TASKNAME', 'MASTER_ADDR', 'NNODES', 'GPUS_PER_NODE', 'NODE_RANK']),
              'export PYTHONUNBUFFERED=1',
              'export PYTHONPATH="${AMRP_ROOT}:${PYTHONPATH:-}"',
              'set -- ' + config['arguments'],
              'printf "%s\\0" "$@"', 'printf "__AMRP_ENV__\\0"', 'env -0']
    result = subprocess.run(['bash', '-c', '\n'.join(lines)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError('Cannot resolve preset: ' + result.stderr.decode(errors='replace').strip())
    argv_bytes, env_bytes = result.stdout.split(b'__AMRP_ENV__\0', 1)
    argv = [s.decode() for s in argv_bytes.rstrip(b'\0').split(b'\0')]
    child_env = dict(item.decode().split('=', 1) for item in env_bytes.split(b'\0') if b'=' in item)
    interpreter = env.get('PYTHON_CMD') or sys.executable
    command = [interpreter, '-m', 'torch.distributed.run'] + argv
    return preset, config, command, child_env


def validate(config, env):
    nodes = int(env['NNODES'])
    gpus = int(env['GPUS_PER_NODE'])
    rank = int(env['NODE_RANK'])
    if nodes < 1 or gpus < 1 or not 0 <= rank < nodes:
        raise ValueError('Invalid NNODES, GPUS_PER_NODE, or NODE_RANK.')
    expected = config.get('required_world_size')
    if expected and nodes * gpus != expected:
        raise ValueError(f'This preset requires WORLD_SIZE={expected}; got {nodes * gpus}.')
    if nodes > 1 and env['MASTER_ADDR'] in ('127.0.0.1', 'localhost'):
        raise ValueError('Multi-node training requires an explicit reachable MASTER_ADDR.')


def main(task):
    parser = argparse.ArgumentParser(description=f'AMRP {task} training')
    parser.add_argument('--preset', default=os.environ.get('AMRP_PRESET'))
    parser.add_argument('--list', action='store_true', help='List presets without loading training dependencies')
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', help='Override a configuration environment variable')
    parser.add_argument('--dry-run', action='store_true', help='Print command and output paths; do not create files or load models')
    parser.add_argument('--json', action='store_true', help='Emit a machine-readable dry-run plan (requires --dry-run)')
    args = parser.parse_args()
    if args.list:
        for name in json.loads((ROOT / 'examples' / task / 'configs.json').read_text())['presets']:
            print(name)
        return
    if not args.preset:
        parser.error('--preset is required; use --list to see available experiments')
    if args.json and not args.dry_run:
        parser.error('--json requires --dry-run')
    overrides = {}
    for item in args.set:
        key, sep, value = item.partition('=')
        if not sep or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            parser.error('--set expects NAME=VALUE with an uppercase environment variable name')
        overrides[key] = value
    try:
        preset, config, command, env = resolve(task, args.preset, overrides)
        # Allow dry-run to display a multi-node preset before its host is configured.
        if not args.dry_run:
            validate(config, env)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps({'preset': preset, 'cwd': str(ROOT), 'command': command,
                          'output_dir': env['OUTPUT_DIR'], 'log_file': env['LOG_FILE'],
                          'environment': {key: env[key] for key in config['exports'] if key in env}}, indent=2))
        return
    print(f'[train] preset={preset}\n[train] output={env["OUTPUT_DIR"]}\n[train] log={env["LOG_FILE"]}')
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    # Fail before launching if the chosen destination is not writable. Unlike the
    # cluster-specific legacy wrappers, never silently redirect or rsync results.
    (ROOT / env['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
    log_path = ROOT / env['LOG_FILE']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as logfile:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        def forward(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)
        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                logfile.write(line)
                logfile.flush()
            status = process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    raise SystemExit(status if status >= 0 else 128 - status)
