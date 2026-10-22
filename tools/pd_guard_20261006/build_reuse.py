#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Rebuild frozen PD translation units against SHA-checked existing archives.

This is intentionally an incremental build, not a clean CMake build. All output
is private to --output. It never overwrites the dependency cache or live binary.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source(relative: str, role: str) -> Path:
    overlay = HERE / 'decode_overlay' / relative
    return overlay if role == 'decode' and overlay.is_file() else ROOT / relative


def _compile(record: dict, args: argparse.Namespace, manifest: dict) -> dict:
    source = _source(record['source'], args.role)
    if _sha(source) != record['source_sha256']:
        raise ValueError(f'Frozen source mismatch: {source}')
    target = args.output / record['member']
    command = []
    original = record['command']
    index = 0
    while index < len(original):
        value = original[index]
        if value in ('-o', '-MT', '-MF', '-c'):
            replacement = str(source) if value == '-c' else str(target)
            if value == '-MF':
                replacement += '.d'
            command.extend([value, replacement])
            index += 2
            continue
        if value.startswith('-I/tmp/'):
            index += 1
            continue
        old = manifest['old_source_root']
        if value.startswith('-I' + old + '/xllm'):
            value = value.replace(old, str(ROOT), 1)
        elif value == '-I' + old:
            value = '-I' + str(ROOT)
        command.append(value)
        index += 1
    includes = []
    if args.role == 'decode':
        includes += ['-I' + str(HERE / 'decode_overlay/xllm')]
    includes += ['-I' + str(ROOT), '-I' + str(ROOT / 'xllm'), '-I' + str(ROOT / 'xllm/core'),
                 '-I' + str(ROOT / 'xllm/core/layers/musa/deepseek_v4')]
    compiler = next(i for i, item in enumerate(command) if item.endswith(('mcc_wrapper', '/mcc', '/g++', '/c++')))
    command[compiler + 1:compiler + 1] = includes
    (args.output / (record['member'] + '.command')).write_text(shlex.join(command) + '\n')
    with (args.output / (record['member'] + '.log')).open('x') as log:
        subprocess.run(command, cwd=manifest['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
    logger.info('Compiled %s', record['source'])
    return {'source': str(source), 'source_sha256': _sha(source),
            'member': record['member'], 'archive': record['archive'],
            'object_sha256': _sha(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['prefill', 'decode'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--relink-only', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 8:
        parser.error('--jobs must be between 1 and 8')
    args.output = args.output.resolve()
    manifest = json.loads((HERE / 'pd_release_build_recipes_20261006.json').read_text())
    role = manifest['roles'][args.role]
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, expected in role['link_inputs_sha256'].items():
        if _sha(Path(filename)) != expected:
            raise ValueError(f'Link input changed: {filename}')
    logger.info('Verified %d link inputs', len(role['link_inputs_sha256']))
    results = []
    if not args.relink_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_compile, record, args, manifest) for record in role['compile']]
            results = [future.result() for future in futures]
    replacements = {}
    for archive in sorted({record['archive'] for record in results}):
        inputs = [Path(name) for name in role['link_inputs_sha256'] if Path(name).name == archive]
        if len(inputs) != 1:
            raise ValueError(f'Ambiguous archive: {archive}: {inputs}')
        original = inputs[0]
        target = args.output / archive
        shutil.copy2(original, target)
        members = subprocess.check_output(['ar', 't', str(original)], text=True).splitlines()
        objects = [str(args.output / record['member']) for record in results if record['archive'] == archive]
        for obj in objects:
            if Path(obj).name not in members:
                raise ValueError(f'Object is not an original archive member: {obj}')
        subprocess.run(['ar', 'r', str(target), *objects], check=True)
        replacements[str(original)] = str(target)
    command = []
    for value in role['link']:
        for old, new in replacements.items():
            value = value.replace(old, new)
        if '-Wl,-Map,' in value:
            value = '-Wl,-Map,' + str(args.output / 'link.map')
        command.append(value)
    command[command.index('-o') + 1] = str(args.output / 'xllm')
    (args.output / 'link.command').write_text(shlex.join(command) + '\n')
    with (args.output / 'link.log').open('x') as log:
        subprocess.run(command, cwd=manifest['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {'role': args.role, 'incremental_not_clean_build': True,
              'relink_only': args.relink_only, 'rebuilt': results,
              'recipes_sha256': _sha(HERE / 'pd_release_build_recipes_20261006.json'),
              'binary_sha256': _sha(args.output / 'xllm')}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    logger.info('Built %s sha256=%s', args.output / 'xllm', result['binary_sha256'])


if __name__ == '__main__':
    main()
