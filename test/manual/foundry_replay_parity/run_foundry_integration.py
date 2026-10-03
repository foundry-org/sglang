"""Real SGLang Foundry SAVE/LOAD probe; all GPU work is bounded by the launcher.

Dependencies: latest SGLang/Foundry checkouts, including the opt-in
Foundry research bridge. Retained standalone helper code is colocated. This harness never replaces Foundry's archive loader or manifest.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from integration_helpers import (write_json, gpu_state, generation_ids, prefill_defaults,
                        required_backend_options, resolved_prefill_snapshot)
from integration_helpers import paired_ratios


def manifest_groups(archive):
    paths = sorted(Path(archive).rglob('graph_manifest.json'))
    if not paths:
        raise RuntimeError(f'No actual SAVE graph_manifest.json in {archive}')
    result = []
    for path in paths:
        groups = []
        for group in json.loads(path.read_text())['topology_groups']:
            if group.get('partition', 'decode') != 'decode':
                continue
            members = []
            for name in group['members']:
                match = re.search(r'_FULL_t(\d+)_', name)
                if match is None:
                    raise ValueError(f'Unsupported actual archive graph filename {name}')
                members.append(int(match[1]))
            groups.append({'template': group['template'], 'members': members,
                           'topology_key': group['topology_key']})
        result.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'groups': groups})
    signatures = [sorted(sorted(g['members']) for g in row['groups']) for row in result]
    if any(sig != signatures[0] for sig in signatures[1:]):
        raise RuntimeError(f'Ranks have different SAVE groups: {signatures}')
    return result


def complete_pair_sequence(groups):
    """Walk every directed pair per actual manifest group, with returns."""
    sequence = []
    for group in groups:
        members = group['members']
        if len(members) == 1:
            sequence.append(members[0])
            continue
        # A compact complete digraph Euler walk, rooted at the saved template.
        remaining = {u: [v for v in reversed(members) if v != u] for u in members}
        stack, path = [members[0]], []
        while stack:
            if remaining[stack[-1]]:
                stack.append(remaining[stack[-1]].pop())
            else:
                path.append(stack.pop())
        sequence.extend(reversed(path))
    return sequence


def install_worker_probe(rank):
    import torch
    from foundry._qmd_research.graph_probe import DriverGraph
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    cfg = json.loads(Path(os.environ['FOUNDRY_INTEGRATION_PROBE_CONFIG']).read_text())
    out = Path(cfg['output']) / f'rank_{rank}'
    out.mkdir(parents=True, exist_ok=True)
    original = FullCudaGraphBackend.replay
    seen = set()
    driver = None
    dp_replay_index = 0
    backend_observer = None
    if cfg['mode'] == 'save':
        from foundry_backend_capture import install
        backend_observer = install(cfg['target_backend'])
    def replay(backend, key, batch, **kwargs):
        nonlocal driver, dp_replay_index
        runner = backend._cuda_graph_runner
        try:
            try:
                phase = json.loads(Path(cfg['phase_file']).read_text())
                phase_error = None
            except Exception as exc:
                phase, phase_error = {'armed': False}, repr(exc)
            group = backend._tp_group.cpu_group
            world = torch.distributed.get_world_size(group)
            if cfg['target_backend'] == 'deepep_dp':
                from dp_probe import make_offer, gather_and_decide
                from sglang.srt.runtime_context import get_parallel
                parallel = get_parallel()
                if phase_error:
                    phase = {'read_error': phase_error}
                offer = make_offer(rank=rank, world_size=world, dp_rank=parallel.attn_dp_rank,
                    dp_size=parallel.attn_dp_size, attn_tp_size=parallel.attn_tp_size,
                    replay_index=dp_replay_index, phase=phase, seen_phase_ids=seen,
                    forward_mode=batch.forward_mode.name, raw_batch=batch.batch_size,
                    runner_raw_batch=runner.raw_bs, padded_batch=runner.bs,
                    capture_batch=key.size, global_num_tokens=batch.original_global_num_tokens_cpu,
                    require_mlp_tp_gather=runner.require_mlp_tp_gather,
                    can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
                    runner_name=type(runner).__name__)
                def gather(value):
                    values = [None] * world
                    torch.distributed.all_gather_object(values, value, group=group)
                    return values
                decision, offers = gather_and_decide(offer, gather)
            else:
                from tp_probe import decide
                offer = {'rank': rank, 'world_size': world, 'replay_index': dp_replay_index,
                         'phase': phase, 'seen': phase.get('id') in seen,
                         'runner_name': type(runner).__name__,
                         'forward_mode': batch.forward_mode.name,
                         'raw_batch': int(batch.batch_size), 'capture_batch': int(key.size),
                         'error': phase_error}
                offers = [None] * world
                torch.distributed.all_gather_object(offers, offer, group=group)
                decision = decide(offers)
            dp_replay_index += 1
            dp_admission = {'decision': decision, 'offers': offers}
            with (out/'admission.jsonl').open('a') as stream:
                stream.write(json.dumps(dp_admission)+'\n')
            if decision['action'] == 'reject':
                raise RuntimeError(f'Common probe admission rejected: {decision}')
            if decision['action'] == 'skip':
                return original(backend, key, batch, **kwargs)
            phase = offers[0]['phase']
            graph = backend._graphs[key]
            state_before = None
            if cfg['mode'] == 'load':
                from foundry import research_qmd
                state_before = research_qmd.state(graph)
            diagnostic_inputs = None
            if cfg.get('diagnostic_save_mismatch'):
                from logit_diagnostics import live_inputs
                diagnostic_inputs = live_inputs(batch, runner, int(key.size), torch)
            # At this boundary the runner already populated the current inputs.
            # Poison before the first actual backend launch selected for a phase.
            initial_logits = backend._outputs[key].next_token_logits
            if initial_logits is None or initial_logits.ndim != 2:
                raise RuntimeError('Full output logits missing before original replay')
            initial_logits.fill_(float('nan'))
            # The actual adapter/Foundry replay performs member rewrite/update.
            result = original(backend, key, batch, **kwargs)
            seen.add(phase['id'])
            graph = backend._graphs[key]
            if not type(graph).__module__.startswith('foundry'):
                raise RuntimeError(f'Actual backend graph is not Foundry: {type(graph)}')
            logits = result.next_token_logits
            if logits is None or logits.ndim != 2:
                raise RuntimeError('Full model logits missing')
            torch.cuda.synchronize()
            reference = logits.detach().clone()
            if not bool(torch.isfinite(reference).all()):
                raise RuntimeError('Nonfinite actual Foundry replay baseline')
            report = {'phase': phase, 'rank': rank, 'status': 'running',
                      'graph_type': f'{type(graph).__module__}.{type(graph).__name__}',
                      'shape_key': repr(key), 'output_shape': list(logits.shape),
                      'capture_keys': [repr(k) for k in backend._graphs],
                      'baseline_logits_sha256': hashlib.sha256(reference.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                      'validation': [], 'timings': {}, 'mode': cfg['mode'], 'admission': dp_admission,
                      'initial_actual_replay_poisoned': True,
                      'diagnostic_only': bool(cfg.get('diagnostic_save_mismatch')),
                      'live_inputs': diagnostic_inputs,
                      'scope': 'Actual Foundry SAVE capture or independent LOAD graph replay at live SGLang decode boundary'}
            if backend_observer is not None:
                report['communication_backend'] = backend_observer[1](graph)
                write_json(out/'communication_backend.json', backend_observer[0].snapshot())
            cpu_reference = reference.cpu()
            reference_path = out / 'references' / f"batch_{phase['batch']:04d}.pt"
            reference_path.parent.mkdir(exist_ok=True)
            if cfg['mode'] == 'save' and not reference_path.exists():
                torch.save(cpu_reference, reference_path)
                report['saved_reference'] = str(reference_path)
            if cfg.get('save_reference'):
                saved_path = Path(cfg['save_reference']) / f'rank_{rank}' / 'references' / reference_path.name
                saved = torch.load(saved_path, map_location='cpu', weights_only=True)
                exact_saved = bool(torch.equal(cpu_reference, saved))
                report['cross_process_save_reference'] = {
                    'path': str(saved_path), 'bitwise': exact_saved,
                    'argmax': bool(torch.equal(cpu_reference.argmax(-1), saved.argmax(-1))),
                    'max_abs': float((cpu_reference.float() - saved.float()).abs().max().item()),
                    'elements': saved.numel()}
                if cfg.get('diagnostic_save_mismatch'):
                    from logit_diagnostics import compare_logits
                    report['cross_process_diagnostic'] = compare_logits(cpu_reference, saved, torch)
                    diagnostic_path = out / f"phase_{phase['id']:04d}_reference_tensors.pt"
                    torch.save({'actual': cpu_reference, 'save': saved,
                                'live_inputs': diagnostic_inputs}, diagnostic_path)
                    report['diagnostic_tensor_path'] = str(diagnostic_path)
                    # Every rank records before proceeding. Reference differences
                    # remain failures and cannot promote a diagnostic run to pass.
                    verdicts = [None] * world
                    torch.distributed.all_gather_object(verdicts, exact_saved, group=group)
                    report['all_rank_independent_save_matches'] = verdicts
                    report['independent_save_requirement_passed'] = all(verdicts)
                    write_json(out / f"phase_{phase['id']:04d}.json", report)
                elif not exact_saved:
                    write_json(out / f"phase_{phase['id']:04d}.json", report)
                    raise RuntimeError('Independent SAVE/LOAD complete logits differ')
            variants = {'actual': graph.replay}
            fresh = None
            if cfg['mode'] == 'load':
                from foundry import research_qmd
                research_qmd.retain_owners(backend, runner, tuple(backend._graphs.values()))
                state = research_qmd.state(graph)
                report['foundry_state'] = state
                report['foundry_state_before_actual_replay'] = state_before
                previous_member = (state_before.get('current_member_id', state_before.get('current_params_id'))
                                   if state_before else None)
                report['from_member_id'] = previous_member
                report['to_member_id'] = state['current_member_id']
                report['actual_source_switch'] = previous_member != state['current_member_id']
                prior_sequence = (state_before or {}).get('last_receipt', {}).get('sequence')
                receipt = state.get('last_receipt') or {}
                report['exec_update_this_replay'] = (receipt.get('event') == 'updated' and receipt.get('sequence') != prior_sequence)
                if report['actual_source_switch'] != report['exec_update_this_replay']:
                    raise RuntimeError('Member-switch evidence disagrees with actual update receipt')
                if not state or not state.get('candidate_exec'):
                    raise RuntimeError(f'No actual Foundry persistent candidate: {state}')
                handle = int(state['candidate_exec'])
                info = graph._research_info()
                report['foundry_info'] = info
                # Source already has the current member's reconstructed params.
                if int(info['current_params_id']) != int(info['graph_id']):
                    raise RuntimeError('Foundry shared source does not name live target')
                if driver is None:
                    driver = DriverGraph()
                fresh, fresh_instantiate_us = driver.instantiate(int(state['source_graph']), flags=int(state['initial_flags']), register=False)
                if fresh in (handle, int(info['template_exec'])):
                    raise RuntimeError('Fresh executable aliases candidate or original template')
                report['fresh_exec'] = fresh
                report['variant_execs'] = {'updated': handle, 'fresh': fresh, 'original_template': int(info['template_exec'])}
                report['fresh_instantiate_us'] = fresh_instantiate_us
                report['fresh_source_graph'] = int(state['source_graph'])
                stream = int(torch.cuda.current_stream().cuda_stream)
                driver.upload(fresh, stream)
                torch.cuda.synchronize()
                variants = {'updated': lambda: graph._research_replay_exec(handle),
                            'fresh': lambda: graph._research_replay_exec(fresh)}
            group = backend._tp_group.cpu_group
            def consensus(payload):
                world = torch.distributed.get_world_size(group)
                values = [None] * world
                torch.distributed.all_gather_object(values, payload, group=group)
                if any(v != values[0] for v in values[1:]):
                    raise RuntimeError(f'Rank stage disagreement {values}')
            def validate(name, fn, when):
                consensus((phase['id'], when, name))
                backend._tp_group.barrier()
                logits.fill_(float('nan'))
                fn()
                torch.cuda.synchronize()
                exact = bool(torch.equal(logits, reference))
                finite = bool(torch.isfinite(logits).all())
                maxabs = float((logits.float() - reference.float()).abs().max().item())
                argmax = bool(torch.equal(logits.argmax(-1), reference.argmax(-1)))
                verdict = torch.tensor([int(exact and finite and argmax)], device='cpu')
                torch.distributed.all_reduce(verdict, op=torch.distributed.ReduceOp.MIN, group=group)
                report['validation'].append({'variant': name, 'when': when, 'bitwise': exact,
                                             'finite': finite, 'argmax': argmax, 'max_abs': maxabs,
                                             'elements': logits.numel(), 'all_ranks_pass': bool(verdict[0])})
                write_json(out / f"phase_{phase['id']:04d}.json", report)
                if not bool(verdict[0]):
                    raise RuntimeError('Full logits mismatch')
            for name, fn in variants.items():
                validate(name, fn, 'before_timing')
            names = list(variants)
            random.Random(cfg['seed'] + phase['id']).shuffle(names)
            for block in range(cfg['blocks']):
                order = names if block % 2 == 0 else list(reversed(names))
                for position, name in enumerate(order):
                    consensus((phase['id'], 'timing', block, name))
                    backend._tp_group.barrier()
                    fn = variants[name]
                    for _ in range(3):
                        fn()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(cfg['launches']):
                        fn()
                    end.record()
                    end.synchronize()
                    sample = {'block': block, 'order': position,
                              'gpu_per_launch_us': start.elapsed_time(end) * 1000 / cfg['launches']}
                    report['timings'].setdefault(name, []).append(sample)
            for name, fn in variants.items():
                validate(name, fn, 'after_timing')
            variants[next(iter(variants))]()
            torch.cuda.synchronize()
            if fresh is not None:
                driver.destroy(fresh)
                report['paired_updated_vs_fresh'] = paired_ratios(report['timings'], seed=cfg['seed'] + phase['id'])
                report['foundry_state_after'] = research_qmd.state(graph)
            report['status'] = 'complete'
            write_json(out / f"phase_{phase['id']:04d}.json", report)
            print(f"FOUNDRY_INTEGRATION_PHASE rank={rank} id={phase['id']} bs={phase['batch']} complete", flush=True)
            return result
        except BaseException as exc:
            write_json(out / 'fail_stop.json', {'phase': phase, 'error': repr(exc)})
            print(f'FOUNDRY_INTEGRATION_FAIL_STOP rank={rank}: {exc!r}', flush=True)
            os._exit(86)
    FullCudaGraphBackend.replay = replay
    import sglang, foundry
    write_json(out / 'worker_hook.json', {'rank': rank, 'pid': os.getpid(),
              'sglang': sglang.__file__, 'foundry': foundry.__file__,
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda})


def scheduler_with_probe(*args, **kwargs):
    from sglang.srt.managers.scheduler import run_scheduler_process
    bound = inspect.signature(run_scheduler_process).bind(*args, **kwargs)
    install_worker_probe(int(bound.arguments['tp_rank']))
    return run_scheduler_process(*args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['save', 'load'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpu-uuids', required=True)
    ap.add_argument('--target-backend', choices=['single_gpu', 'symmem', 'deepep_dp'], default='single_gpu')
    ap.add_argument('--capture-batches', default='2,4,8,16,30,31,32,64')
    ap.add_argument('--sequence', default='2,4,2')
    ap.add_argument('--manifest-sequence', action='store_true')
    ap.add_argument('--save-reference', help='SAVE report directory: require exact independent-process logits and generated tokens')
    ap.add_argument('--diagnostic-save-mismatch', action='store_true',
                    help='Preserve cross-process mismatch tensors/inputs and continue diagnosis; never produces passed status, updated/fresh remains strict')
    ap.add_argument('--blocks', type=int, default=11)
    ap.add_argument('--launches', type=int, default=32)
    ap.add_argument('--prompt-tokens', type=int, default=64)
    ap.add_argument('--generate-tokens', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--mem-fraction-static', type=float, default=0.45)
    ap.add_argument('--foundry-config', help='Use an existing TOML instead of generated config')
    args = ap.parse_args()
    if args.mode == 'load' and not args.save_reference:
        raise ValueError('LOAD requires --save-reference for independent-process correctness')
    if args.save_reference:
        args.save_reference = str(Path(args.save_reference).resolve())
    out, archive = Path(args.output).resolve(), Path(args.archive).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metadata.json').exists():
        raise FileExistsError('Refusing report overwrite')
    capture = list(map(int, args.capture_batches.split(',')))
    sequence = list(map(int, args.sequence.split(',')))
    if args.mode == 'save':
        # Every captured shape receives one identical-input reference for LOAD.
        sequence = list(dict.fromkeys(capture + sequence))
    manifests = manifest_groups(archive) if args.mode == 'load' else None
    if args.manifest_sequence:
        if manifests is None:
            raise ValueError('Manifest sequence requires an actual SAVE archive')
        sequence = complete_pair_sequence(manifests[0]['groups'])
    if not sequence or not set(sequence) <= set(capture) or max(capture) > 128:
        raise ValueError('Invalid bounded batch selection')
    if not 1 <= args.blocks <= 31 or not 1 <= args.launches <= 128:
        raise ValueError('Timing budget out of bounds')
    tp = 1 if args.target_backend == 'single_gpu' else 2
    if len(args.gpu_uuids.split(',')) != tp:
        raise ValueError('GPU UUID count mismatch')
    venv = Path(sys.prefix).resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuids, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HOME=str(venv / '.cache/huggingface'), OMP_NUM_THREADS='2', MAX_JOBS='2',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', SGLANG_EARLY_FORKSERVER='0')
    cache = venv / '.cache/serving'
    for name, suffix in [('XDG_CACHE_HOME','xdg'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
                         ('SGLANG_CACHE_DIR','sglang'),('SGLANG_JIT_CACHE_DIR','sglang_jit'),('CUDA_CACHE_PATH','cuda'),
                         ('TMPDIR','tmp'),('FLASHINFER_WORKSPACE_BASE','flashinfer'),('TORCH_EXTENSIONS_DIR','extensions')]:
        path = venv / 'tmp' if name == 'TMPDIR' else cache / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['FOUNDRY_LAZY_GRAPH_EXEC'] = '1'
    config = Path(args.foundry_config).resolve() if args.foundry_config else out / 'foundry.toml'
    if not args.foundry_config:
        config.write_text(f'mode = "{args.mode}"\nbase_addr = 0x600000000000\nregion_size = "256GB"\n'
                          f'workspace_root = {json.dumps(str(archive))}\nscratch_space_size = "1024MB"\ngraph_templates = true\n')
    phase_file = out / 'active_phase.json'
    write_json(phase_file, {'armed': False})
    cfg = {**vars(args), 'output': str(out), 'phase_file': str(phase_file)}
    write_json(out / 'worker_config.json', cfg)
    os.environ['FOUNDRY_INTEGRATION_PROBE_CONFIG'] = str(out / 'worker_config.json')
    meta = {'args': vars(args), 'sequence': sequence, 'started': time.time(), 'status': 'initializing',
            'actual_save_manifests': manifests, 'gpu_before': gpu_state(),
            'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for uuid in args.gpu_uuids.split(','):
        if uuid in meta['gpu_before']['apps']:
            raise RuntimeError('GPU occupied before run')
    write_json(out / 'metadata.json', meta)
    from sglang.srt.entrypoints.engine import Engine
    Engine.run_scheduler_process_func = staticmethod(scheduler_with_probe)
    options = dict(model_path=args.model, load_format='dummy', dtype='bfloat16', skip_tokenizer_init=True,
                   mem_fraction_static=args.mem_fraction_static, max_running_requests=max(capture) * (2 if args.target_backend == 'deepep_dp' else 1),
                   context_length=2048, **prefill_defaults(args.target_backend), disable_radix_cache=True,
                   disable_overlap_schedule=True, attention_backend='fa3', random_seed=args.seed,
                   cuda_graph_backend_decode='full', cuda_graph_backend_prefill='disabled',
                   cuda_graph_bs_decode=sorted(capture), watchdog_timeout=180,
                   file_storage_path=str(out / 'sglang_storage'), crash_dump_folder=str(out / 'crash_dumps'),
                   cuda_graph_persistence=args.mode, cuda_graph_persistence_config=str(config), log_level='info')
    options.update(required_backend_options(args.target_backend))
    meta['engine_args'] = options
    write_json(out / 'metadata.json', meta)
    engine, checks = None, []
    try:
        engine = Engine(**options)
        meta['resolved_engine_config'] = resolved_prefill_snapshot(engine)
        write_json(out / 'metadata.json', meta)
        def generate(prompts, params, request_ids=None):
            if args.target_backend != 'deepep_dp':
                return engine.generate(input_ids=prompts, sampling_params=params, rid=request_ids)
            async def both():
                n = len(prompts)//2
                return await asyncio.gather(*(engine.async_generate(input_ids=prompts[r*n:(r+1)*n], sampling_params=params, routed_dp_rank=r, rid=request_ids[r*n:(r+1)*n] if request_ids else None) for r in range(2)))
            groups = engine.loop.run_until_complete(both())
            return [item for group in groups for item in (group if isinstance(group,list) else [group])]
        for index, batch in enumerate(sequence):
            count = batch * (2 if args.target_backend == 'deepep_dp' else 1)
            prompts = [[100 + (i+j+batch*17)%200 for j in range(args.prompt_tokens)] for i in range(count)]
            params = {'temperature':0,'max_new_tokens':args.generate_tokens,'ignore_eos':True}
            phase = {'armed':True,'id':index,'batch':batch,'global_batch':count}
            request_ids = ([f'foundry-b{batch}-p{index}-probe-{i}' for i in range(count)]
                           if args.diagnostic_save_mismatch else None)
            if request_ids is not None:
                phase['request_ids'] = request_ids
            write_json(phase_file, phase)
            actual = generation_ids(generate(prompts, params, request_ids))
            write_json(phase_file, {'armed':False})
            repeat_ids = ([f'foundry-b{batch}-p{index}-repeat-{i}' for i in range(count)]
                          if args.diagnostic_save_mismatch else None)
            repeated = generation_ids(generate(prompts, params, repeat_ids))
            covered = [(out / f'rank_{rank}' / f'phase_{index:04d}.json').exists() for rank in range(tp)]
            check = {'phase':phase,'tokens':actual,'repeated_tokens':repeated,'exact_match':actual == repeated,'rank_coverage':covered, 'input_ids':prompts}
            if args.save_reference:
                saved_checks = json.loads((Path(args.save_reference) / 'generation_checks.json').read_text())
                matches = [row for row in saved_checks if row['phase']['batch'] == batch and row['input_ids'] == prompts]
                if not matches:
                    raise RuntimeError(f'No matching SAVE inputs for batch {batch}')
                check['independent_save_tokens'] = matches[0]['tokens']
                check['independent_save_exact_match'] = matches[0]['tokens'] == actual
                if not check['independent_save_exact_match'] and not args.diagnostic_save_mismatch:
                    checks.append(check)
                    write_json(out/'generation_checks.json',checks)
                    raise RuntimeError('Independent SAVE/LOAD generation tokens differ')
            checks.append(check)
            write_json(out/'generation_checks.json',checks)
            if (actual != repeated and not args.diagnostic_save_mismatch) or not all(covered):
                raise RuntimeError(f'Generation mismatch/missing graph phase {index}')
        aggregate = []
        for index in range(len(sequence)):
            reports = [json.loads((out/f'rank_{rank}'/f'phase_{index:04d}.json').read_text()) for rank in range(tp)]
            if any(r['status'] != 'complete' for r in reports):
                raise RuntimeError('Incomplete worker phase')
            row = {'phase': index, 'batch':sequence[index], 'rank_reports':[f'rank_{r}/phase_{index:04d}.json' for r in range(tp)]}
            if args.mode == 'load':
                combined = {}
                for name in ('updated','fresh'):
                    combined[name] = [{'block':block,'gpu_per_launch_us':max(r['timings'][name][block]['gpu_per_launch_us'] for r in reports)} for block in range(args.blocks)]
                row.update(slowest_rank_timings=combined, paired_updated_vs_fresh=paired_ratios(combined,seed=args.seed+index))
            aggregate.append(row)
        write_json(out/'aggregate.json',{'phases':aggregate,'tp_size':tp,'mode':args.mode})
        meta['status']='diagnostic_complete_not_accepted' if args.diagnostic_save_mismatch else 'passed'
        if args.diagnostic_save_mismatch:
            meta['performance_acceptance']='not_evaluated_diagnostic_only'
        if args.mode == 'save':
            meta['actual_save_manifests']=manifest_groups(archive)
    except BaseException as exc:
        meta.update(status='failed',error=repr(exc))
        raise
    finally:
        write_json(phase_file, {'armed':False})
        if engine is not None:
            engine.shutdown()
        meta.update(finished=time.time(),gpu_after=gpu_state())
        write_json(out/'metadata.json',meta)

if __name__ == '__main__':
    main()
