"""Reuse strict communication-call observations with Foundry graph identities.

Foundry SAVE does not expose keep_graph=True, so capture receipts identify the
actual Python graph and archive shape, never mislabel an object ID as CUgraph.
"""
import copy
import importlib
from backend_assertions import (BackendAssertions, _install_on_classes,
                                _install_torch_distributed)


def install(target, strict_bank=False):
    import torch
    group = importlib.import_module('sglang.srt.distributed.parallel_state')
    symmem = importlib.import_module('sglang.srt.distributed.device_communicators.torch_symm_mem')
    pynccl = importlib.import_module('sglang.srt.distributed.device_communicators.pynccl')
    deep = importlib.import_module('deep_ep') if target == 'deepep_dp' else None
    full = importlib.import_module('sglang.srt.model_executor.runner_backend.full_cuda_graph_backend')
    state = BackendAssertions(target == 'symmem', target == 'deepep_dp', True)
    for module in (group, symmem, pynccl, deep, full):
        if module is not None:
            state._module_source(module)
    _install_on_classes(state, torch, group.GroupCoordinator, symmem.TorchSymmMemCommunicator,
                        pynccl.PyNcclCommunicator, deep.Buffer if deep is not None else None)
    _install_torch_distributed(state, torch.distributed, group)
    dispatch_capacities = []
    if strict_bank and deep is not None:
        dispatch_original = deep.Buffer.low_latency_dispatch
        def dispatch(instance, *args, **kwargs):
            hidden = args[0] if args else kwargs['x']
            capacity = args[2] if len(args)>2 else kwargs['num_max_dispatch_tokens_per_rank']
            row = {'tokens':int(hidden.shape[0]),'capacity':int(capacity)}
            if row['tokens']>row['capacity']:
                raise RuntimeError(f'DeepEP actual dispatch exceeds capacity: {row}')
            if state._capturing(): dispatch_capacities.append(row)
            return dispatch_original(instance,*args,**kwargs)
        deep.Buffer.low_latency_dispatch = dispatch
    original = full.FullCudaGraphBackend.capture_one
    receipts = {}
    def capture(backend, shape_key, *args, **kwargs):
        capacity_start = len(dispatch_capacities)
        before = [counter.copy() for counter in (state.counts, state.details, state.violations)]
        result = original(backend, shape_key, *args, **kwargs)
        graph = backend._graphs[shape_key]
        receipt = {'python_graph_id': id(graph), 'graph_type': type(graph).__module__,
                   'shape_key': repr(shape_key), 'batch': int(shape_key.size),
                   'identity_kind': 'Foundry SAVE Python graph and archive shape'}
        for name, now, old in zip(('counts', 'details', 'violations'),
                                  (state.counts, state.details, state.violations), before):
            receipt[name] = dict(sorted((now-old).items()))
        errors = state._errors(receipt['counts'], receipt['violations'])
        if strict_bank and deep is not None:
            receipt['low_latency_dispatch_capacities']=dispatch_capacities[capacity_start:]
            if not receipt['low_latency_dispatch_capacities'] or any(c['capacity']<receipt['batch'] for c in receipt['low_latency_dispatch_capacities']):
                errors.append('Missing/insufficient observed DeepEP dispatch capacity for bank scope')
        receipt['errors'] = errors
        receipts[id(graph)] = receipt
        return result
    full.FullCudaGraphBackend.capture_one = capture
    def observed(graph):
        receipt = receipts[id(graph)]
        if receipt['errors']:
            raise RuntimeError(f'Foundry SAVE communication backend rejected: {receipt}')
        return copy.deepcopy(receipt)
    return state, observed
