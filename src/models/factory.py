from src.models.nodeo import BrainNet
from src.models.siren import Siren, GroupedSiren
from src.models.wire import WireReal, WireRealT, WireRealLateT

def get_func(func_name, network_kwargs):
    if func_name == 'nodeo':
        func = BrainNet(**network_kwargs)
    elif func_name == 'siren':
        # Single network shared by all frame intervals: one static velocity field over
        # the whole time period.
        func = Siren(**network_kwargs)
    elif func_name == 'groupsiren':
        # One independent Siren per frame interval (non-stationary velocity).
        func = GroupedSiren(**network_kwargs)
    elif func_name == 'wire':
        func = WireReal(**network_kwargs)
    elif func_name == 'wiret':
        func = WireRealT(**network_kwargs)
    elif func_name == 'wirelatet':
        func = WireRealLateT(**network_kwargs)
    else:
        raise ValueError(f'Unknown func_name: {func_name}')

    return func
