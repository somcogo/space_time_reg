from src.models.nodeo import BrainNet
from src.models.siren import Siren, SirenT, SirenLateT, SirenEnsemble, GroupedSiren
from src.models.wire import WireReal, WireRealT, WireRealLateT

def get_func(func_name, network_kwargs):
    if func_name == 'nodeo':
        func = BrainNet(**network_kwargs)
    elif func_name == 'siren':
        func = Siren(**network_kwargs)
    elif func_name == 'sirent':
        func = SirenT(**network_kwargs)
    elif func_name == 'sirenlatet':
        func = SirenLateT(**network_kwargs)
    elif func_name == 'sirenensemble':
        func = SirenEnsemble(**network_kwargs)
    elif func_name == 'groupsiren':
        func = GroupedSiren(**network_kwargs)
    elif func_name == 'wire':
        func = WireReal(**network_kwargs)
    elif func_name == 'wiret':
        func = WireRealT(**network_kwargs)
    elif func_name == 'wirelatet':
        func = WireRealLateT(**network_kwargs)

    return func