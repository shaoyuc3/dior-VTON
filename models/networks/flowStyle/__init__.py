# model_init
import importlib
from .afwm import *
from .networks import load_checkpoint

def init_net(net, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        
        net = net.cuda()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net

def define_tool_networks(opt, load_ckpt_path="", gpu_ids=[]):
    net = AFWM(opt, 45)
    
    #import pdb; pdb.set_trace()
    #net.eval()
    #net = net.cuda()
    #net = load_checkpoint(net, opt.flownet_path)
    #net.to(gpu_ids[0])
    #return torch.nn.DataParallel(net, gpu_ids)
    
    import pdb; pdb.set_trace()
    if load_ckpt_path:
        ckpt = torch.load(load_ckpt_path)
        net.load_state_dict(ckpt, strict=False)
        print("load ckpt from %s."%load_ckpt_path)
        return net
    else:
        return init_net(net, gpu_ids)