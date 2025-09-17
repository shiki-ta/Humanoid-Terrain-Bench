import os, sys
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import Actor,get_activation
import argparse
import code
import shutil

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareVisionNN(nn.Module):
    def __init__(self,  num_prop,
                        num_scan,
                        num_actions,
                        tanh,
                        scan_encoder_dims = [128, 64, 32],
                        actor_hidden_dims = [512, 256, 128],
                        activation='elu',
                        ):
        super(HardwareVisionNN, self).__init__()
        
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_actions = num_actions
        num_obs = num_prop + num_scan
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, 
                          actor_hidden_dims, activation, tanh_encoder_output=tanh)
        
    def forward(self, obs):
        return self.actor(obs, eval=False)

def play(args):    
    load_run = "../../logs/parkour_new/" + args.exptid
    checkpoint = args.checkpoint

    num_scan = 132
    num_actions = 12
    n_proprio = 46

    device = torch.device('cpu')
    policy = HardwareVisionNN(n_proprio, num_scan, num_actions,args.tanh).to(device)
    
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    
    ac_state_dict = torch.load(load_path, map_location=device)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    
    policy = policy.to(device)
    
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    policy.eval()
    with torch.no_grad(): 
        num_envs = 1
        obs_input = torch.ones(num_envs, n_proprio + num_scan, device=device)
        
        test = policy(obs_input)
        traced_policy = torch.jit.trace(policy, (obs_input,))
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
        traced_policy.save(save_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)