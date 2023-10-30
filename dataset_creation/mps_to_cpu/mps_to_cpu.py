import os
#from softq import *
from softq_models import *
import torch

# Save model parameters
def save(q_net, path, suffix=""):
    critic_path = f"{path}{suffix}"
    #print('Saving models to {} and {}'.format(path, critic_path))
    torch.save(q_net.state_dict(), critic_path)

# Load model parameters
def load(q_net, path, suffix=""):
    #critic_path = f'{path}/{self.args.agent.name}{suffix}'
    critic_path = path
    print('Loading models from {}'.format(critic_path))
    state_dict = torch.load(critic_path, map_location="cpu")
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("_orig_mod"):
            new_key = key[10:]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    q_net.load_state_dict(new_state_dict)
    return q_net
        
def main():
    models = os.listdir("ToNoam")
    obs_dim = (4, 20, 10)
    act_dim = 6
    args = None
    output_dir = "ToNoamCPU"
    device = "cpu"
    for model_name in models:
        if model_name.startswith("."):
            continue    
        model_path = os.path.join("ToNoam", model_name)
        # find the number after the seond occurance of v
        print(model_name)
        v_index = model_name.find("v", 10)
        #print(model_name[v_index:])
        version_number = int(model_name[v_index+1])

        if version_number == 1:
            q_net = TetrisQNetwork(obs_dim, act_dim, args, device)
        elif version_number == 2:
            q_net = TetrisQNetwork2(obs_dim, act_dim, args, device)
        elif version_number == 3:
            q_net = TetrisQNetwork3(obs_dim, act_dim, args, device)
        elif version_number == 4:
            q_net = TetrisQNetwork4(obs_dim, act_dim, args, device)
        else:
            raise ValueError("version number not supported")
        
        #torch.compile(q_net)
        q_net = load(q_net, model_path)
        q_net = q_net.cpu()
        output_path = os.path.join(output_dir, model_name)
        save(q_net, output_path)
        print("saved model to ", output_path)
        
if __name__ == "__main__":
    main()
        
        