import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

ckpt_path = "rl_finetune/model_epoch_220_Coffee_D1_success_0.9.pth"
device = TorchUtils.get_torch_device(try_to_use_cuda=True)




import torch.nn.functional as F

def gmm_rsample_with_log_prob(dist):
    """
    Reparameterized sample from a MixtureSameFamily GMM.
    Uses Gumbel-softmax to differentiably select a component,
    then rsamples from that component's Gaussian.
    """
    mixture = dist.mixture_distribution      # Categorical over 5 modes
    components = dist.component_distribution # 5 Gaussians over action_dim=7

    # Gumbel-softmax: differentiable one-hot over the 5 modes
    # hard=True gives a one-hot in the forward pass, but gradient flows through soft version
    logits = mixture.logits  # (B, 5)
    gumbel_weights = F.gumbel_softmax(logits, tau=1.0, hard=True)  # (B, 5)

    # rsample from ALL 5 components simultaneously: (B, 5, 7)
    all_samples = components.rsample()  # gradient flows through this

    # Select the chosen component: (B, 7)
    sample = (gumbel_weights.unsqueeze(-1) * all_samples).sum(dim=-2)

    # log_prob is already implemented on MixtureSameFamily
    log_prob = dist.log_prob(sample)  # (B,) — sum over action dims handled internally

    return sample, log_prob


# Load the policy
policy, ckpt_dict = FileUtils.policy_from_checkpoint(
    ckpt_path=ckpt_path,
    device=device,
    verbose=True
)

# Pull out the actual network (not the wrapper)
bc_algo = policy.policy
actor = bc_algo.nets["policy"]

print("Actor type:", type(actor))
print("\nActor subnetworks:")
for k, v in actor.nets.items():
    print(f"  {k}: {type(v)}")

print("\nObs encoder output shape:", actor.nets["encoder"].output_shape())

# Check LSTM
rnn = actor.nets["rnn"]
print("\nLSTM:", rnn.nets)   # should show LSTM(137, 1000, num_layers=2)



print("ckpt_dict keys:", list(ckpt_dict.keys()))
env_meta = ckpt_dict["env_metadata"]
# env = EnvUtils.create_env_from_metadata(env_meta, render=False, render_offscreen=False)
env = EnvUtils.create_env_from_metadata(
    env_meta, 
    render=False, 
    render_offscreen=True,   # this is what enables image obs
    use_image_obs=True       # explicitly request images
)

obs = env.reset()
print("\nObs keys:", list(obs.keys()))
print("Obs shapes:", {k: v.shape for k, v in obs.items()})

policy.start_episode()
action = policy(obs)
print("\nAction shape:", action.shape)

# forward_train check
obs_tensor = {
    k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    for k, v in obs.items()
}

actor.train()

out = actor.forward_train(obs_tensor, rnn_init_state=None, return_state=True)

print("\nforward_train output type:", type(out))
if isinstance(out, dict):
    print("keys:", out.keys())
else:
    print("out[0] type:", type(out[0]))
    print("out[1] type:", type(out[1]))

# Test it
dist = out[0]
sample, log_prob = gmm_rsample_with_log_prob(dist)

# Squeeze the seq dimension out — (1,1,7) -> (1,7), (1,1) -> (1,)
sample = sample.squeeze(1)
log_prob = log_prob.squeeze(1)

print("Sample shape:", sample.shape)    # expect (1, 7)
print("Log prob shape:", log_prob.shape) # expect (1,)
print("Log prob value:", log_prob)       # should be finite

# Now gradient check works
log_prob.mean().backward()
print("Gradient check passed")
