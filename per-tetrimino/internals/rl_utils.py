"""
Utility functions for reinforcement learning.
"""

# Function to sync target network
def update_target_model(net, target_net, tau=1.0):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data
        )
        
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def capped_cubic_video_schedule(episode_id: int) -> bool:
    if episode_id < 100:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    return episode_id % 100 == 0
