from unia.general_agent import *
from env import *
from logging_init import *
import time as t

action_space = 4
state_shape = 8 # can be image (tupel) or int
max_time_steps = 2000 # if unnecessary, set it to a high value
update_freq = 10 # define target-network update frequency

build = ("LunarLander-v3")
logger.info(f"Using gym-environment: {build}")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Detected device: {device}")
t.sleep(1)
# usability

if device == "cpu":
    threads_num = int(input(f"Enter number of threads (available: {torch.get_num_threads()}): "))
    torch.set_num_threads(threads_num)
    logger.info(f"...Using {threads_num} cpu threads")

logger.info(f"Beginning the training process")

epochs = 100000 # use as many epochs as you want
env = Env(build, state_shape=state_shape)
agent = Agent(state_shape, action_space, device=device)
logger.info(f"Architecture: {agent.main_network}")

# agent.main_network = torch.load("breakout.pt") # optionally: load existing models
# agent.target_network = torch.load("breakout.pt")

for epoch in range(epochs):
    average_loss = 0
    average_rew = 0
    done = False

    if epoch % update_freq == 0 and epoch != 0: agent.target_network.load_state_dict(agent.main_network.state_dict())
    state = env.start_mdp()
    if env.shape != state_shape: logger.warn("Predefined state-shape does not match calculated state-shape.")

    for step in range(max_time_steps):
        if done:
            break
        action = agent.select_action(state)
        nstate, rew, done = env.step(action)

        agent.replay_buffer.push(state, torch.tensor([action]), rew, nstate, torch.Tensor([done]).long())

        state = nstate.clone()

        loss = agent.training_main()

        average_loss += loss
        average_rew += float(rew.clone())

    print(f"Average loss in epoch {epoch}: {average_loss/step}... and average reward in this epoch: {average_rew/step}, {agent.decay}... Got it for {step} steps")
    torch.save(agent.main_network, "lunar.pt")