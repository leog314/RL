from unia.general_agent import *
from env import *
from logging_init import *

action_space = 3
state_shape = 2
max_time_steps = 2000

build = "MountainCar-v0"
logger.info(f"Using gym-environment: {build}")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Detected device: {device}")
if device == "cpu":
    threads_num = int(input(f"Enter number of threads (available: {torch.get_num_threads()}): "))
    torch.set_num_threads(threads_num)
    logger.info(f"Using {threads_num} cpu threads")

epochs = 100000
env = Env(build, state_shape=state_shape)
agent = Agent(state_shape, action_space, device=device)
logger.info(f"Architecture: {agent.main_network}")

# agent.main_network = torch.load("breakout.pt")
# agent.target_network = torch.load("breakout.pt")

for epoch in range(epochs):
    average_loss = 0
    average_rew = 0
    done = False

    if epoch % 20 == 0 and epoch != 0: agent.target_network.load_state_dict(agent.main_network.state_dict()) # add update frequency optionally
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

    print(f"Average loss in epoch {epoch}: {average_loss/step}... and average reward in this epoch: {average_rew/step}, {agent.decay}")
    torch.save(agent.main_network, "backgammon_transfer.pt")