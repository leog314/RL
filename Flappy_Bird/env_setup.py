import rl_agent as rl

def obs_to_tensor(inp) -> rl.torch.Tensor:
    img = rl.torch.from_numpy(rl.np.array(inp))
    return img.permute((2, 1, 0))

agent = rl.Agent()
print(sum([p.numel() for p in agent.parameters()]))
env = rl.gym.make("FlappyBird-v0", render_mode="human")
epochs = 1000

for epoch in range(epochs):
    done = False
    aloss = 0.
    arew = 0.
    stes = 0.

    init_obs = env.reset()
    data = obs_to_tensor(init_obs[0])/255.
    # print(agent(data).shape)
    env.render()

    while not done:
        action = agent.select_action(data)
        obs, rew, done, _, _ = env.step(action)

        ndata = obs_to_tensor(obs)/255.

        agent.mem.push(data, rl.torch.Tensor([action]), rl.torch.Tensor([rew]), ndata, rl.torch.Tensor([done]).long())

        data = ndata.clone()

        # loss = agent.training_()

        # aloss += loss
        arew += float(rew)
        stes += 1

    for loop in range(1):
        loss = agent.training_()
        aloss += loss


    print(f"Loss in epoch {epoch}: {aloss/10} | Average reward in epoch {epoch}: {arew/stes}")
    print(agent.decay)
    rl.torch.save(agent, "rl_flappy_bird.pt")



