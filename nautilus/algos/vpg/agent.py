from nautilus.algos.ppo.agent import PPOAgent


class VPGAgent(PPOAgent):
    """
    Vanilla Policy Gradient (Actor-Critic style).
    - No Clipping
    - Single Update Step per batch (usually)
    - Prone to catastrophic collapse if Learning Rate is too high
    """

    def update_params(self, batch_tensors: dict):
        obs = batch_tensors["obs"]
        act = batch_tensors["act"]
        ret = batch_tensors["ret"]
        adv = batch_tensors["adv"]

        # VPG typically does a single pass over the data
        # (Multiple passes without clipping usually leads to explosion)
        for _ in range(1):
            self.pi_optimizer.zero_grad()
            self.v_optimizer.zero_grad()

            # 1. Policy Loss (Standard REINFORCE/A2C style)
            # L = - mean( log_prob * advantage )
            dist = self.ac.pi(obs)
            log_prob = dist.log_prob(act)

            # Handle continuous action summing
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(1)

            # Basic Policy Gradient Loss
            loss_pi = -(log_prob * adv).mean()

            # 2. Value Loss (MSE)
            # VPG doesn't clip value updates either
            pred_val = self.ac.v(obs)
            loss_v = ((pred_val - ret) ** 2).mean()

            # 3. Update
            loss_pi.backward()
            loss_v.backward()

            self.pi_optimizer.step()
            self.v_optimizer.step()

        # Store metrics for logging
        self.latest_metrics = {
            "loss_pi": loss_pi.item(),
            "loss_v": loss_v.item(),
        }
