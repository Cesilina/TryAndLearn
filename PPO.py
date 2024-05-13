

{
    'prompts': prompts,
    'input_ids': seq,
    "attention_mask": attention_mask,
    'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),        # batch_size  * (seq_len - 1)
    'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]), # batch_size  * (seq_len - 1)
    'value': values,                                                    # batch_size * seq_len
    'rewards': reward_score,                                            # torch.Size([batch_size])
}

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)






prompts = inputs['prompts']
log_probs = inputs['logprobs']
ref_log_probs = inputs['ref_logprobs']
reward_score = inputs['rewards']
values = inputs['value']
attention_mask = inputs['attention_mask']
seq = inputs['input_ids']
start = prompts.size()[-1] - 1
action_mask = attention_mask[:, 1:]
old_values = values
old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
ends = start + action_mask[:, start:].sum(1) + 1

# 计算reward
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
    kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
    rewards = kl_divergence_estimate
    start = prompts.shape[1] - 1
    ends = start + action_mask[:, start:].sum(1) + 1
    reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
    batch_size = log_probs.shape[0]
    for j in range(batch_size):
        rewards[j, start:ends[j]][-1] += reward_clip[j]

    return rewards




batch = {'input_ids': seq, "attention_mask": attention_mask}
advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

## 优势函数的返回
def get_advantages_and_returns(self, values, rewards, start):
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
        lastgaelam = delta + self.gamma * self.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, start:]
    return advantages.detach(), returns  #advantages 优势函数结果， returns表示Q函数




#更新Actor模型

batch = {'input_ids': seq, "attention_mask": attention_mask}
actor_prob = self.actor_model(**batch, use_cache=False).logits
actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
actor_loss = self.actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, action_mask[:, start:])
self.actor_model.backward(actor_loss)
self.actor_model.step()

## loss的计算
def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
    ## policy gradient loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss


# 更新critic模型


value = self.critic_model.forward_value(**batch, return_value_only=True, use_cache=False)[:, :-1]
critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,start:], returns, action_mask[:, start:])
self.critic_model.backward(critic_loss)
self.critic_model.step()

## loss的计算
def critic_loss_fn(self, values, old_values, returns, mask):
    values_clipped = torch.clamp(values, old_values - self.cliprange_value, old_values + self.cliprange_value)
    if self.compute_fp32_loss:
        values = values.float()
        values_clipped = values_clipped.float()
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss