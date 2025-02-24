import torch
import torch.nn as nn

class ControllerModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        # Controller 生成状态更新规则
        self.controller = nn.Linear(2 * hidden_dim, 3 * hidden_dim)
        # 预测层
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, prev_state):
        x_embed = self.embed(x)
        # Controller 输入：当前嵌入 + 历史状态
        control = self.controller(torch.cat([x_embed, prev_state], dim=-1))
        gate, reset, candidate = torch.split(control, hidden_dim, dim=-1)
        # 动态状态更新
        reset_gate = torch.sigmoid(reset)
        candidate = torch.tanh(reset_gate * prev_state + x_embed)
        update_gate = torch.sigmoid(gate)
        new_state = (1 - update_gate) * prev_state + update_gate * candidate
        # 预测下一 Token
        logits = self.output(new_state)
        return logits, new_state
