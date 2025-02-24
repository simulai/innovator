class ControllerRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Controller 模块（轻量级网络）
        self.controller = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 3 * hidden_dim)  # 输出控制信号
        )
        # 状态更新和预测模块
        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, prev_state):
        # 拼接当前输入与历史状态
        controller_input = torch.cat([x, prev_state], dim=-1)
        # 生成控制信号（拆分为三个部分）
        gate, candidate, reset = torch.split(
            self.controller(controller_input), 
            hidden_dim, 
            dim=-1
        )
        # 动态状态更新（类似 GRU，但参数由 Controller 生成）
        reset_signal = torch.sigmoid(reset)
        candidate_state = torch.tanh(self.state_proj(x) + reset_signal * prev_state
        new_state = (1 - torch.sigmoid(gate)) * prev_state + torch.sigmoid(gate) * candidate_state
        # 生成预测
        output = self.output_layer(new_state)
        return output, new_state


# 示例：Controller 驱动的缓存读写
class MemoryController(nn.Module):
    def __init__(self, dim, num_slots):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(num_slots, dim))
        self.controller = nn.Linear(dim, 2 * dim)  # 生成读/写信号

    def forward(self, x):
        # 生成读/写权重
        read_weight, write_weight = torch.split(
            self.controller(x), 
            dim, 
            dim=-1
        )
        # 从记忆库读取
        read = torch.matmul(torch.softmax(read_weight, dim=-1), self.memory)
        # 更新记忆库
        self.memory = self.memory + torch.matmul(
            torch.softmax(write_weight, dim=-1).unsqueeze(-1), 
            x.unsqueeze(1)
        )
        return read


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

# 使用示例
model = ControllerModel(vocab_size=10000, hidden_dim=512)
x = torch.randint(0, 10000, (32, 1))  # batch_size=32, seq_len=1
state = torch.zeros(32, 512)
logits, new_state = model(x, state)
