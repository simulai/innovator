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
