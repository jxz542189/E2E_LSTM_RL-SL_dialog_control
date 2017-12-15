class Config:
    def __init__(self):
        self.actions = ['api_call', 'update_call', 'display_options', 'extra_info', 'full_dialog']
        self.embedding_size = 300
        self.d_state_size = 200
        self.u_state_size = 150
        self.h1 = 100
        self.optmzr = 'adam_1e-3'
        self.dropout = 0.5
        self.max_turn = 15
        self.success_reward = 15
        self.step_penalty = -1
        self.init_lr = 1e-3
        self.n_slots = 10
