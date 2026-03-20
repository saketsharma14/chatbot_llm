class Memory:
    def __init__(self, max_turns=3):
        self.history = []
        self.max_turns = max_turns

    def add(self, user_input, response):
        self.history.append((user_input, response))

    def get_recent(self):
        return self.history[-self.max_turns:]
