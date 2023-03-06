class Function:
    def __init__(self, kind, mem_size, run_time, warm_time):
        self.kind = kind
        self.mem_size = mem_size
        self.run_time = run_time
        self.warm_time = warm_time

    def __eq__(self, other):
        if isinstance(other, Function):
            return self.kind == other.kind

    def __repr__(self):
        return str((self.kind, self.mem_size, self.run_time, self.warm_time))