class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, inputs, outputs):
        # self.inputs = inputs
        self.outputs = outputs
        
    def clear(self):
        self.hook.remove()