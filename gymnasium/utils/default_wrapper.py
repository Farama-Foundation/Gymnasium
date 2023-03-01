from functools import wraps

def default_wrapper(wrapper_cls):
    @wraps(wrapper_cls, updated=())
    class DefaultWrapper(wrapper_cls):
        @property
        def spec(self):
            return self.env.spec
    
    return DefaultWrapper
                