class BaseParameters(dict):
    def __call__(self, *args, **kwargs):
        ...

class ChunkerParameters(BaseParameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TripletExtractorParameters(BaseParameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RerankerParameters(BaseParameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GeneratorParameters(BaseParameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GraphParameters(BaseParameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




