import torch


def bottle(func, *tensors) -> torch.Tensor:
    n, t = tensors[0].shape[:2]
    out = func(*[x.contiguous().view((n*t, *x.shape[2:])) for x in tensors])
    return out.view(n, t, *out.shape[1:])


# class Metric:
#     def __init__(self,
#         name: str
#         ) -> None:
#         self.name = name

#         self.values = []
    
#     def reset(self):
#         self.__init__(self.name)
    
#     def store(self, value):


# class Tracker:
#     def __init__(self) -> None:
#         pass