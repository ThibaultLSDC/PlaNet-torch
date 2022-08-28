import torch
import wandb

def bottle(func, *tensors) -> torch.Tensor:
    n, t = tensors[0].shape[:2]
    out = func(*[x.contiguous().view((n*t, *x.shape[2:])) for x in tensors])
    return out.view(n, t, *out.shape[1:])


class RunningMetric:
    def __init__(self,
        name: str
        ) -> None:
        self.name = name
        self.value = 0.
        self.steps = 0
    
    def step(self, new_value):
        self.value += new_value
        self.steps += 1

    def terminate(self):
        res = self.showcase()
        self.__init__(self.name)
        return res

    def showcase(self):
        return self.value / (self.steps + 1e-9)


class ImageMetric:
    def __init__(self,
        name: str
        ) -> None:
        self.name = name
        self.value = None
    
    def step(self, value):
        self.value = value
    
    def terminate(self):
        res = self.value
        self.__init__(self.name)
        return wandb.Video(res)


class Tracker:
    def __init__(self, list_recurrent_tracks: list[str], list_img_tracks: list[str]) -> None:
        for metric in list_recurrent_tracks:
            setattr(self, metric, RunningMetric(metric))

        for metric in list_img_tracks:
            setattr(self, metric, ImageMetric(metric))

        self.metrics = list_img_tracks + list_recurrent_tracks
        self.list_onetime_tracks = list_img_tracks
        self.list_recurrent_tracks = list_recurrent_tracks

    def step(self, metrics: dict):
        for key in metrics.keys():
            getattr(self, key).step(metrics[key])
    
    def terminate(self):
        return {metric: getattr(self, metric).terminate() for metric in self.metrics}

    def showcase(self, showcase_list):
        return {metric: getattr(self, metric).showcase() for metric in showcase_list}