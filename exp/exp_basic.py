from models import timer, timer_xl, moirai, moment, ttm
try:
    from models import gpt4ts, time_llm, autotimes
except Exception:  # Optional deps (e.g., transformers) may be missing
    gpt4ts = None
    time_llm = None
    autotimes = None


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "timer": timer,
            "timer_xl": timer_xl,
            "moirai": moirai,
            "moment": moment,
            "ttm": ttm,
        }
        if gpt4ts is not None:
            self.model_dict["gpt4ts"] = gpt4ts
        if time_llm is not None:
            self.model_dict["time_llm"] = time_llm
        if autotimes is not None:
            self.model_dict["autotimes"] = autotimes
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
