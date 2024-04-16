from src.optimization import optimize_single_objective, optimize_double_objective
from src.parameters import OPT_ALPHA, OPT_MIN_AP
from src.bologna.city import City

class BeamsManager:
    objective_func_val = None
    selected_beams = None
    beams_math_ap = None

    def __init__(self, city, visbility_graph):
        self.city = city
        self.visbility_graph = visbility_graph

    def select_beams(self, opt_alpha=OPT_ALPHA, min_ap=OPT_MIN_AP):
        if opt_alpha == 0:
            self.selected_beams, self.objective_func_val, self.beams_math_ap = optimize_single_objective(self.city,
                                                                                                         min_ap=min_ap)
        else:
            self.selected_beams, self.objective_func_val, self.beams_math_ap = \
                optimize_double_objective(self.city, min_ap=min_ap, alpha=opt_alpha)



if __name__ == "__main__":
    city = City()