import mesa
from model import FairfaxABM


def get_bev_owners(model):
    """
    Display a text count of how many BEV owning agents there are.
    """
    return f"EV Owners: {model.bevs}"

def draw(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}

    if agent.type == 'charger':
        portrayal = {"Shape": "rect", "w": 1.0, "h": 1.0, "Filled": "true", "Layer": 0}
        portrayal["Color"] = ["#000000", "#000000"]
        portrayal["stroke_color"] = "#000000"
    else:
        if agent.type == 'bev_owner':
            portrayal["Color"] = ["#008000", "#008000"]
            portrayal["stroke_color"] = "#008000"
        else:
            portrayal["Color"] = ["#808080", "#808080"]
            portrayal["stroke_color"] = "#808080"
    return portrayal



canvas_element = mesa.visualization.CanvasGrid(draw, 29, 29, 500, 500)
bev_chart = mesa.visualization.ChartModule([{"Label": "bevs", "Color": "Black"}])

model_params = {
    "height": 29,
    "width": 29,
    "rand_or_gis": mesa.visualization.Slider("Random (0) or GIS (1) Placement", 0, 0, 1, 1),
}

server = mesa.visualization.ModularServer(
    FairfaxABM,
    [canvas_element, get_bev_owners, bev_chart],
    "FairfaxABM",
    model_params,
)
