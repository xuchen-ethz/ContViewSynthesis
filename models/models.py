
def create_model(opt):
    from .view_synthesis_model import ViewSynthesisModel
    model = ViewSynthesisModel()

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
