class Generator:
    def __init__(self, model):
        self.model = model

    def generate_response(self, context, input_text):
        combined_input = f"{context} {input_text}"
        response = self.model.generate(combined_input)
        return response

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model