import pickle
import numpy as np
from mlserver import MLModel, types


class XGBoostModel(MLModel):
    async def load(self):
        self._model = pickle.load(open(self._settings.parameters.uri, "rb"))
        return True

    async def predict(
        self, payload: types.InferenceRequest
    ) -> types.InferenceResponse:
        inputs = payload.inputs
        model_input = np.array(inputs[0].data)

        model_output = self._model.predict(model_input)

        output_data = model_output.tolist()
        return types.InferenceResponse(
            model_name=self._settings.name,
            model_version=self._settings.version,
            outputs=[
                types.ResponseOutput(
                    name="output",
                    shape=model_output.shape,
                    datatype="FP32",
                    data=output_data,
                )
            ],
        )
