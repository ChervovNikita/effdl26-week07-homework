
import logging

from concurrent import futures

import grpc
import inference_pb2
import inference_pb2_grpc

from src.model import ToxicityModel
from src.config import Settings

class InferenceClassifier(inference_pb2_grpc.TextClassifierServicer):
    def __init__(self):
        self.model = ToxicityModel()
        self.model.load()

    def Predict(self, request, context):
        pred = self.model.predict(request.text)
        return inference_pb2.TextClassificationOutput(is_toxic=pred)


def serve():
    settings = Settings.from_env()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_TextClassifierServicer_to_server(InferenceClassifier(), server)
    server.add_insecure_port(f'[::]:{settings.grpc_inference_port}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    print("start serving...")
    serve()
