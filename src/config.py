from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 5000
    grpc_port: int = 9090
    grpc_inference_port: int = 50051

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5000")),
            grpc_port=int(os.getenv("GRPC_PORT", "9090")),
            grpc_inference_port=int(os.getenv("GRPC_INFERENCE_PORT", "50051")),
        )
