import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Union

class BaseModel:
    """Base class for all models with serialization capabilities."""

    @abstractmethod
    def predict(self, X):
        pass

    def serialize(self, file_path: Union[str, Path]) -> None:
        """
        Serialize the model to a file using pickle.

        Args:
            file_path: Path where the model will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(cls, file_path: Union[str, Path]) -> 'BaseModel':
        """
        Deserialize a model from a file.

        Args:
            file_path: Path to the serialized model.

        Returns:
            The deserialized model instance.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)