from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TextClassificationInput(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class TextClassificationOutput(_message.Message):
    __slots__ = ("is_toxic",)
    IS_TOXIC_FIELD_NUMBER: _ClassVar[int]
    is_toxic: bool
    def __init__(self, is_toxic: bool = ...) -> None: ...
