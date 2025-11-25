from typing import Optional

class FlexKVError(Exception):
    def __init__(self, message: str = "", error_type: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.error_type is not None:
            return f"[{self.error_type}] {self.message}"
        return self.message

class InvalidConfigError(FlexKVError):
    def __init__(self, message: str = ""):
        super().__init__(message, "Invalid config")

class LogicError(FlexKVError):
    def __init__(self, message: str = ""):
        super().__init__(message, "Logic error")

class TransferError(FlexKVError):
    def __init__(self, message: str = "", src: Optional[str] = None, dst: Optional[str] = None):
        self.src = src
        self.dst = dst
        if src or dst:
            message = f"{message} (Source: {src or '-'}, Destination: {dst or '-'})"
        super().__init__(message, "Transfer failed")

class NotEnoughSpaceError(FlexKVError):
    def __init__(self, message: str = "", required: Optional[int] = None, available: Optional[int] = None):
        self.required = required
        self.available = available
        if required is not None and available is not None:
            message = f"{message} (Required: {required}, Available: {available})"
        super().__init__(message, "Not enough space")

class TimeOutError(FlexKVError):
    def __init__(self, message: str = "", timeout: Optional[int] = None):
        self.timeout = timeout
        if timeout is not None:
            message = f"{message} (Timeout: {timeout})"
        super().__init__(message, "Time out")

class HashCollisionError(FlexKVError):
    def __init__(self, message: str = ""):
        super().__init__(message, "Hash collision")
