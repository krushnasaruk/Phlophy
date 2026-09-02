"""
Abstract Hardware Controller Interface for physical autonomous chessboard deployment.
Decouples high-level reinforcement learning move decisions from mechanical kinematics and actuators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class RobotMoveCommand:
    """Standardized physical move command emitted by RL agent."""
    from_square_uci: str
    to_square_uci: str
    is_capture: bool = False
    promotion: Optional[str] = None
    cartesian_from_mm: Optional[Tuple[float, float]] = None
    cartesian_to_mm: Optional[Tuple[float, float]] = None
    speed_percentage: float = 100.0


@dataclass
class MoveExecutionStatus:
    """Feedback from robotic physical actuators and sensors."""
    success: bool
    execution_time_sec: float
    error_message: Optional[str] = None
    sensor_confirmed: bool = True
    sensor_metadata: Optional[Dict[str, Any]] = None


class BaseHardwareController(ABC):
    """
    Abstract Base Class for autonomous chessboard physical controllers.
    Supports Cartesian Gantry, SCARA, CoreXY, and Robotic Arm mechanisms.
    """

    @abstractmethod
    def connect(self, port: str = "COM3", baudrate: int = 115200) -> bool:
        """Establish serial/USB communication with microcontroller / motor drivers."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Safely close connection and power down motors."""
        pass

    @abstractmethod
    def home_axes(self) -> bool:
        """Execute limit-switch homing sequence."""
        pass

    @abstractmethod
    def execute_move(self, command: RobotMoveCommand) -> MoveExecutionStatus:
        """
        Execute physical move by translating algebraic/UCI notation into motor step pulses
        and gripper / electromagnet state changes.
        """
        pass

    @abstractmethod
    def query_board_state(self) -> Optional[str]:
        """
        Query magnetic sensor matrix or overhead vision camera to detect physical piece locations.
        Returns FEN string or square occupancy representation if available.
        """
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediately cut motor power and halt all motion."""
        pass
