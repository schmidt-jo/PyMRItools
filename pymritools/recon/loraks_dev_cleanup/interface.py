from abc import ABC, abstractmethod
from typing import Optional, Any
from pymritools.recon.loraks_dev_cleanup.loraks import LoraksImplementation, LoraksOptions
from pymritools.recon.loraks_dev_cleanup.ac_loraks import AcLoraks
from pymritools.recon.loraks_dev_cleanup.p_loraks import PLoraks


# class Loraks:
#     @staticmethod
#     def create(implementation: Optional[LoraksImplementation] = None, options: LoraksOptions = LoraksOptions()):
#         """Factory method to instantiate the appropriate LORAKS implementation"""
#
#         recon = None
#         match implementation:
#             case None:
#                 # Choose the implementation based on option characteristics
#                 # TODO: should we do a small AC region estimate,
#                 #  like try to find a central region and based on its size compared to the data size, choose AC vs P?
#                 if options.fast_compute == ComputationType.REGULAR:
#                     recon = PLoraks()
#                 else:
#                     recon = AcLoraks()
#             case LoraksImplementation.P_LORAKS:
#                 recon = PLoraks()
#             case LoraksImplementation.AC_LORAKS:
#                 recon = AcLoraks()
#         if recon is None:
#             raise RuntimeError("This should never happen. Please report this issue to the developers.")
#         recon.configure(options)
#         return recon


class AcLoraksOptions(LoraksOptions):
    def __init__(self):
        super().__init__()
        # Additional options specific to AC Loraks
        self.ac_specific_param1: Optional[float] = None
        self.ac_specific_param2: bool = False


class PLoraksOptions(LoraksOptions):
    def __init__(self):
        super().__init__()
        # Additional options specific to P Loraks
        self.p_specific_param1: Optional[int] = None
        self.p_specific_param2: str = "default"


class LoraksBase(ABC):
    @abstractmethod
    def configure(self, options: LoraksOptions) -> None:
        """
        Configure the Loraks implementation with given options

        Can accept the base options or implementation-specific options
        """
        pass

    @abstractmethod
    def reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        """Perform reconstruction"""
        pass


class Loraks:
    @staticmethod
    def create(
            implementation: Optional[LoraksImplementation] = None,
            options: LoraksOptions = LoraksOptions()
    ) -> LoraksBase:
        """Factory method with support for implementation-specific options"""
        if implementation is None:
            implementation = (
                LoraksImplementation.P_LORAKS
            )

        implementation_map = {
            LoraksImplementation.P_LORAKS: PLoraks,
            LoraksImplementation.AC_LORAKS: AcLoraks
        }

        recon = implementation_map.get(implementation)()
        recon.configure(options)
        return recon

