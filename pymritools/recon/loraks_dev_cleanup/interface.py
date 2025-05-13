from abc import ABC, abstractmethod
from typing import Optional, Any
from pymritools.recon.loraks_dev_cleanup.loraks import LoraksImplementation, ComputationType
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


class LoraksOptions:
    def __init__(self):
        # Base options common to all Loraks implementations
        self.fast_compute: ComputationType = ComputationType.REGULAR


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


class AcLoraks(LoraksBase):
    def configure(self, options: AcLoraksOptions) -> None:
        """
        Configure AC Loraks with AC-specific options

        Accepts AcLoraksOptions which inherits from LoraksOptions
        """
        # Handle base options
        self.fast_compute = options.fast_compute

        # Handle AC-specific options
        self.ac_specific_param1 = options.ac_specific_param1
        self.ac_specific_param2 = options.ac_specific_param2

    def reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        # AC Loraks specific reconstruction
        pass


class PLoraks(LoraksBase):
    def configure(self, options: PLoraksOptions) -> None:
        """
        Configure P Loraks with P-specific options

        Accepts PLoraksOptions which inherits from LoraksOptions
        """
        # Handle base options
        self.fast_compute = options.fast_compute

        # Handle P-specific options
        self.p_specific_param1 = options.p_specific_param1
        self.p_specific_param2 = options.p_specific_param2

    def reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        # P Loraks specific reconstruction
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
                if options.fast_compute == ComputationType.REGULAR
                else LoraksImplementation.AC_LORAKS
            )

        implementation_map = {
            LoraksImplementation.P_LORAKS: PLoraks,
            LoraksImplementation.AC_LORAKS: AcLoraks
        }

        recon = implementation_map.get(implementation)()
        recon.configure(options)
        return recon

# Use P-specific options
p_options = PLoraksOptions()
p_loraks = Loraks.create(
    implementation=LoraksImplementation.P_LORAKS,
    options=p_options
)
