from typing import Optional
from pymritools.recon.loraks_dev_cleanup.loraks import LoraksImplementation, LoraksOptions, ComputationType
from pymritools.recon.loraks_dev_cleanup.ac_loraks import AcLoraks
from pymritools.recon.loraks_dev_cleanup.p_loraks import PLoraks


class Loraks:
    @staticmethod
    def create(implementation: Optional[LoraksImplementation] = None, options: LoraksOptions = LoraksOptions()):
        """Factory method to instantiate the appropriate LORAKS implementation"""

        recon = None
        match implementation:
            case None:
                # Choose the implementation based on option characteristics
                # TODO: should we do a small AC region estimate,
                #  like try to find a central region and based on its size compared to the data size, choose AC vs P?
                if options.fast_compute == ComputationType.REGULAR:
                    recon = PLoraks()
                else:
                    recon = AcLoraks()
            case LoraksImplementation.P_LORAKS:
                recon = PLoraks()
            case LoraksImplementation.AC_LORAKS:
                recon = AcLoraks()
        if recon is None:
            raise RuntimeError("This should never happen. Please report this issue to the developers.")
        recon.configure(options)
        return recon
