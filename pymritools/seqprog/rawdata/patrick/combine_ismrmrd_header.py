"""Functions to combine ismrmrd data files and meta files."""

from pathlib import Path

import ismrmrd


def read_ismrmrd_dataset(fname: Path) -> tuple:
    """Read ismrmrd dataset from file.

    Parameters
    ----------
    fname
        file path to the ismrmrd dataset.

    Returns
    -------
        ismrmrd header and list of acquisitions.
    """
    with ismrmrd.File(str(fname), 'r') as file:
        ds = file[list(file.keys())[-1]]
        header = ds.header
        acqs = ds.acquisitions[:]

    return header, acqs


def insert_traj_from_meta(
    data_acqs: list[ismrmrd.acquisition.Acquisition],
    meta_acqs: list[ismrmrd.acquisition.Acquisition],
) -> list[ismrmrd.acquisition.Acquisition]:
    """
    Insert trajectory information from the meta file into the data file.

    Parameters
    ----------
    data_acqs : list
        list of acquisitions from the data file.
    meta_acqs : list
        list of acquisitions from the meta file.

    Returns
    -------
    list of acquisitions with the trajectory information from the meta file.
    """
    if not len(data_acqs) == len(meta_acqs):
        raise ValueError('Number of acquisitions in data and meta file do not match.')

    for i, (acq_d, acq_m) in enumerate(zip(data_acqs, meta_acqs, strict=False)):
        if not acq_d.number_of_samples == acq_m.number_of_samples:
            raise ValueError(f'Number of samples in acquisition {i} do not match.')

        # insert trajectory information from meta file
        acq_d.resize(
            number_of_samples=acq_d.number_of_samples,
            active_channels=acq_d.active_channels,
            trajectory_dimensions=acq_m.trajectory_dimensions,
        )
        acq_d.traj[:] = acq_m.traj[:]
        data_acqs[i] = acq_d

    return data_acqs


def update_header_from_meta(
    data_header: ismrmrd.xsd.ismrmrdHeader,
    meta_header: ismrmrd.xsd.ismrmrdHeader,
    enc_idx: int = 0,
) -> ismrmrd.xsd.ismrmrdHeader:
    """
    Update the header of the data file with the information from the meta file.

    Parameters
    ----------
    data_header : ismrmrd.xsd.ismrmrdHeader
        header of the ismrmrd data file.
    meta_header : ismrmrd.xsd.ismrmrdHeader
        header of the ismrmrd meta file created with the seq-file.

    Returns
    -------
    ismrmrd.xsd.ismrmrdHeader
        Updated header.
    """
    # overwrite encoded matrix and fov
    if meta_header.encoding[enc_idx].encodedSpace.matrixSize is not None:
        data_header.encoding[enc_idx].encodedSpace.matrixSize = meta_header.encoding[enc_idx].encodedSpace.matrixSize
    if meta_header.encoding[enc_idx].encodedSpace.fieldOfView_mm is not None:
        data_header.encoding[enc_idx].encodedSpace.fieldOfView_mm = meta_header.encoding[
            enc_idx
        ].encodedSpace.fieldOfView_mm

    # overwrite recon matrix and fov
    if meta_header.encoding[enc_idx].reconSpace.matrixSize is not None:
        data_header.encoding[enc_idx].reconSpace.matrixSize = meta_header.encoding[enc_idx].reconSpace.matrixSize
    if meta_header.encoding[enc_idx].reconSpace.fieldOfView_mm is not None:
        data_header.encoding[enc_idx].reconSpace.fieldOfView_mm = meta_header.encoding[
            enc_idx
        ].reconSpace.fieldOfView_mm

    # overwrite trajectory type
    if meta_header.encoding[enc_idx].trajectory is not None:
        data_header.encoding[enc_idx].trajectory = meta_header.encoding[enc_idx].trajectory

    return data_header


def combine_ismrmrd_files(data_file: Path, meta_file: Path, filename_ext: str = '_with_traj.h5'):
    """Combine ismrmrd data file and meta file.

    Parameters
    ----------
    data_file
        path to the ismrmrd data file
    meta_file
        path to the ismrmrd meta file
    filename_ext, optional
        filename extension of the output file, by default '_with_traj.h5'

    Returns
    -------
        combined ismrmrd file from data and meta file.
    """
    filename_out = data_file.parent / (data_file.stem + filename_ext)

    data_header, data_acqs = read_ismrmrd_dataset(data_file)
    meta_header, meta_acqs = read_ismrmrd_dataset(meta_file)

    new_acqs = insert_traj_from_meta(data_acqs, meta_acqs)
    new_header = update_header_from_meta(data_header, meta_header)

    # Create new file
    ds = ismrmrd.Dataset(filename_out)
    ds.write_xml_header(new_header.toXML())

    # add acquisitions with trajectory information
    for acq in new_acqs:
        ds.append_acquisition(acq)

    ds.close()

    return ds
