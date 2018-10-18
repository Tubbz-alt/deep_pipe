"""Tools for grouping DICOM metadata into images."""
import pandas as pd


def _remove_dots(x):
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def aggregate_images(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Groups DICOM ``metadata`` into images (series).

    Required columns: PatientID, SeriesInstanceUID, StudyInstanceUID, PathToFolder, FileName.

    Notes
    -----
    The following columns are added:
        | SlicesCount: the number of files/slices in the image.
        | FileNames: a list of slash ("/") separated file names.
        | InstanceNumbers: (if InstanceNumber is in columns) a list of comma separated InstanceNumber values.

    The following columns are removed:
        FileName (replaced by FileNames), InstanceNumber (replaced by InstanceNumbers),
        any other columns that differ from file to file.
    """

    def get_unique_cols(df):
        return [col for col in df.columns if len(df[col].dropna().unique()) == 1]

    def process_group(entry):
        # TODO: should check that some typical cols are unique, e.g. ImageOrientationPatient*
        res = entry.iloc[[0]][get_unique_cols(entry)]
        res['FileNames'] = '/'.join(entry.FileName)
        res['SlicesCount'] = len(entry)
        # entries sometimes have no `InstanceNumber`
        # TODO: probably partially sorted slices will also do
        # TODO: detect duplicates
        try:
            res['InstanceNumbers'] = ','.join(map(_remove_dots, entry.InstanceNumber))
        except ValueError:
            res['InstanceNumbers'] = None

        return res.drop(['InstanceNumber', 'FileName'], 1, errors='ignore')

    # TODO: move constants out. Make PixelArrayShape required
    group_by = ['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID', 'PathToFolder']
    optional = {'SequenceName', 'PixelArrayShape'}
    group_by.extend(optional & set(metadata))

    not_string = metadata[group_by].applymap(lambda x: not isinstance(x, str)).any()
    if not_string.any():
        not_strings = ', '.join(not_string.index[not_string])
        raise ValueError(f'The following columns do not contain only strings: {not_strings}. '
                         'You should probably check for NaN values.')

    return metadata.groupby(group_by).apply(process_group).reset_index(drop=True)


def normalize_identifiers(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Converts PatientID to str and fills nan values in SequenceName.

    Notes
    -----
    The input dataframe will be mutated.
    """
    metadata['PatientID'] = metadata.PatientID.apply(_remove_dots)
    if 'SequenceName' in metadata:
        metadata.SequenceName.fillna('', inplace=True)
    return metadata


def select(dataframe: pd.DataFrame, query: str, **where: str) -> pd.DataFrame:
    query = ' '.join(query.format(**where).splitlines())
    return dataframe.query(query).dropna(axis=1, how='all').dropna(axis=0, how='all')