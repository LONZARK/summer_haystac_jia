from haystac_kw.data.schemas.hos import InternalHOS
from haystac_kw.data.schemas.has import InternalHAS
from pathlib import Path, PurePath
from typing import Dict


class HASGenerator:
    """Base class for an HAS generation."""

    def __init__(
            self,
            source_data_dir: str,
            internal_hos_dir: str,
            output_dir: str):
        """
        Class Constructor

        Parameters
        ----------
        train_data_dir : str
            Path to directory containing simulation training data and metadata
            about the simulation. The file structure is assumed to match that
            of [this directory](https://drive.google.com/drive/folders/1SNwgfu-IDYIsMWc4gqt8xafC9h0DU3iF)
            with the zip files decompressed.
        internal_hos_dir : str
            Path to directory containing HOS jsons
        output_dir : str
            Path to directory to save out HAS jsons
        """
        # Directory to save Interal HAS files to
        self.output_dir = Path(output_dir)

        # Directory containing training data and metadata for
        # simulation.
        self.source_data_dir = source_data_dir

        # Load HOS files into memory
        self.injest_internal_hos(internal_hos_dir)

    def injest_internal_hos(self, internal_hos_dir: str) -> None:
        """
        Load HOS into memory from a directory of JSON
        files.

        Parameters
        ----------
        internal_hos_dir : str
            Directory containing HOS JSON files
        """

        # Get list of jsons
        hos_files = Path(internal_hos_dir).glob('*.json')

        # Parse each HOS and store in dictonary
        self._hos = {}
        for hos_file in hos_files:
            fname = PurePath(hos_file).stem
            self._hos[fname] = InternalHOS.from_json(hos_file.read_text())

    @property
    def hos(self) -> Dict[str, InternalHOS]:
        """
        Dictonary of HOS Data Structures

        Returns
        -------
        Dict[str, InternalHOS]
            HOS dictionary
        """
        return self._hos

    def preprocess(self) -> None:
        """
        Optional Preprocessing Method
        """
        pass

    def train(self) -> None:
        """
        Optional Training Method
        """
        pass

    def validate_output(self) -> bool:
        """
        Validate that the HAS files satisfy the HOS
        files. Currently this just validates that there
        is a HAS for each HOS and that they can be loaded
        from a JSON.

        Returns
        -------
        bool
            All HAS satisfy their corresponding HOS
        """
        for hos in self.hos.keys():
            try:
                InternalHAS.from_json(Path(self.output_dir,
                                           hos + '.json').read_text())

                # TODO Add logic that actually validates
                # that the HAS satisfies the HOS.

            except Exception as e:
                return False
        return True

    def generate_has(self) -> None:
        """
        Run Activity Injection for all HOS
        """
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Run preprocessing steps if any
        self.preprocess()
        # Run training steps if any
        self.train()

        # Create All HAS files
        for hos_name, has in self._generate_has().items():

            # Create HAS for single HOS
            assert isinstance(has, InternalHAS)
            has_json = has.model_dump_json()
            Path(self.output_dir, hos_name + '.json').write_text(has_json)

        # Run HAS validation
        assert self.validate_output()

    def _generate_has(self) -> Dict[str, InternalHAS]:
        """
        Create HAS for all HOS files

        Returns
        -------
        Dict[str, InternalHAS]
            HAS satisfying the Hide Objectives with the keys
            being the HOS filename without json extension or
            directory.

        Raises
        ------
        NotImplementedError
            Subclasses should implement this
        """
        raise NotImplementedError("Subclasses should implement this.")
